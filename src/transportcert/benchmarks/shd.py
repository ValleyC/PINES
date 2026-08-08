from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from ..artifacts import array_hash, code_revision, sha256_file, sha256_json, write_json_immutable
from ..models import DenseRecurrentSNN
from ..protocol import assert_disjoint_splits, deterministic_partition


@dataclass(frozen=True)
class SHDPreprocessConfig:
    schema_version: str = "SHDPreprocess/v1"
    time_bins: int = 50
    duration_seconds: float = 1.4
    input_channels: int = 700
    binary_frames: bool = True


@dataclass(frozen=True)
class SHDTrainConfig:
    schema_version: str = "SHDTrain/v1"
    hidden_size: int = 128
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    tau_mem: float = 5.0
    threshold: float = 1.0
    recurrent_scale: float = 0.25
    gradient_clip: float = 1.0


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("SHD training requires transportcert[torch]") from error
    return torch


class PackedSHD:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.packed = np.asarray(data["packed"], dtype=np.uint8)
            self.labels = np.asarray(data["labels"], dtype=np.int64)
            self.sample_ids = np.asarray(data["sample_ids"]).astype(str)
            self.metadata = json.loads(str(data["metadata"]))
        self.time_bins = int(self.metadata["config"]["time_bins"])
        self.input_channels = int(self.metadata["config"]["input_channels"])
        if self.packed.shape[:2] != (len(self.labels), self.time_bins):
            raise ValueError("packed SHD artifact shape is inconsistent with metadata")

    def frames(self, indices: np.ndarray | list[int]) -> np.ndarray:
        selected = self.packed[np.asarray(indices)]
        return np.unpackbits(
            selected, axis=-1, count=self.input_channels, bitorder="little"
        ).astype(np.float32)

    @property
    def data_hash(self) -> str:
        return sha256_file(self.path)


def preprocess_shd(
    raw_h5: str | Path,
    output_path: str | Path,
    split_name: str,
    config: SHDPreprocessConfig,
) -> Path:
    raw_h5 = Path(raw_h5)
    output_path = Path(output_path)
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as existing:
            metadata = json.loads(str(existing["metadata"]))
        expected = {
            "schema_version": config.schema_version,
            "config": asdict(config),
            "raw_sha256": sha256_file(raw_h5),
            "split": split_name,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"existing preprocess artifact differs at {key}")
        return output_path
    if config.time_bins <= 1 or config.duration_seconds <= 0:
        raise ValueError("invalid SHD temporal discretization")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_h5, "r") as source:
        labels = np.asarray(source["labels"], dtype=np.int64)
        packed_width = math.ceil(config.input_channels / 8)
        packed = np.zeros(
            (len(labels), config.time_bins, packed_width), dtype=np.uint8
        )
        event_counts = np.zeros(len(labels), dtype=np.int64)
        clipped_counts = np.zeros(len(labels), dtype=np.int64)
        for index in range(len(labels)):
            times = np.asarray(source["spikes/times"][index], dtype=np.float64)
            units = np.asarray(source["spikes/units"][index], dtype=np.int64)
            bins = np.floor(
                times * config.time_bins / config.duration_seconds
            ).astype(np.int64)
            valid = (
                (bins >= 0)
                & (bins < config.time_bins)
                & (units >= 0)
                & (units < config.input_channels)
            )
            frame = np.zeros(
                (config.time_bins, config.input_channels), dtype=np.uint8
            )
            frame[bins[valid], units[valid]] = 1
            packed[index] = np.packbits(frame, axis=-1, bitorder="little")
            event_counts[index] = len(times)
            clipped_counts[index] = int(np.count_nonzero(~valid))
            if (index + 1) % 1000 == 0:
                print(f"preprocess {split_name}: {index + 1}/{len(labels)}", flush=True)
    sample_ids = np.asarray(
        [f"shd-{split_name}-{index:05d}" for index in range(len(labels))]
    )
    metadata = {
        "schema_version": config.schema_version,
        "config": asdict(config),
        "raw_path": str(raw_h5.resolve()),
        "raw_sha256": sha256_file(raw_h5),
        "split": split_name,
        "samples": len(labels),
        "packed_hash": array_hash(packed),
        "labels_hash": array_hash(labels),
        "sample_ids_hash": sha256_json(sample_ids.tolist()),
        "total_events": int(np.sum(event_counts)),
        "events_outside_horizon": int(np.sum(clipped_counts)),
    }
    with output_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            packed=packed,
            labels=labels,
            sample_ids=sample_ids,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    return output_path


class _SurrogateSpike:
    @staticmethod
    def apply(value: Any) -> Any:
        torch = _torch()

        class SpikeFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, membrane_minus_threshold):
                ctx.save_for_backward(membrane_minus_threshold)
                return (membrane_minus_threshold >= 0).to(
                    membrane_minus_threshold.dtype
                )

            @staticmethod
            def backward(ctx, gradient):
                (value,) = ctx.saved_tensors
                return gradient / (1.0 + 10.0 * torch.abs(value)) ** 2

        return SpikeFunction.apply(value)


def build_torch_srnn(
    input_size: int,
    hidden_size: int,
    output_size: int,
    tau_mem: float,
    threshold: float,
    recurrent_scale: float,
) -> Any:
    torch = _torch()

    class TrainableSRNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_weights = torch.nn.Parameter(
                torch.empty(input_size, hidden_size)
            )
            self.recurrent_weights = torch.nn.Parameter(
                torch.empty(hidden_size, hidden_size)
            )
            self.output_weights = torch.nn.Parameter(
                torch.empty(hidden_size, output_size)
            )
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.register_buffer("tau_mem", torch.full((hidden_size,), tau_mem))
            self.register_buffer("threshold", torch.full((hidden_size,), threshold))
            torch.nn.init.xavier_uniform_(self.input_weights)
            torch.nn.init.orthogonal_(self.recurrent_weights)
            with torch.no_grad():
                self.recurrent_weights.mul_(recurrent_scale)
            torch.nn.init.xavier_uniform_(self.output_weights)

        def forward(self, events):
            batch = events.shape[0]
            voltage = torch.zeros(
                (batch, hidden_size), dtype=events.dtype, device=events.device
            )
            spikes = torch.zeros_like(voltage)
            logits = torch.zeros(
                (batch, output_size), dtype=events.dtype, device=events.device
            )
            for step in range(events.shape[1]):
                current = (
                    events[:, step] @ self.input_weights
                    + spikes @ self.recurrent_weights
                    + self.bias
                )
                voltage = voltage + (-voltage + current) / self.tau_mem
                spikes = _SurrogateSpike.apply(voltage - self.threshold)
                voltage = voltage - spikes * self.threshold
                logits = logits + spikes @ self.output_weights
            return logits

    return TrainableSRNN()


def _seed_everything(seed: int) -> None:
    torch = _torch()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def evaluate_torch_model(
    model: Any,
    store: PackedSHD,
    indices: np.ndarray,
    batch_size: int,
    device: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    torch = _torch()
    predictions: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_indices in _batches(indices, batch_size):
            events = torch.as_tensor(store.frames(batch_indices), device=device)
            batch_logits = model(events)
            logits.append(batch_logits.cpu().numpy())
            predictions.append(torch.argmax(batch_logits, dim=1).cpu().numpy())
    all_predictions = np.concatenate(predictions)
    all_logits = np.concatenate(logits)
    accuracy = float(np.mean(all_predictions == store.labels[indices]))
    return accuracy, all_predictions, all_logits


def train_shd_seed(
    train_store: PackedSHD,
    test_store: PackedSHD,
    output_dir: str | Path,
    seed: int,
    config: SHDTrainConfig,
    partition_salt: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    torch = _torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_manifest.json"
    model_path = output_dir / "model.npz"
    checkpoint_path = output_dir / "checkpoint.pt"
    if manifest_path.exists() or model_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"training output already exists: {output_dir}")
    _seed_everything(seed)
    splits = deterministic_partition(
        train_store.sample_ids,
        {"train": 0.8, "repair_calibration": 0.1, "certificate_audit": 0.1},
        partition_salt,
    )
    assert_disjoint_splits(splits.values())
    id_to_index = {
        sample_id: index for index, sample_id in enumerate(train_store.sample_ids)
    }
    train_indices = np.asarray(
        [id_to_index[item] for item in splits["train"].sample_ids], dtype=np.int64
    )
    calibration_indices = np.asarray(
        [id_to_index[item] for item in splits["repair_calibration"].sample_ids],
        dtype=np.int64,
    )
    audit_indices = np.asarray(
        [id_to_index[item] for item in splits["certificate_audit"].sample_ids],
        dtype=np.int64,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_srnn(
        train_store.input_channels,
        config.hidden_size,
        20,
        config.tau_mem,
        config.threshold,
        config.recurrent_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    generator = np.random.default_rng(seed)
    for epoch in range(config.epochs):
        model.train()
        shuffled = generator.permutation(train_indices)
        total_loss = 0.0
        correct = 0
        seen = 0
        epoch_started = time.perf_counter()
        for batch_indices in _batches(shuffled, config.batch_size):
            events = torch.as_tensor(train_store.frames(batch_indices), device=device)
            targets = torch.as_tensor(
                train_store.labels[batch_indices], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(events)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_indices)
            correct += int(torch.sum(torch.argmax(logits, dim=1) == targets))
            seen += len(batch_indices)
        epoch_record = {
            "epoch": epoch + 1,
            "loss": total_loss / seen,
            "train_accuracy": correct / seen,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_record)
        print(
            f"seed={seed} epoch={epoch + 1}/{config.epochs} "
            f"loss={epoch_record['loss']:.4f} "
            f"train_acc={epoch_record['train_accuracy']:.4f} "
            f"seconds={epoch_record['seconds']:.2f}",
            flush=True,
        )
    reference = DenseRecurrentSNN(
        input_weights=model.input_weights.detach().cpu().double().numpy(),
        recurrent_weights=model.recurrent_weights.detach().cpu().double().numpy(),
        output_weights=model.output_weights.detach().cpu().double().numpy(),
        bias=model.bias.detach().cpu().double().numpy(),
        threshold=model.threshold.detach().cpu().double().numpy(),
        tau_mem=model.tau_mem.detach().cpu().double().numpy(),
        reset_value=np.zeros(config.hidden_size),
        name=f"shd-srnn-seed-{seed}",
    )
    reference.save(model_path)
    test_indices = np.arange(len(test_store.labels), dtype=np.int64)
    train_accuracy, _, _ = evaluate_torch_model(
        model, train_store, train_indices, config.batch_size, device
    )
    calibration_accuracy, _, _ = evaluate_torch_model(
        model, train_store, calibration_indices, config.batch_size, device
    )
    audit_accuracy, _, _ = evaluate_torch_model(
        model, train_store, audit_indices, config.batch_size, device
    )
    test_accuracy, test_predictions, _ = evaluate_torch_model(
        model, test_store, test_indices, config.batch_size, device
    )
    checkpoint = {
        "schema_version": "SHDCheckpoint/v1",
        "state_dict": model.state_dict(),
        "model_hash": reference.model_hash,
        "seed": seed,
        "config": asdict(config),
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "splits": {name: split.split_hash for name, split in splits.items()},
        "code_revision": code_revision(repository_root),
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    split_indices_path = output_dir / "split_indices.npz"
    with split_indices_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            train=train_indices,
            repair_calibration=calibration_indices,
            certificate_audit=audit_indices,
            test=test_indices,
        )
    test_predictions_path = output_dir / "reference_test_predictions.npz"
    with test_predictions_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            predictions=test_predictions,
            sample_ids=test_store.sample_ids,
        )
    manifest = {
        "schema_version": "SHDTrainingManifest/v1",
        "seed": seed,
        "config": asdict(config),
        "preprocess": train_store.metadata["config"],
        "model_hash": reference.model_hash,
        "checkpoint_hash": sha256_file(checkpoint_path),
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "partition_salt": partition_salt,
        "split_counts": {name: len(split.sample_ids) for name, split in splits.items()},
        "split_hashes": {name: split.split_hash for name, split in splits.items()},
        "history": history,
        "final_accuracy": {
            "train": train_accuracy,
            "repair_calibration": calibration_accuracy,
            "certificate_audit": audit_accuracy,
            "sequestered_test": test_accuracy,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "code_revision": code_revision(repository_root),
    }
    write_json_immutable(manifest_path, manifest)
    return manifest


def load_experiment_config(path: str | Path) -> tuple[SHDPreprocessConfig, SHDTrainConfig, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != "SHDExperiment/v1":
        raise ValueError("unsupported SHD experiment config")
    preprocess = SHDPreprocessConfig(**raw["preprocess"])
    train = SHDTrainConfig(**raw["training"])
    return preprocess, train, raw

