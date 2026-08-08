from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..artifacts import array_hash, code_revision, sha256_file, sha256_json, write_json_immutable
from ..models import DenseRecurrentSNN
from ..protocol import assert_disjoint_splits, deterministic_partition
from .shd import _SurrogateSpike


@dataclass(frozen=True)
class NMNISTPreprocessConfig:
    schema_version: str = "NMNISTPreprocess/v1"
    time_bins: int = 30
    duration_microseconds: int = 310_000
    sensor_width: int = 34
    sensor_height: int = 34
    polarities: int = 2
    binary_frames: bool = True

    @property
    def input_channels(self) -> int:
        return self.sensor_width * self.sensor_height * self.polarities


@dataclass(frozen=True)
class NMNISTTrainConfig:
    schema_version: str = "NMNISTTrain/v1"
    hidden_size: int = 128
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    tau_mem: float = 5.0
    threshold: float = 1.0
    gradient_clip: float = 1.0


class PackedNMNIST:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.packed = np.asarray(data["packed"], dtype=np.uint8)
            self.labels = np.asarray(data["labels"], dtype=np.int64)
            self.sample_ids = np.asarray(data["sample_ids"]).astype(str)
            self.metadata = json.loads(str(data["metadata"]))
        config = self.metadata["config"]
        self.time_bins = int(config["time_bins"])
        self.input_channels = int(
            config["sensor_width"] * config["sensor_height"] * config["polarities"]
        )

    def frames(self, indices: np.ndarray | list[int]) -> np.ndarray:
        selected = self.packed[np.asarray(indices)]
        return np.unpackbits(
            selected, axis=-1, count=self.input_channels, bitorder="little"
        ).astype(np.float32)

    @property
    def data_hash(self) -> str:
        return sha256_file(self.path)


def preprocess_nmnist(
    raw_root: str | Path,
    output_path: str | Path,
    split_name: str,
    config: NMNISTPreprocessConfig,
) -> Path:
    from tonic.datasets import NMNIST

    raw_root = Path(raw_root)
    output_path = Path(output_path)
    dataset = NMNIST(str(raw_root), train=split_name == "train")
    source_archive = raw_root / "NMNIST" / (
        "train.zip" if split_name == "train" else "test.zip"
    )
    source_hash = sha256_file(source_archive)
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as existing:
            metadata = json.loads(str(existing["metadata"]))
        if metadata.get("source_archive_sha256") != source_hash or metadata.get(
            "config"
        ) != asdict(config):
            raise ValueError("existing N-MNIST preprocessing artifact differs")
        return output_path
    packed_width = (config.input_channels + 7) // 8
    packed = np.zeros(
        (len(dataset), config.time_bins, packed_width), dtype=np.uint8
    )
    labels = np.asarray(dataset.targets, dtype=np.int64)
    sample_ids = []
    clipped = 0
    total_events = 0
    for index in range(len(dataset)):
        events, label = dataset[index]
        bins = (
            events["t"].astype(np.int64)
            * config.time_bins
            // config.duration_microseconds
        )
        channels = (
            (events["p"].astype(np.int64) * config.sensor_height + events["y"])
            * config.sensor_width
            + events["x"]
        )
        valid = (
            (bins >= 0)
            & (bins < config.time_bins)
            & (channels >= 0)
            & (channels < config.input_channels)
        )
        frame = np.zeros((config.time_bins, config.input_channels), dtype=np.uint8)
        frame[bins[valid], channels[valid]] = 1
        packed[index] = np.packbits(frame, axis=-1, bitorder="little")
        clipped += int(np.count_nonzero(~valid))
        total_events += len(events)
        relative = Path(dataset.data[index]).relative_to(raw_root / "NMNIST")
        sample_ids.append(f"nmnist-{split_name}-{relative.as_posix()}")
        if int(label) != labels[index]:
            raise ValueError("N-MNIST label mismatch")
        if (index + 1) % 5000 == 0:
            print(f"preprocess N-MNIST {split_name}: {index + 1}/{len(dataset)}", flush=True)
    sample_ids_array = np.asarray(sample_ids)
    metadata = {
        "schema_version": config.schema_version,
        "config": asdict(config),
        "split": split_name,
        "samples": len(dataset),
        "source_archive": str(source_archive.resolve()),
        "source_archive_sha256": source_hash,
        "packed_hash": array_hash(packed),
        "labels_hash": array_hash(labels),
        "sample_ids_hash": sha256_json(sample_ids),
        "total_events": total_events,
        "events_outside_horizon": clipped,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            packed=packed,
            labels=labels,
            sample_ids=sample_ids_array,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    return output_path


def build_feedforward_snn(
    input_size: int,
    hidden_size: int,
    output_size: int,
    tau_mem: float,
    threshold: float,
) -> Any:
    import torch

    class FeedforwardSNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_weights = torch.nn.Parameter(torch.empty(input_size, hidden_size))
            self.output_weights = torch.nn.Parameter(torch.empty(hidden_size, output_size))
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.register_buffer("tau_mem", torch.full((hidden_size,), tau_mem))
            self.register_buffer("threshold", torch.full((hidden_size,), threshold))
            torch.nn.init.xavier_uniform_(self.input_weights)
            torch.nn.init.xavier_uniform_(self.output_weights)

        def forward(self, events):
            voltage = torch.zeros(
                (events.shape[0], hidden_size), dtype=events.dtype, device=events.device
            )
            logits = torch.zeros(
                (events.shape[0], output_size), dtype=events.dtype, device=events.device
            )
            for step in range(events.shape[1]):
                current = events[:, step] @ self.input_weights + self.bias
                voltage = voltage + (-voltage + current) / self.tau_mem
                spikes = _SurrogateSpike.apply(voltage - self.threshold)
                voltage = voltage - spikes * self.threshold
                logits = logits + spikes @ self.output_weights
            return logits

    return FeedforwardSNN()


def _batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _evaluate(model, store, indices, batch_size, device):
    import torch

    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_indices in _batches(indices, batch_size):
            events = torch.as_tensor(store.frames(batch_indices), device=device)
            predictions.append(torch.argmax(model(events), dim=1).cpu().numpy())
    predictions_array = np.concatenate(predictions)
    return float(np.mean(predictions_array == store.labels[indices])), predictions_array


def train_nmnist_seed(
    train_store: PackedNMNIST,
    test_store: PackedNMNIST,
    output_dir: str | Path,
    seed: int,
    config: NMNISTTrainConfig,
    partition_salt: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = (
        output_dir / "training_manifest.json",
        output_dir / "model.npz",
        output_dir / "checkpoint.pt",
        output_dir / "split_indices.npz",
    )
    if any(path.exists() for path in destinations):
        raise FileExistsError(f"N-MNIST output exists: {output_dir}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    splits = deterministic_partition(
        train_store.sample_ids,
        {"train": 0.8, "repair_calibration": 0.1, "certificate_audit": 0.1},
        partition_salt,
    )
    assert_disjoint_splits(splits.values())
    id_to_index = {sample_id: index for index, sample_id in enumerate(train_store.sample_ids)}
    indices = {
        name: np.asarray([id_to_index[item] for item in split.sample_ids], dtype=np.int64)
        for name, split in splits.items()
    }
    indices["test"] = np.arange(len(test_store.labels), dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feedforward_snn(
        train_store.input_channels,
        config.hidden_size,
        10,
        config.tau_mem,
        config.threshold,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        shuffled = generator.permutation(indices["train"])
        seen = 0
        correct = 0
        total_loss = 0.0
        epoch_started = time.perf_counter()
        model.train()
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
            seen += len(batch_indices)
            total_loss += float(loss.detach()) * len(batch_indices)
            correct += int(torch.sum(torch.argmax(logits, dim=1) == targets))
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / seen,
            "train_accuracy": correct / seen,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(
            f"N-MNIST seed={seed} epoch={epoch + 1}/{config.epochs} "
            f"loss={record['loss']:.4f} train_acc={record['train_accuracy']:.4f} "
            f"seconds={record['seconds']:.2f}",
            flush=True,
        )
    reference = DenseRecurrentSNN(
        input_weights=model.input_weights.detach().cpu().double().numpy(),
        recurrent_weights=np.zeros((config.hidden_size, config.hidden_size)),
        output_weights=model.output_weights.detach().cpu().double().numpy(),
        bias=model.bias.detach().cpu().double().numpy(),
        threshold=model.threshold.detach().cpu().double().numpy(),
        tau_mem=model.tau_mem.detach().cpu().double().numpy(),
        reset_value=np.zeros(config.hidden_size),
        name=f"nmnist-feedforward-seed-{seed}",
    )
    reference.save(destinations[1])
    accuracies = {}
    reference_test_predictions = None
    for name, store in (
        ("train", train_store),
        ("repair_calibration", train_store),
        ("certificate_audit", train_store),
        ("test", test_store),
    ):
        accuracy, predictions = _evaluate(
            model, store, indices[name], config.batch_size, device
        )
        accuracies[name] = accuracy
        if name == "test":
            reference_test_predictions = predictions
    checkpoint = {
        "schema_version": "NMNISTCheckpoint/v1",
        "state_dict": model.state_dict(),
        "model_hash": reference.model_hash,
        "seed": seed,
        "config": asdict(config),
        "code_revision": code_revision(repository_root),
    }
    with destinations[2].open("xb") as handle:
        torch.save(checkpoint, handle)
    with destinations[3].open("xb") as handle:
        np.savez_compressed(handle, **indices)
    with (output_dir / "reference_test_predictions.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            predictions=reference_test_predictions,
            sample_ids=test_store.sample_ids,
        )
    manifest = {
        "schema_version": "NMNISTTrainingManifest/v1",
        "seed": seed,
        "config": asdict(config),
        "preprocess": train_store.metadata["config"],
        "model_hash": reference.model_hash,
        "model_artifact_hash": sha256_file(destinations[1]),
        "checkpoint_hash": sha256_file(destinations[2]),
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "partition_salt": partition_salt,
        "split_counts": {name: len(value) for name, value in indices.items()},
        "split_hashes": {name: split.split_hash for name, split in splits.items()},
        "history": history,
        "final_accuracy": accuracies,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "code_revision": code_revision(repository_root),
    }
    write_json_immutable(destinations[0], manifest)
    return manifest


def load_nmnist_config(path: str | Path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != "NMNISTExperiment/v1":
        raise ValueError("unsupported N-MNIST experiment config")
    return (
        NMNISTPreprocessConfig(**raw["preprocess"]),
        NMNISTTrainConfig(**raw["training"]),
        raw,
    )

