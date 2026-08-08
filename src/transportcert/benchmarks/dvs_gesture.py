from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    sha256_json,
    write_json_immutable,
)
from ..protocol import assert_disjoint_splits, deterministic_partition
from ..semantics import (
    ExecutionSemantics,
    IntegrationRule,
    NumericFormat,
    OverflowMode,
    ResetRule,
    RoundingMode,
    ThresholdTiming,
)
from ..statistics import bonferroni_alpha, clopper_pearson_upper
from .semantic_matrix import primary_semantic_conditions
from .shd import _SurrogateSpike, _seed_everything


@dataclass(frozen=True)
class DVSGesturePreprocessConfig:
    schema_version: str = "DVSGesturePreprocess/v1"
    time_bins: int = 60
    window_microseconds: int = 1_500_000
    sensor_width: int = 32
    sensor_height: int = 32
    source_width: int = 128
    source_height: int = 128
    polarities: int = 2
    window_position: str = "center"
    windows_per_sample: int = 1
    binary_frames: bool = True

    @property
    def input_channels(self) -> int:
        return self.sensor_width * self.sensor_height * self.polarities


@dataclass(frozen=True)
class DVSGestureTrainConfig:
    schema_version: str = "DVSGestureTrain/v1"
    conv1_channels: int = 8
    conv2_channels: int = 16
    hidden_size: int = 128
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    tau_mem: float = 3.0
    threshold: float = 1.0
    recurrent_scale: float = 0.2
    dropout: float = 0.0
    gradient_clip: float = 1.0


class PackedDVSGesture:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.packed = np.asarray(data["packed"], dtype=np.uint8)
            self.labels = np.asarray(data["labels"], dtype=np.int64)
            self.sample_ids = np.asarray(data["sample_ids"]).astype(str)
            self.metadata = json.loads(str(data["metadata"]))
        config = self.metadata["config"]
        self.time_bins = int(config["time_bins"])
        self.sensor_width = int(config["sensor_width"])
        self.sensor_height = int(config["sensor_height"])
        self.polarities = int(config["polarities"])
        self.input_channels = self.sensor_width * self.sensor_height * self.polarities
        if self.packed.ndim == 3:
            self.windows_per_sample = 1
            temporal_shape = self.packed.shape[:2]
        elif self.packed.ndim == 4:
            self.windows_per_sample = self.packed.shape[1]
            temporal_shape = (self.packed.shape[0], self.packed.shape[2])
        else:
            raise ValueError("packed DVS Gesture artifact must have 3 or 4 axes")
        if temporal_shape != (len(self.labels), self.time_bins):
            raise ValueError("packed DVS Gesture artifact shape is inconsistent")

    def frames(
        self,
        indices: np.ndarray | list[int],
        window_indices: np.ndarray | list[int] | None = None,
    ) -> np.ndarray:
        indices_array = np.asarray(indices)
        if self.packed.ndim == 3:
            selected = self.packed[indices_array]
        else:
            if window_indices is None:
                raise ValueError("window_indices are required for multi-window data")
            selected = self.packed[indices_array, np.asarray(window_indices)]
        flat = np.unpackbits(
            selected, axis=-1, count=self.input_channels, bitorder="little"
        ).astype(np.float32)
        return flat.reshape(
            len(selected),
            self.time_bins,
            self.polarities,
            self.sensor_height,
            self.sensor_width,
        )

    def windowed_frames(self, indices: np.ndarray | list[int]) -> np.ndarray:
        indices_array = np.asarray(indices)
        if self.packed.ndim == 3:
            return self.frames(indices_array)[:, None]
        selected = self.packed[indices_array]
        flat = np.unpackbits(
            selected, axis=-1, count=self.input_channels, bitorder="little"
        ).astype(np.float32)
        return flat.reshape(
            len(selected),
            self.windows_per_sample,
            self.time_bins,
            self.polarities,
            self.sensor_height,
            self.sensor_width,
        )

    @property
    def data_hash(self) -> str:
        return sha256_file(self.path)


def preprocess_dvs_gesture(
    raw_root: str | Path,
    output_path: str | Path,
    split_name: str,
    config: DVSGesturePreprocessConfig,
) -> Path:
    from tonic.datasets import DVSGesture

    if config.window_position not in {"center", "uniform"}:
        raise ValueError("window_position must be center or uniform")
    if config.windows_per_sample < 1:
        raise ValueError("windows_per_sample must be positive")
    if config.window_position == "center" and config.windows_per_sample != 1:
        raise ValueError("center windowing supports exactly one window")
    raw_root = Path(raw_root)
    output_path = Path(output_path)
    train = split_name == "train"
    dataset = DVSGesture(str(raw_root), train=train)
    archive = Path(dataset.location_on_system) / dataset.filename
    archive_hash = sha256_file(archive)
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as existing:
            metadata = json.loads(str(existing["metadata"]))
        if metadata.get("source_archive_sha256") != archive_hash or metadata.get(
            "config"
        ) != asdict(config):
            raise ValueError("existing DVS Gesture preprocessing artifact differs")
        return output_path

    packed_width = math.ceil(config.input_channels / 8)
    packed = np.zeros(
        (
            len(dataset),
            config.windows_per_sample,
            config.time_bins,
            packed_width,
        ),
        dtype=np.uint8,
    )
    labels = np.asarray(dataset.targets, dtype=np.int64)
    sample_ids: list[str] = []
    selected_events = 0
    total_events = 0
    for index in range(len(dataset)):
        events, label = dataset[index]
        timestamps = events["t"].astype(np.int64)
        duration = int(timestamps[-1] - timestamps[0]) if len(timestamps) else 0
        available = max(0, duration - config.window_microseconds)
        if config.window_position == "center":
            offsets = np.asarray([available // 2], dtype=np.int64)
        else:
            offsets = np.rint(
                np.linspace(0, available, config.windows_per_sample)
            ).astype(np.int64)
        x = events["x"].astype(np.int64) * config.sensor_width // config.source_width
        y = events["y"].astype(np.int64) * config.sensor_height // config.source_height
        polarity = events["p"].astype(np.int64)
        channels = (
            (polarity * config.sensor_height + y) * config.sensor_width + x
        )
        for window_index, offset in enumerate(offsets):
            window_start = (int(timestamps[0]) if len(timestamps) else 0) + int(
                offset
            )
            bins = (
                (timestamps - window_start)
                * config.time_bins
                // config.window_microseconds
            )
            valid = (
                (bins >= 0)
                & (bins < config.time_bins)
                & (channels >= 0)
                & (channels < config.input_channels)
            )
            frame = np.zeros(
                (config.time_bins, config.input_channels), dtype=np.uint8
            )
            frame[bins[valid], channels[valid]] = 1
            packed[index, window_index] = np.packbits(
                frame, axis=-1, bitorder="little"
            )
            selected_events += int(np.count_nonzero(valid))
        total_events += len(events)
        relative = Path(dataset.data[index]).relative_to(dataset.location_on_system)
        sample_ids.append(f"dvs-gesture-{split_name}-{relative.as_posix()}")
        if int(label) != labels[index]:
            raise ValueError("DVS Gesture label mismatch")
        if (index + 1) % 100 == 0:
            print(
                f"preprocess DVS Gesture {split_name}: {index + 1}/{len(dataset)}",
                flush=True,
            )

    sample_ids_array = np.asarray(sample_ids)
    metadata = {
        "schema_version": config.schema_version,
        "config": asdict(config),
        "split": split_name,
        "samples": len(dataset),
        "source_archive": str(archive.resolve()),
        "source_archive_sha256": archive_hash,
        "source_archive_md5": dataset.file_md5,
        "packed_hash": array_hash(packed),
        "labels_hash": array_hash(labels),
        "sample_ids_hash": sha256_json(sample_ids),
        "total_events": total_events,
        "selected_window_events": selected_events,
        "selected_event_fraction": selected_events / total_events,
        "timestep_microseconds": config.window_microseconds / config.time_bins,
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


def _quantize_tensor(value: Any, numeric: NumericFormat, generator: Any) -> Any:
    import torch

    if not numeric.is_fixed:
        return value.to(torch.float32 if numeric.kind == "float32" else torch.float64)
    assert numeric.fractional_bits is not None
    assert numeric.total_bits is not None
    scale = float(1 << numeric.fractional_bits)
    scaled = value * scale
    if numeric.rounding is RoundingMode.NEAREST_EVEN:
        integers = torch.round(scaled)
    elif numeric.rounding is RoundingMode.FLOOR:
        integers = torch.floor(scaled)
    elif numeric.rounding is RoundingMode.TRUNCATE:
        integers = torch.trunc(scaled)
    else:
        lower = torch.floor(scaled)
        integers = lower + (
            torch.rand(
                scaled.shape,
                generator=generator,
                device=scaled.device,
                dtype=scaled.dtype,
            )
            < (scaled - lower)
        ).to(scaled.dtype)
    minimum = -(1 << (numeric.total_bits - 1))
    maximum = (1 << (numeric.total_bits - 1)) - 1
    if numeric.overflow is OverflowMode.SATURATE:
        integers = torch.clamp(integers, minimum, maximum)
    else:
        modulus = 1 << numeric.total_bits
        integers = torch.remainder(integers - minimum, modulus) + minimum
    return integers / scale


def build_dvs_conv_srnn(
    sensor_width: int,
    sensor_height: int,
    config: DVSGestureTrainConfig,
    output_size: int = 11,
) -> Any:
    import torch

    class DVSConvSRNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv2d(
                2, config.conv1_channels, kernel_size=5, stride=2, bias=True
            )
            self.conv2 = torch.nn.Conv2d(
                config.conv1_channels,
                config.conv2_channels,
                kernel_size=3,
                stride=2,
                bias=True,
            )
            width1 = (sensor_width - 5) // 2 + 1
            height1 = (sensor_height - 5) // 2 + 1
            width2 = (width1 - 3) // 2 + 1
            height2 = (height1 - 3) // 2 + 1
            self.feature_shape = (config.conv2_channels, height2, width2)
            self.hidden_input = torch.nn.Linear(
                config.conv2_channels * height2 * width2,
                config.hidden_size,
                bias=True,
            )
            self.recurrent = torch.nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.readout = torch.nn.Linear(
                config.hidden_size, output_size, bias=False
            )
            torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="linear")
            torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="linear")
            torch.nn.init.xavier_uniform_(self.hidden_input.weight)
            torch.nn.init.orthogonal_(self.recurrent.weight)
            torch.nn.init.xavier_uniform_(self.readout.weight)
            with torch.no_grad():
                self.recurrent.weight.mul_(config.recurrent_scale)

        def forward(self, events, semantics: ExecutionSemantics | None = None):
            training_surrogate = semantics is None
            if semantics is None:
                float32 = NumericFormat("float32")
                semantics = ExecutionSemantics(
                    state_format=float32, weight_format=float32
                )
            generator = torch.Generator(device=events.device)
            generator.manual_seed(semantics.randomness.seed)
            qs = semantics.state_format
            qw = semantics.weight_format

            def weight(parameter):
                return _quantize_tensor(parameter, qw, generator)

            conv1_weight = weight(self.conv1.weight)
            conv2_weight = weight(self.conv2.weight)
            hidden_weight = weight(self.hidden_input.weight)
            recurrent_weight = weight(self.recurrent.weight)
            readout_weight = weight(self.readout.weight)
            conv1_bias = _quantize_tensor(self.conv1.bias, qs, generator)
            conv2_bias = _quantize_tensor(self.conv2.bias, qs, generator)
            hidden_bias = _quantize_tensor(self.hidden_input.bias, qs, generator)
            batch = events.shape[0]
            width1 = (sensor_width - 5) // 2 + 1
            height1 = (sensor_height - 5) // 2 + 1
            conv1_voltage = torch.zeros(
                (batch, config.conv1_channels, height1, width1),
                dtype=events.dtype,
                device=events.device,
            )
            conv2_voltage = torch.zeros(
                (batch, *self.feature_shape),
                dtype=events.dtype,
                device=events.device,
            )
            hidden_voltage = torch.zeros(
                (batch, config.hidden_size),
                dtype=events.dtype,
                device=events.device,
            )
            hidden_spikes = torch.zeros_like(hidden_voltage)
            logits = torch.zeros(
                (batch, output_size), dtype=events.dtype, device=events.device
            )
            queue1 = [
                torch.zeros_like(conv1_voltage)
                for _ in range(semantics.synaptic_delay_steps)
            ]
            queue2 = [
                torch.zeros_like(conv2_voltage)
                for _ in range(semantics.synaptic_delay_steps)
            ]
            queue_hidden = [
                torch.zeros_like(hidden_voltage)
                for _ in range(semantics.synaptic_delay_steps)
            ]
            output_queue = [
                torch.zeros_like(logits)
                for _ in range(semantics.output_delay_steps)
            ]
            tau = config.tau_mem
            threshold = config.threshold

            def delay(raw, queue):
                if queue:
                    queue.append(raw)
                    return queue.pop(0)
                return raw

            def integrate(voltage, current):
                if semantics.integration_rule is IntegrationRule.FORWARD_EULER:
                    return voltage + semantics.timestep * (-voltage + current) / tau
                alpha = math.exp(-semantics.timestep / tau)
                return alpha * voltage + (1.0 - alpha) * current

            def emit(voltage):
                if training_surrogate:
                    return _SurrogateSpike.apply(voltage - threshold)
                return (voltage >= threshold).to(voltage.dtype)

            def reset(voltage, spikes):
                if semantics.reset_rule is ResetRule.SUBTRACTIVE:
                    return voltage - spikes * threshold
                return torch.where(spikes > 0, torch.zeros_like(voltage), voltage)

            def lif_step(voltage, current):
                if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                    spikes = emit(voltage)
                    voltage = integrate(reset(voltage, spikes), current)
                else:
                    voltage = _quantize_tensor(
                        integrate(voltage, current), qs, generator
                    )
                    spikes = emit(voltage)
                    voltage = reset(voltage, spikes)
                return _quantize_tensor(voltage, qs, generator), spikes

            for step in range(events.shape[1]):
                raw1 = torch.nn.functional.conv2d(
                    events[:, step], conv1_weight, conv1_bias, stride=2
                )
                current1 = delay(_quantize_tensor(raw1, qs, generator), queue1)
                conv1_voltage, spikes1 = lif_step(conv1_voltage, current1)
                raw2 = torch.nn.functional.conv2d(
                    spikes1, conv2_weight, conv2_bias, stride=2
                )
                current2 = delay(_quantize_tensor(raw2, qs, generator), queue2)
                conv2_voltage, spikes2 = lif_step(conv2_voltage, current2)
                flattened = torch.nn.functional.dropout(
                    spikes2.flatten(1),
                    p=config.dropout,
                    training=self.training and training_surrogate,
                )
                raw_hidden = (
                    torch.nn.functional.linear(
                        flattened, hidden_weight, hidden_bias
                    )
                    + torch.nn.functional.linear(
                        hidden_spikes, recurrent_weight, None
                    )
                )
                hidden_current = delay(
                    _quantize_tensor(raw_hidden, qs, generator), queue_hidden
                )
                hidden_voltage, hidden_spikes = lif_step(
                    hidden_voltage, hidden_current
                )
                readout_spikes = torch.nn.functional.dropout(
                    hidden_spikes,
                    p=config.dropout,
                    training=self.training and training_surrogate,
                )
                contribution = _quantize_tensor(
                    torch.nn.functional.linear(readout_spikes, readout_weight, None),
                    qs,
                    generator,
                )
                delivered = delay(contribution, output_queue)
                logits = _quantize_tensor(logits + delivered, qs, generator)
            return logits

    return DVSConvSRNN()


def _batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _model_hash(model: Any) -> str:
    return sha256_json(
        {
            name: array_hash(value.detach().cpu().numpy())
            for name, value in sorted(model.state_dict().items())
        }
    )


def evaluate_dvs_model(
    model: Any,
    store: PackedDVSGesture,
    indices: np.ndarray,
    batch_size: int,
    device: Any,
    semantics: ExecutionSemantics,
) -> tuple[float, np.ndarray, np.ndarray]:
    import torch

    predictions: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_indices in _batches(indices, batch_size):
            windows = store.windowed_frames(batch_indices)
            batch, window_count = windows.shape[:2]
            events = torch.as_tensor(
                windows.reshape(batch * window_count, *windows.shape[2:]),
                device=device,
            )
            window_logits = model(events, semantics)
            batch_logits = window_logits.reshape(batch, window_count, -1).mean(dim=1)
            logits.append(batch_logits.cpu().numpy())
            predictions.append(torch.argmax(batch_logits, dim=1).cpu().numpy())
    all_predictions = np.concatenate(predictions)
    all_logits = np.concatenate(logits)
    return (
        float(np.mean(all_predictions == store.labels[indices])),
        all_predictions,
        all_logits,
    )


def train_dvs_gesture_seed(
    train_store: PackedDVSGesture,
    test_store: PackedDVSGesture,
    output_dir: str | Path,
    seed: int,
    config: DVSGestureTrainConfig,
    partition_salt: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_manifest.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    split_path = output_dir / "split_indices.npz"
    prediction_path = output_dir / "reference_test_predictions.npz"
    if any(path.exists() for path in (manifest_path, checkpoint_path, split_path)):
        raise FileExistsError(f"DVS Gesture output exists: {output_dir}")
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
    indices = {
        name: np.asarray(
            [id_to_index[item] for item in split.sample_ids], dtype=np.int64
        )
        for name, split in splits.items()
    }
    indices["test"] = np.arange(len(test_store.labels), dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dvs_conv_srnn(
        train_store.sensor_width, train_store.sensor_height, config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        shuffled = generator.permutation(indices["train"])
        total_loss = 0.0
        correct = 0
        seen = 0
        epoch_started = time.perf_counter()
        for batch_indices in _batches(shuffled, config.batch_size):
            window_indices = generator.integers(
                train_store.windows_per_sample, size=len(batch_indices)
            )
            events = torch.as_tensor(
                train_store.frames(batch_indices, window_indices), device=device
            )
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
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / seen,
            "train_accuracy": correct / seen,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(
            f"DVS seed={seed} epoch={epoch + 1}/{config.epochs} "
            f"loss={record['loss']:.4f} train_acc={record['train_accuracy']:.4f} "
            f"seconds={record['seconds']:.2f}",
            flush=True,
        )

    float32 = NumericFormat("float32")
    reference_semantics = ExecutionSemantics(
        state_format=float32, weight_format=float32
    )
    final_accuracy: dict[str, float] = {}
    test_predictions: np.ndarray | None = None
    for name, store in (
        ("train", train_store),
        ("repair_calibration", train_store),
        ("certificate_audit", train_store),
        ("test", test_store),
    ):
        accuracy, predictions, _ = evaluate_dvs_model(
            model,
            store,
            indices[name],
            config.batch_size,
            device,
            reference_semantics,
        )
        final_accuracy[name] = accuracy
        if name == "test":
            test_predictions = predictions
    model_hash = _model_hash(model)
    checkpoint = {
        "schema_version": "DVSGestureCheckpoint/v1",
        "state_dict": model.state_dict(),
        "model_hash": model_hash,
        "seed": seed,
        "config": asdict(config),
        "sensor_width": train_store.sensor_width,
        "sensor_height": train_store.sensor_height,
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "reference_semantics": reference_semantics.to_dict(),
        "code_revision": code_revision(repository_root),
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    with split_path.open("xb") as handle:
        np.savez_compressed(handle, **indices)
    assert test_predictions is not None
    with prediction_path.open("xb") as handle:
        np.savez_compressed(
            handle, predictions=test_predictions, sample_ids=test_store.sample_ids
        )
    manifest = {
        "schema_version": "DVSGestureTrainingManifest/v1",
        "seed": seed,
        "config": asdict(config),
        "preprocess": train_store.metadata["config"],
        "model_hash": model_hash,
        "checkpoint_hash": sha256_file(checkpoint_path),
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "partition_salt": partition_salt,
        "split_counts": {name: len(split.sample_ids) for name, split in splits.items()},
        "split_hashes": {name: split.split_hash for name, split in splits.items()},
        "history": history,
        "final_accuracy": final_accuracy,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "code_revision": code_revision(repository_root),
    }
    write_json_immutable(manifest_path, manifest)
    return manifest


def run_dvs_semantic_matrix(
    checkpoint_path: str | Path,
    train_store: PackedDVSGesture,
    test_store: PackedDVSGesture,
    split_indices_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
    *,
    confidence: float = 0.95,
    batch_size: int = 64,
) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(checkpoint_path)
    split_indices_path = Path(split_indices_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "semantic_matrix.json"
    predictions_path = output_dir / "semantic_predictions.npz"
    if summary_path.exists() or predictions_path.exists():
        raise FileExistsError(f"DVS semantic output exists: {output_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DVSGestureTrainConfig(**checkpoint["config"])
    model = build_dvs_conv_srnn(
        int(checkpoint["sensor_width"]), int(checkpoint["sensor_height"]), config
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with np.load(split_indices_path, allow_pickle=False) as split_data:
        audit_indices = np.asarray(split_data["certificate_audit"], dtype=np.int64)
        calibration_indices = np.asarray(
            split_data["repair_calibration"], dtype=np.int64
        )
        test_indices = np.asarray(split_data["test"], dtype=np.int64)
    conditions = primary_semantic_conditions()
    targets = [name for name in conditions if name != "reference"]
    alpha_each = bonferroni_alpha(1.0 - confidence, len(targets))
    predictions: dict[str, np.ndarray] = {}
    logits: dict[str, np.ndarray] = {}
    for split_name, store, indices in (
        ("audit", train_store, audit_indices),
        ("calibration", train_store, calibration_indices),
        ("test", test_store, test_indices),
    ):
        for condition_name, semantics in conditions.items():
            accuracy, prediction, output = evaluate_dvs_model(
                model, store, indices, batch_size, device, semantics
            )
            key = f"{split_name}__{condition_name}"
            predictions[key] = prediction
            logits[key] = output
            print(
                f"DVS executed {split_name} {condition_name} "
                f"samples={len(indices)} accuracy={accuracy:.4f}",
                flush=True,
            )
    reference_audit = predictions["audit__reference"]
    reference_test = predictions["test__reference"]
    reference_test_accuracy = float(
        np.mean(reference_test == test_store.labels[test_indices])
    )
    rows: list[dict[str, Any]] = []
    family_agreement = np.ones(len(audit_indices), dtype=bool)
    for name in targets:
        audit_prediction = predictions[f"audit__{name}"]
        test_prediction = predictions[f"test__{name}"]
        disagreements = int(np.count_nonzero(audit_prediction != reference_audit))
        upper_bound = clopper_pearson_upper(
            disagreements, len(audit_indices), alpha_each
        )
        target_accuracy = float(
            np.mean(test_prediction == test_store.labels[test_indices])
        )
        absolute_change = abs(target_accuracy - reference_test_accuracy)
        family_agreement &= audit_prediction == reference_audit
        rows.append(
            {
                "condition": name,
                "semantics_hash": conditions[name].semantics_hash,
                "audit_disagreements": disagreements,
                "audit_samples": len(audit_indices),
                "audit_disagreement_rate": disagreements / len(audit_indices),
                "simultaneous_upper_bound": upper_bound,
                "reference_test_accuracy": reference_test_accuracy,
                "target_test_accuracy": target_accuracy,
                "accuracy_loss": reference_test_accuracy - target_accuracy,
                "absolute_accuracy_change": absolute_change,
                "bound_slack_vs_test_change": upper_bound - absolute_change,
                "bound_not_violated": absolute_change <= upper_bound,
                "budget_verdicts": {
                    str(budget): "accept" if upper_bound <= budget else "reject"
                    for budget in (0.01, 0.02, 0.05)
                },
            }
        )
    payload: dict[str, Any] = {
        "audit_indices": audit_indices,
        "calibration_indices": calibration_indices,
        "test_indices": test_indices,
    }
    payload.update({f"pred__{key}": value for key, value in predictions.items()})
    payload.update({f"logits__{key}": value for key, value in logits.items()})
    with predictions_path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    summary = {
        "schema_version": "DVSGestureSemanticMatrix/v1",
        "benchmark": "DVS Gesture",
        "model_hash": checkpoint["model_hash"],
        "checkpoint_hash": sha256_file(checkpoint_path),
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "split_indices_hash": sha256_file(split_indices_path),
        "prediction_artifact_hash": sha256_file(predictions_path),
        "reference_semantics_hash": conditions["reference"].semantics_hash,
        "condition_semantics": {
            name: semantics.to_dict() for name, semantics in conditions.items()
        },
        "confidence": confidence,
        "simultaneous_method": "Bonferroni-corrected one-sided Clopper-Pearson",
        "per_condition_alpha": alpha_each,
        "audit_samples": len(audit_indices),
        "calibration_samples": len(calibration_indices),
        "test_samples": len(test_indices),
        "reference_test_accuracy": reference_test_accuracy,
        "family_observed_agreement_fraction": float(np.mean(family_agreement)),
        "conditions_over_five_point_loss": int(
            sum(row["accuracy_loss"] > 0.05 for row in rows)
        ),
        "simultaneous_bound_violations": int(
            sum(not row["bound_not_violated"] for row in rows)
        ),
        "rows": rows,
        "device": str(device),
        "runtime_dtype": "float32",
        "torch_version": torch.__version__,
        "code_revision": code_revision(repository_root),
        "interpretation": (
            "Software-only prospective audit; conditional on emulator and not a physical certificate."
        ),
    }
    write_json_immutable(summary_path, summary)
    return summary


def load_dvs_gesture_config(
    path: str | Path,
) -> tuple[DVSGesturePreprocessConfig, DVSGestureTrainConfig, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != "DVSGestureExperiment/v1":
        raise ValueError("unsupported DVS Gesture experiment config")
    return (
        DVSGesturePreprocessConfig(**raw["preprocess"]),
        DVSGestureTrainConfig(**raw["training"]),
        raw,
    )
