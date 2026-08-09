from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import array_hash, code_revision, sha256_file, write_json_immutable
from ..semantics import ExecutionSemantics
from ..statistics import clopper_pearson_upper
from .dvs_gesture import (
    DVSGestureTrainConfig,
    PackedDVSGesture,
    _model_hash,
    build_dvs_conv_srnn,
)
from .semantic_matrix import primary_semantic_conditions
from .shd import _seed_everything


@dataclass(frozen=True)
class DVSGestureRepairConfig:
    schema_version: str = "DVSGestureRepair/v2"
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 0.02
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    logit_weight: float = 0.05
    regularization_weight: float = 1e-4
    supervised_learning_rate: float = 1e-3
    simultaneous_family_size: int = 10


class _OutputChannelScale:
    @staticmethod
    def build(size: int, dimensions: int) -> Any:
        import torch

        class Scale(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.log_scale = torch.nn.Parameter(torch.zeros(size))

            def forward(self, weight):
                shape = (size,) + (1,) * (dimensions - 1)
                return weight * torch.exp(self.log_scale).reshape(shape)

        return Scale()


class _InputChannelScale:
    @staticmethod
    def build(size: int) -> Any:
        import torch

        class Scale(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.log_scale = torch.nn.Parameter(torch.zeros(size))

            def forward(self, weight):
                return weight * torch.exp(self.log_scale).unsqueeze(0)

        return Scale()


def make_restricted_dvs_repairable(model: Any) -> Any:
    """Expose only per-channel scales and existing neuron biases."""

    from torch.nn.utils import parametrize

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parametrize.register_parametrization(
        model.conv1,
        "weight",
        _OutputChannelScale.build(model.conv1.out_channels, 4),
    )
    parametrize.register_parametrization(
        model.conv2,
        "weight",
        _OutputChannelScale.build(model.conv2.out_channels, 4),
    )
    hidden_incoming_scale = _OutputChannelScale.build(
        model.hidden_input.out_features, 2
    )
    parametrize.register_parametrization(
        model.hidden_input, "weight", hidden_incoming_scale
    )
    parametrize.register_parametrization(
        model.recurrent,
        "weight",
        hidden_incoming_scale,
    )
    parametrize.register_parametrization(
        model.readout,
        "weight",
        _InputChannelScale.build(model.readout.in_features),
    )
    model.conv1.bias.requires_grad_(True)
    model.conv2.bias.requires_grad_(True)
    model.hidden_input.bias.requires_grad_(True)
    return model


def materialize_dvs_repair(model: Any) -> Any:
    from torch.nn.utils import parametrize

    for module in (
        model.conv1,
        model.conv2,
        model.hidden_input,
        model.recurrent,
        model.readout,
    ):
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


def _predict(
    model: Any,
    store: PackedDVSGesture,
    indices: np.ndarray,
    semantics: ExecutionSemantics,
    batch_size: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    predictions: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            sample_indices = indices[start : start + batch_size]
            windows = store.windowed_frames(sample_indices)
            batch, window_count = windows.shape[:2]
            events = torch.as_tensor(
                windows.reshape(batch * window_count, *windows.shape[2:]),
                device=device,
            )
            window_logits = model(events, semantics).reshape(batch, window_count, -1)
            if model.aggregation_temperature > 0:
                logits = torch.softmax(
                    window_logits / model.aggregation_temperature, dim=2
                ).mean(dim=1)
            else:
                logits = window_logits.mean(dim=1)
            outputs.append(logits.cpu().numpy())
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(outputs)


def _train_batch_logits(
    model: Any,
    store: PackedDVSGesture,
    sample_indices: np.ndarray,
    semantics: ExecutionSemantics,
    device: Any,
) -> Any:
    import torch

    windows = store.windowed_frames(sample_indices)
    batch, window_count = windows.shape[:2]
    events = torch.as_tensor(
        windows.reshape(batch * window_count, *windows.shape[2:]), device=device
    )
    window_logits = model(
        events, semantics, surrogate_gradients=True
    ).reshape(batch, window_count, -1)
    if model.aggregation_temperature > 0:
        return torch.softmax(
            window_logits / model.aggregation_temperature, dim=2
        ).mean(dim=1)
    return window_logits.mean(dim=1)


def _log_scale_regularization(model: Any) -> Any:
    import torch

    values = [
        parameter.square().mean()
        for name, parameter in model.named_parameters()
        if name.endswith("log_scale")
    ]
    return torch.stack(values).sum() if values else torch.zeros(())


def _clamp_log_scales(model: Any) -> None:
    for name, parameter in model.named_parameters():
        if name.endswith("log_scale"):
            parameter.data.clamp_(np.log(0.25), np.log(4.0))


def run_dvs_repair(
    checkpoint_path: str | Path,
    train_store: PackedDVSGesture,
    test_store: PackedDVSGesture,
    split_indices_path: str | Path,
    semantic_predictions_path: str | Path,
    condition: str,
    method: str,
    output_dir: str | Path,
    repository_root: str | Path,
    config: DVSGestureRepairConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    import torch

    if method not in {
        "certificate_directed",
        "logit_only",
        "global_threshold",
        "per_platform_qat",
        "supervised_target_retraining",
    }:
        raise ValueError("unsupported DVS repair method")
    conditions = primary_semantic_conditions()
    if condition == "reference" or condition not in conditions:
        raise ValueError("repair requires a declared non-reference condition")
    target = conditions[condition]
    checkpoint_path = Path(checkpoint_path)
    split_indices_path = Path(split_indices_path)
    semantic_predictions_path = Path(semantic_predictions_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "repair_report.json"
    checkpoint_output = output_dir / "repaired_checkpoint.pt"
    predictions_output = output_dir / "repaired_predictions.npz"
    if any(path.exists() for path in (report_path, checkpoint_output, predictions_output)):
        raise FileExistsError(f"DVS repair output already exists: {output_dir}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = DVSGestureTrainConfig(**checkpoint["config"])
    sensor_width = int(checkpoint["sensor_width"])
    sensor_height = int(checkpoint["sensor_height"])
    with np.load(split_indices_path, allow_pickle=False) as split_data:
        calibration_indices = np.asarray(
            split_data["repair_calibration"], dtype=np.int64
        )
        audit_indices = np.asarray(split_data["certificate_audit"], dtype=np.int64)
        test_indices = np.asarray(split_data["test"], dtype=np.int64)
    with np.load(semantic_predictions_path, allow_pickle=False) as predictions:
        reference_calibration = np.asarray(
            predictions["pred__calibration__reference"]
        )
        reference_calibration_logits = np.asarray(
            predictions["logits__calibration__reference"]
        )
        reference_audit = np.asarray(predictions["pred__audit__reference"])
        reference_test = np.asarray(predictions["pred__test__reference"])
        target_audit_before = np.asarray(predictions[f"pred__audit__{condition}"])
        target_test_before = np.asarray(predictions[f"pred__test__{condition}"])

    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    optimization_steps = 0
    label_budget = 0
    selection_criterion = "reference prediction disagreement"

    if method == "global_threshold":
        best_model = None
        best_config = train_config
        best_disagreement = len(calibration_indices) + 1
        best_scale = 1.0
        for scale in np.linspace(0.5, 1.5, 21):
            candidate_config = replace(
                train_config, threshold=train_config.threshold * float(scale)
            )
            candidate = build_dvs_conv_srnn(
                sensor_width, sensor_height, candidate_config
            )
            candidate.load_state_dict(checkpoint["state_dict"])
            candidate.to(device)
            prediction, _ = _predict(
                candidate,
                train_store,
                calibration_indices,
                target,
                config.batch_size,
                device,
            )
            disagreement = int(np.count_nonzero(prediction != reference_calibration))
            history.append(
                {"scale": float(scale), "calibration_disagreements": disagreement}
            )
            if disagreement < best_disagreement:
                best_disagreement = disagreement
                best_model = candidate
                best_config = candidate_config
                best_scale = float(scale)
        assert best_model is not None
        selected = {
            "global_threshold_scale": best_scale,
            "best_calibration_disagreements": best_disagreement,
        }
        trainable_parameters = 1
    elif method in {"certificate_directed", "logit_only"}:
        model = build_dvs_conv_srnn(sensor_width, sensor_height, train_config)
        model.load_state_dict(checkpoint["state_dict"])
        model = make_restricted_dvs_repairable(model).to(device)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        trainable_parameters = int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        rng = np.random.default_rng(seed)
        initial_prediction, _ = _predict(
            model,
            train_store,
            calibration_indices,
            target,
            config.batch_size,
            device,
        )
        best_disagreement = int(
            np.count_nonzero(initial_prediction != reference_calibration)
        )
        best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": 0,
                "calibration_disagreements": best_disagreement,
            }
        )
        for epoch in range(config.epochs):
            order = rng.permutation(len(calibration_indices))
            total_loss = 0.0
            seen = 0
            model.train()
            for start in range(0, len(order), config.batch_size):
                positions = order[start : start + config.batch_size]
                sample_indices = calibration_indices[positions]
                teacher_logits = torch.as_tensor(
                    reference_calibration_logits[positions], device=device
                )
                teacher_predictions = torch.as_tensor(
                    reference_calibration[positions], dtype=torch.long, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                logits = _train_batch_logits(
                    model, train_store, sample_indices, target, device
                )
                logit_loss = torch.nn.functional.smooth_l1_loss(
                    logits, teacher_logits
                ) / (torch.mean(torch.abs(teacher_logits)) + 1e-6)
                if method == "certificate_directed":
                    rows = torch.arange(len(positions), device=device)
                    selected_scores = logits[rows, teacher_predictions]
                    masked = logits.clone()
                    masked[rows, teacher_predictions] = -torch.inf
                    margins = selected_scores - torch.max(masked, dim=1).values
                    margin_loss = torch.nn.functional.softplus(-margins).mean()
                    loss = (
                        margin_loss
                        + config.logit_weight * logit_loss
                        + config.regularization_weight
                        * _log_scale_regularization(model).to(device)
                    )
                else:
                    loss = (
                        logit_loss
                        + config.regularization_weight
                        * _log_scale_regularization(model).to(device)
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    config.gradient_clip,
                )
                optimizer.step()
                _clamp_log_scales(model)
                optimization_steps += 1
                total_loss += float(loss.detach()) * len(positions)
                seen += len(positions)
            calibration_prediction, _ = _predict(
                model,
                train_store,
                calibration_indices,
                target,
                config.batch_size,
                device,
            )
            disagreement = int(
                np.count_nonzero(calibration_prediction != reference_calibration)
            )
            record = {
                "epoch": epoch + 1,
                "loss": total_loss / seen,
                "calibration_disagreements": disagreement,
            }
            history.append(record)
            print(
                f"DVS repair {method} {condition} epoch={epoch + 1}/{config.epochs} "
                f"loss={record['loss']:.4f} disagreements={disagreement}",
                flush=True,
            )
            if disagreement < best_disagreement:
                best_disagreement = disagreement
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        best_model = materialize_dvs_repair(model)
        best_config = train_config
        selected = {"best_calibration_disagreements": best_disagreement}
    else:
        model = build_dvs_conv_srnn(sensor_width, sensor_height, train_config)
        if method == "per_platform_qat":
            model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.supervised_learning_rate,
            weight_decay=config.weight_decay,
        )
        trainable_parameters = int(
            sum(parameter.numel() for parameter in model.parameters())
        )
        label_budget = len(calibration_indices)
        selection_criterion = "labeled calibration accuracy"
        rng = np.random.default_rng(seed)
        initial_prediction, _ = _predict(
            model,
            train_store,
            calibration_indices,
            target,
            config.batch_size,
            device,
        )
        best_accuracy = float(
            np.mean(initial_prediction == train_store.labels[calibration_indices])
        )
        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": 0, "calibration_accuracy": best_accuracy})
        for epoch in range(config.epochs):
            order = rng.permutation(calibration_indices)
            total_loss = 0.0
            seen = 0
            model.train()
            for start in range(0, len(order), config.batch_size):
                sample_indices = order[start : start + config.batch_size]
                targets = torch.as_tensor(
                    train_store.labels[sample_indices], dtype=torch.long, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                logits = _train_batch_logits(
                    model, train_store, sample_indices, target, device
                )
                if model.aggregation_temperature > 0:
                    loss = torch.nn.functional.nll_loss(
                        torch.log(logits.clamp_min(1e-8)), targets
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                optimizer.step()
                optimization_steps += 1
                total_loss += float(loss.detach()) * len(sample_indices)
                seen += len(sample_indices)
            calibration_prediction, _ = _predict(
                model,
                train_store,
                calibration_indices,
                target,
                config.batch_size,
                device,
            )
            accuracy = float(
                np.mean(
                    calibration_prediction
                    == train_store.labels[calibration_indices]
                )
            )
            epoch_loss = total_loss / seen
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "calibration_accuracy": accuracy,
                }
            )
            print(
                f"DVS repair {method} {condition} epoch={epoch + 1}/{config.epochs} "
                f"loss={epoch_loss:.4f} accuracy={accuracy:.4f}",
                flush=True,
            )
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and epoch_loss < best_loss
            ):
                best_accuracy = accuracy
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        best_model = model
        best_config = train_config
        selected = {
            "initialization": (
                "source" if method == "per_platform_qat" else "random"
            ),
            "best_calibration_accuracy": best_accuracy,
            "best_calibration_loss": best_loss,
        }

    best_model.eval()
    repaired_hash = _model_hash(best_model)
    repaired_checkpoint = {
        "schema_version": "DVSGestureRepairedCheckpoint/v1",
        "state_dict": best_model.state_dict(),
        "model_hash": repaired_hash,
        "source_model_hash": checkpoint["model_hash"],
        "method": method,
        "condition": condition,
        "config": asdict(best_config),
        "sensor_width": sensor_width,
        "sensor_height": sensor_height,
        "code_revision": code_revision(repository_root),
    }
    with checkpoint_output.open("xb") as handle:
        torch.save(repaired_checkpoint, handle)

    audit_prediction, _ = _predict(
        best_model, train_store, audit_indices, target, 32, device
    )
    test_prediction, _ = _predict(
        best_model, test_store, test_indices, target, 32, device
    )
    after_disagreements = int(np.count_nonzero(audit_prediction != reference_audit))
    before_disagreements = int(
        np.count_nonzero(target_audit_before != reference_audit)
    )
    reference_accuracy = float(
        np.mean(reference_test == test_store.labels[test_indices])
    )
    before_accuracy = float(
        np.mean(target_test_before == test_store.labels[test_indices])
    )
    after_accuracy = float(np.mean(test_prediction == test_store.labels[test_indices]))
    before_loss = reference_accuracy - before_accuracy
    after_loss = reference_accuracy - after_accuracy
    recovery = (
        (before_loss - after_loss) / before_loss if before_loss > 0 else float("nan")
    )
    alpha = 0.05 / config.simultaneous_family_size
    after_bound = clopper_pearson_upper(
        after_disagreements, len(audit_indices), alpha
    )
    with predictions_output.open("xb") as handle:
        np.savez_compressed(
            handle,
            audit_predictions=audit_prediction,
            test_predictions=test_prediction,
            audit_indices=audit_indices,
            test_indices=test_indices,
        )
    report = {
        "schema_version": "DVSGestureRepairExperiment/v1",
        "method": method,
        "condition": condition,
        "source_model_hash": checkpoint["model_hash"],
        "repaired_model_hash": repaired_hash,
        "target_semantics_hash": target.semantics_hash,
        "source_checkpoint_hash": sha256_file(checkpoint_path),
        "repaired_checkpoint_hash": sha256_file(checkpoint_output),
        "semantic_predictions_hash": sha256_file(semantic_predictions_path),
        "split_indices_hash": sha256_file(split_indices_path),
        "predictions_artifact_hash": sha256_file(predictions_output),
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "calibration_indices_hash": array_hash(calibration_indices),
        "audit_indices_hash": array_hash(audit_indices),
        "test_indices_hash": array_hash(test_indices),
        "calibration_samples": len(calibration_indices),
        "audit_samples": len(audit_indices),
        "test_samples": len(test_indices),
        "label_budget": label_budget,
        "labels_used": "none" if label_budget == 0 else "repair calibration labels",
        "optimization_steps": optimization_steps,
        "trainable_parameters": trainable_parameters,
        "selection_criterion": selection_criterion,
        "test_labels_used_for_selection": False,
        "calibration_audit_disjoint": True,
        "permitted_parameter_changes": (
            ["global_threshold_scale"]
            if method == "global_threshold"
            else ["all_weights", "conv_and_hidden_bias"]
            if method in {
                "per_platform_qat",
                "supervised_target_retraining",
            }
            else [
                "per_channel_incoming_weight_scale",
                "per_hidden_output_weight_scale",
                "conv_and_hidden_bias",
            ]
        ),
        "baseline_definition": (
            "source-initialized supervised target-semantics QAT with NLL on "
            "aggregated window probabilities"
            if method == "per_platform_qat"
            else "random-initialized supervised target-semantics retraining with "
            "NLL on aggregated window probabilities"
            if method == "supervised_target_retraining"
            else "label-free hard-semantics global threshold search"
            if method == "global_threshold"
            else "restricted label-free target-semantics calibration"
        ),
        "config": asdict(config),
        "selected": selected,
        "before": {
            "audit_disagreements": before_disagreements,
            "audit_disagreement_rate": before_disagreements / len(audit_indices),
            "reference_test_accuracy": reference_accuracy,
            "target_test_accuracy": before_accuracy,
            "accuracy_loss": before_loss,
        },
        "after": {
            "audit_disagreements": after_disagreements,
            "audit_disagreement_rate": after_disagreements / len(audit_indices),
            "certificate_upper_bound": after_bound,
            "reference_test_accuracy": reference_accuracy,
            "target_test_accuracy": after_accuracy,
            "accuracy_loss": after_loss,
        },
        "accuracy_recovery_fraction": recovery,
        "confidence_alpha": alpha,
        "simultaneous_family_size": config.simultaneous_family_size,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "conditional_on_emulator": True,
        "development_result": True,
        "random_seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "code_revision": code_revision(repository_root),
    }
    write_json_immutable(report_path, report)
    return report
