from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import array_hash, code_revision, sha256_file, write_json_immutable
from ..models import DenseRecurrentSNN
from ..semantics import (
    ExecutionSemantics,
    IntegrationRule,
    OverflowMode,
    ResetRule,
    RoundingMode,
    ThresholdTiming,
)
from ..statistics import clopper_pearson_upper
from ..torch_emulator import TorchEmulator
from .semantic_matrix import _batched_execute, primary_semantic_conditions
from .shd import PackedSHD, _SurrogateSpike


@dataclass(frozen=True)
class SHDRepairConfig:
    schema_version: str = "SHDRepair/v2"
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 0.02
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    logit_weight: float = 0.05
    spike_weight: float = 0.02
    state_weight: float = 0.005
    regularization_weight: float = 1e-4
    supervised_learning_rate: float = 1e-3


def _seed_everything(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _quantize_ste(value: Any, numeric: Any) -> Any:
    import torch

    if not numeric.is_fixed:
        return value
    scale = float(1 << numeric.fractional_bits)
    scaled = value * scale
    if numeric.rounding is RoundingMode.NEAREST_EVEN:
        integers = torch.round(scaled)
    elif numeric.rounding is RoundingMode.FLOOR:
        integers = torch.floor(scaled)
    elif numeric.rounding is RoundingMode.TRUNCATE:
        integers = torch.trunc(scaled)
    else:
        integers = torch.round(scaled)
    minimum = -(1 << (numeric.total_bits - 1))
    maximum = (1 << (numeric.total_bits - 1)) - 1
    if numeric.overflow is OverflowMode.SATURATE:
        integers = torch.clamp(integers, minimum, maximum)
    else:
        modulus = 1 << numeric.total_bits
        integers = torch.remainder(integers - minimum, modulus) + minimum
    hard = integers / scale
    return value + (hard - value).detach()


def build_repairable_srnn(
    model: DenseRecurrentSNN,
    semantics: ExecutionSemantics,
) -> Any:
    import torch

    class RepairableSRNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "base_input_weights",
                torch.tensor(model.input_weights.copy(), dtype=torch.float32),
            )
            self.register_buffer(
                "base_recurrent_weights",
                torch.tensor(model.recurrent_weights.copy(), dtype=torch.float32),
            )
            self.register_buffer(
                "output_weights",
                torch.tensor(model.output_weights.copy(), dtype=torch.float32),
            )
            self.register_buffer(
                "reset_value", torch.tensor(model.reset_value.copy(), dtype=torch.float32)
            )
            self.log_threshold = torch.nn.Parameter(
                torch.log(torch.tensor(model.threshold.copy(), dtype=torch.float32))
            )
            self.log_tau = torch.nn.Parameter(
                torch.log(torch.tensor(model.tau_mem.copy(), dtype=torch.float32))
            )
            self.bias = torch.nn.Parameter(
                torch.tensor(model.bias.copy(), dtype=torch.float32)
            )
            self.log_incoming_scale = torch.nn.Parameter(
                torch.zeros(model.hidden_size, dtype=torch.float32)
            )
            self.log_output_scale = torch.nn.Parameter(
                torch.zeros(model.hidden_size, dtype=torch.float32)
            )

        def forward(self, events):
            threshold = torch.exp(self.log_threshold).clamp(0.05, 20.0)
            tau = torch.exp(self.log_tau).clamp(0.25, 100.0)
            scale = torch.exp(self.log_incoming_scale).clamp(0.25, 4.0)
            output_scale = torch.exp(self.log_output_scale).clamp(0.25, 4.0)
            w_in = _quantize_ste(
                self.base_input_weights * scale.unsqueeze(0), semantics.weight_format
            )
            w_rec = _quantize_ste(
                self.base_recurrent_weights * scale.unsqueeze(0),
                semantics.weight_format,
            )
            w_out = _quantize_ste(
                self.output_weights * output_scale.unsqueeze(1),
                semantics.weight_format,
            )
            bias = _quantize_ste(self.bias, semantics.state_format)
            voltage = torch.zeros(
                (events.shape[0], model.hidden_size),
                dtype=events.dtype,
                device=events.device,
            )
            spikes = torch.zeros_like(voltage)
            logits = torch.zeros(
                (events.shape[0], model.output_size),
                dtype=events.dtype,
                device=events.device,
            )
            current_queue = [
                torch.zeros_like(voltage)
                for _ in range(semantics.synaptic_delay_steps)
            ]
            output_queue = [
                torch.zeros_like(logits) for _ in range(semantics.output_delay_steps)
            ]
            state_trace = []
            spike_trace = []
            logit_trace = []
            for step in range(events.shape[1]):
                current = events[:, step] @ w_in + spikes @ w_rec + bias
                current = _quantize_ste(current, semantics.state_format)
                if current_queue:
                    current_queue.append(current)
                    current = current_queue.pop(0)

                def integrate(state):
                    if semantics.integration_rule is IntegrationRule.FORWARD_EULER:
                        return state + semantics.timestep * (-state + current) / tau
                    alpha = torch.exp(-semantics.timestep / tau)
                    return alpha * state + (1.0 - alpha) * current

                def reset(state, emitted):
                    if semantics.reset_rule is ResetRule.SUBTRACTIVE:
                        return state - emitted * threshold
                    return (1.0 - emitted) * state + emitted * self.reset_value

                if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                    spikes = _SurrogateSpike.apply(voltage - threshold)
                    voltage = integrate(reset(voltage, spikes))
                else:
                    voltage = _quantize_ste(integrate(voltage), semantics.state_format)
                    spikes = _SurrogateSpike.apply(voltage - threshold)
                    voltage = reset(voltage, spikes)
                voltage = _quantize_ste(voltage, semantics.state_format)
                contribution = _quantize_ste(spikes @ w_out, semantics.state_format)
                if output_queue:
                    output_queue.append(contribution)
                    contribution = output_queue.pop(0)
                logits = _quantize_ste(logits + contribution, semantics.state_format)
                state_trace.append(voltage)
                spike_trace.append(spikes)
                logit_trace.append(logits)
            return (
                logits,
                torch.stack(state_trace, dim=1),
                torch.stack(spike_trace, dim=1),
                torch.stack(logit_trace, dim=1),
            )

    return RepairableSRNN()


def build_supervised_target_srnn(
    model: DenseRecurrentSNN,
    semantics: ExecutionSemantics,
    *,
    initialization: str,
) -> Any:
    """Build a fully trainable target-semantics model for labeled baselines.

    ``source`` is the per-platform QAT/fine-tuning baseline. ``random`` is the
    fully supervised, from-scratch target-retraining baseline. Both execute the
    declared target semantics during every forward pass and use STEs for exact
    fixed-point transitions.
    """

    import torch

    if initialization not in {"source", "random"}:
        raise ValueError("initialization must be 'source' or 'random'")

    class SupervisedTargetSRNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_weights = torch.nn.Parameter(
                torch.tensor(model.input_weights.copy(), dtype=torch.float32)
            )
            self.recurrent_weights = torch.nn.Parameter(
                torch.tensor(model.recurrent_weights.copy(), dtype=torch.float32)
            )
            self.output_weights = torch.nn.Parameter(
                torch.tensor(model.output_weights.copy(), dtype=torch.float32)
            )
            self.bias = torch.nn.Parameter(
                torch.tensor(model.bias.copy(), dtype=torch.float32)
            )
            self.log_threshold = torch.nn.Parameter(
                torch.log(torch.tensor(model.threshold.copy(), dtype=torch.float32))
            )
            self.log_tau = torch.nn.Parameter(
                torch.log(torch.tensor(model.tau_mem.copy(), dtype=torch.float32))
            )
            self.register_buffer(
                "reset_value", torch.tensor(model.reset_value.copy(), dtype=torch.float32)
            )
            if initialization == "random":
                torch.nn.init.xavier_uniform_(self.input_weights)
                torch.nn.init.orthogonal_(self.recurrent_weights)
                with torch.no_grad():
                    self.recurrent_weights.mul_(0.25)
                torch.nn.init.xavier_uniform_(self.output_weights)
                torch.nn.init.zeros_(self.bias)

        def forward(self, events):
            threshold = torch.exp(self.log_threshold).clamp(0.05, 20.0)
            tau = torch.exp(self.log_tau).clamp(0.25, 100.0)
            w_in = _quantize_ste(self.input_weights, semantics.weight_format)
            w_rec = _quantize_ste(self.recurrent_weights, semantics.weight_format)
            w_out = _quantize_ste(self.output_weights, semantics.weight_format)
            bias = _quantize_ste(self.bias, semantics.state_format)
            voltage = torch.zeros(
                (events.shape[0], model.hidden_size),
                dtype=events.dtype,
                device=events.device,
            )
            spikes = torch.zeros_like(voltage)
            logits = torch.zeros(
                (events.shape[0], model.output_size),
                dtype=events.dtype,
                device=events.device,
            )
            current_queue = [
                torch.zeros_like(voltage)
                for _ in range(semantics.synaptic_delay_steps)
            ]
            output_queue = [
                torch.zeros_like(logits) for _ in range(semantics.output_delay_steps)
            ]
            state_trace = []
            spike_trace = []
            logit_trace = []
            for step in range(events.shape[1]):
                current = events[:, step] @ w_in + spikes @ w_rec + bias
                current = _quantize_ste(current, semantics.state_format)
                if current_queue:
                    current_queue.append(current)
                    current = current_queue.pop(0)

                def integrate(state):
                    if semantics.integration_rule is IntegrationRule.FORWARD_EULER:
                        return state + semantics.timestep * (-state + current) / tau
                    alpha = torch.exp(-semantics.timestep / tau)
                    return alpha * state + (1.0 - alpha) * current

                def reset(state, emitted):
                    if semantics.reset_rule is ResetRule.SUBTRACTIVE:
                        return state - emitted * threshold
                    return (1.0 - emitted) * state + emitted * self.reset_value

                if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                    spikes = _SurrogateSpike.apply(voltage - threshold)
                    voltage = integrate(reset(voltage, spikes))
                else:
                    voltage = _quantize_ste(integrate(voltage), semantics.state_format)
                    spikes = _SurrogateSpike.apply(voltage - threshold)
                    voltage = reset(voltage, spikes)
                voltage = _quantize_ste(voltage, semantics.state_format)
                contribution = _quantize_ste(spikes @ w_out, semantics.state_format)
                if output_queue:
                    output_queue.append(contribution)
                    contribution = output_queue.pop(0)
                logits = _quantize_ste(logits + contribution, semantics.state_format)
                state_trace.append(voltage)
                spike_trace.append(spikes)
                logit_trace.append(logits)
            return (
                logits,
                torch.stack(state_trace, dim=1),
                torch.stack(spike_trace, dim=1),
                torch.stack(logit_trace, dim=1),
            )

    return SupervisedTargetSRNN()


def _collect_reference_trace(
    model: DenseRecurrentSNN,
    store: PackedSHD,
    indices: np.ndarray,
    semantics: ExecutionSemantics,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    import torch

    emulator = TorchEmulator(device=device, dtype=torch.float32)
    collected: dict[str, list[np.ndarray]] = {
        "membrane": [],
        "spikes": [],
        "logits_over_time": [],
        "final_logits": [],
        "predictions": [],
    }
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        trace = emulator.run(model, store.frames(batch_indices), semantics).numpy()
        for key in collected:
            collected[key].append(np.asarray(getattr(trace, key)))
    return {key: np.concatenate(values) for key, values in collected.items()}


def _export_repaired(module: Any, source: DenseRecurrentSNN, name: str) -> DenseRecurrentSNN:
    import torch

    scale = torch.exp(module.log_incoming_scale).detach().cpu().double().numpy()
    output_scale = torch.exp(module.log_output_scale).detach().cpu().double().numpy()
    return source.with_parameters(
        input_weights=source.input_weights * scale[None, :],
        recurrent_weights=source.recurrent_weights * scale[None, :],
        output_weights=source.output_weights * output_scale[:, None],
        threshold=torch.exp(module.log_threshold).detach().cpu().double().numpy(),
        tau_mem=torch.exp(module.log_tau).detach().cpu().double().numpy(),
        bias=module.bias.detach().cpu().double().numpy(),
        name=name,
    )


def _export_supervised(
    module: Any, source: DenseRecurrentSNN, name: str
) -> DenseRecurrentSNN:
    return source.with_parameters(
        input_weights=module.input_weights.detach().cpu().double().numpy(),
        recurrent_weights=module.recurrent_weights.detach().cpu().double().numpy(),
        output_weights=module.output_weights.detach().cpu().double().numpy(),
        threshold=np.exp(module.log_threshold.detach().cpu().double().numpy()),
        tau_mem=np.exp(module.log_tau.detach().cpu().double().numpy()),
        bias=module.bias.detach().cpu().double().numpy(),
        name=name,
    )


def _evaluate_candidate(
    candidate: DenseRecurrentSNN,
    target: ExecutionSemantics,
    train_store: PackedSHD,
    test_store: PackedSHD,
    audit_indices: np.ndarray,
    test_indices: np.ndarray,
    reference_audit: np.ndarray,
    reference_test: np.ndarray,
    confidence_alpha: float,
    device: str,
) -> dict[str, Any]:
    audit_predictions, _ = _batched_execute(
        candidate, train_store, audit_indices, target, 256, device
    )
    test_predictions, _ = _batched_execute(
        candidate, test_store, test_indices, target, 256, device
    )
    disagreements = int(np.count_nonzero(audit_predictions != reference_audit))
    reference_accuracy = float(
        np.mean(reference_test == test_store.labels[test_indices])
    )
    target_accuracy = float(
        np.mean(test_predictions == test_store.labels[test_indices])
    )
    return {
        "model_hash": candidate.model_hash,
        "audit_disagreements": disagreements,
        "audit_disagreement_rate": disagreements / len(audit_indices),
        "certificate_upper_bound": clopper_pearson_upper(
            disagreements, len(audit_indices), confidence_alpha
        ),
        "reference_test_accuracy": reference_accuracy,
        "target_test_accuracy": target_accuracy,
        "accuracy_loss": reference_accuracy - target_accuracy,
        "audit_predictions": audit_predictions,
        "test_predictions": test_predictions,
    }


def run_shd_repair(
    model_path: str | Path,
    train_store: PackedSHD,
    test_store: PackedSHD,
    split_indices_path: str | Path,
    semantic_predictions_path: str | Path,
    condition: str,
    method: str,
    output_dir: str | Path,
    repository_root: str | Path,
    config: SHDRepairConfig,
    *,
    seed: int,
    confidence_alpha: float = 0.05,
) -> dict[str, Any]:
    import torch

    supported_methods = {
        "certificate_directed",
        "logit_only",
        "global_threshold",
        "per_platform_qat",
        "supervised_target_retraining",
    }
    if method not in supported_methods:
        raise ValueError("unsupported repair method")
    conditions = primary_semantic_conditions()
    if condition == "reference" or condition not in conditions:
        raise ValueError("repair requires a declared non-reference condition")
    target = conditions[condition]
    reference_semantics = conditions["reference"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "repair_report.json"
    model_output_path = output_dir / "repaired_model.npz"
    predictions_output_path = output_dir / "repaired_predictions.npz"
    if any(path.exists() for path in (report_path, model_output_path, predictions_output_path)):
        raise FileExistsError(f"repair output already exists: {output_dir}")
    source_model = DenseRecurrentSNN.load(model_path)
    with np.load(split_indices_path, allow_pickle=False) as split_data:
        calibration_indices = np.asarray(
            split_data["repair_calibration"], dtype=np.int64
        )
        audit_indices = np.asarray(split_data["certificate_audit"], dtype=np.int64)
        test_indices = np.asarray(split_data["test"], dtype=np.int64)
    with np.load(semantic_predictions_path, allow_pickle=False) as source_predictions:
        reference_audit = np.asarray(source_predictions["pred__audit__reference"])
        reference_calibration = np.asarray(
            source_predictions["pred__calibration__reference"]
        )
        reference_test = np.asarray(source_predictions["pred__test__reference"])
        target_audit_before = np.asarray(
            source_predictions[f"pred__audit__{condition}"]
        )
        target_test_before = np.asarray(source_predictions[f"pred__test__{condition}"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(seed)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    optimization_steps = 0
    label_budget = 0
    selection_criterion = "reference prediction disagreement"

    if method == "global_threshold":
        best_model = source_model
        best_disagreement = len(calibration_indices) + 1
        best_scale = 1.0
        for scale in np.linspace(0.5, 1.5, 21):
            candidate = source_model.with_parameters(
                threshold=source_model.threshold * scale,
                name=f"{source_model.name}-global-threshold-{scale:.2f}",
            )
            predictions, _ = _batched_execute(
                candidate,
                train_store,
                calibration_indices,
                target,
                config.batch_size,
                device,
            )
            disagreement = int(np.count_nonzero(predictions != reference_calibration))
            history.append(
                {"scale": float(scale), "calibration_disagreements": disagreement}
            )
            if disagreement < best_disagreement:
                best_disagreement = disagreement
                best_model = candidate
                best_scale = float(scale)
        selected = {"global_threshold_scale": best_scale}
    elif method in {"certificate_directed", "logit_only"}:
        teacher = _collect_reference_trace(
            source_model,
            train_store,
            calibration_indices,
            reference_semantics,
            config.batch_size,
            device,
        )
        module = build_repairable_srnn(source_model, target).to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        rng = np.random.default_rng(seed)
        best_disagreement = len(calibration_indices) + 1
        best_state = copy.deepcopy(module.state_dict())
        for epoch in range(config.epochs):
            order = rng.permutation(len(calibration_indices))
            total_loss = 0.0
            seen = 0
            for start in range(0, len(order), config.batch_size):
                positions = order[start : start + config.batch_size]
                sample_indices = calibration_indices[positions]
                events = torch.as_tensor(
                    train_store.frames(sample_indices), device=device
                )
                teacher_logits = torch.as_tensor(
                    teacher["final_logits"][positions], device=device
                )
                teacher_states = torch.as_tensor(
                    teacher["membrane"][positions], device=device
                )
                teacher_spikes = torch.as_tensor(
                    teacher["spikes"][positions], device=device
                )
                teacher_predictions = torch.as_tensor(
                    teacher["predictions"][positions],
                    dtype=torch.long,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits, states, spikes, _ = module(events)
                margin_loss = torch.nn.functional.cross_entropy(
                    logits, teacher_predictions
                )
                logit_loss = torch.nn.functional.smooth_l1_loss(
                    logits, teacher_logits
                ) / (torch.mean(torch.abs(teacher_logits)) + 1e-6)
                if method == "certificate_directed":
                    spike_loss = torch.mean(torch.abs(spikes - teacher_spikes))
                    state_loss = torch.mean(torch.abs(states - teacher_states)) / (
                        torch.mean(torch.abs(teacher_states)) + 1e-6
                    )
                else:
                    spike_loss = torch.zeros((), device=device)
                    state_loss = torch.zeros((), device=device)
                regularization = (
                    torch.mean(module.log_threshold**2)
                    + torch.mean((module.log_tau - np.log(5.0)) ** 2)
                    + torch.mean(module.log_incoming_scale**2)
                    + torch.mean(module.log_output_scale**2)
                )
                if method == "logit_only":
                    loss = logit_loss + config.regularization_weight * regularization
                else:
                    loss = (
                        margin_loss
                        + config.logit_weight * logit_loss
                        + config.spike_weight * spike_loss
                        + config.state_weight * state_loss
                        + config.regularization_weight * regularization
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(), config.gradient_clip
                )
                optimizer.step()
                optimization_steps += 1
                total_loss += float(loss.detach()) * len(positions)
                seen += len(positions)
            module.eval()
            calibration_predictions: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(calibration_indices), config.batch_size):
                    sample_indices = calibration_indices[
                        start : start + config.batch_size
                    ]
                    events = torch.as_tensor(
                        train_store.frames(sample_indices), device=device
                    )
                    logits, _, _, _ = module(events)
                    calibration_predictions.append(
                        torch.argmax(logits, dim=1).cpu().numpy()
                    )
            calibration_predictions_array = np.concatenate(calibration_predictions)
            disagreement = int(
                np.count_nonzero(
                    calibration_predictions_array != reference_calibration
                )
            )
            record = {
                "epoch": epoch + 1,
                "loss": total_loss / seen,
                "calibration_disagreements": disagreement,
            }
            history.append(record)
            print(
                f"repair {method} {condition} epoch={epoch + 1}/{config.epochs} "
                f"loss={record['loss']:.4f} disagreements={disagreement}",
                flush=True,
            )
            if disagreement < best_disagreement:
                best_disagreement = disagreement
                best_state = copy.deepcopy(module.state_dict())
            module.train()
        module.load_state_dict(best_state)
        best_model = _export_repaired(
            module, source_model, f"{source_model.name}-{method}-{condition}"
        )
        selected = {
            "best_calibration_disagreements": best_disagreement,
            "threshold_ratio_mean": float(
                np.mean(best_model.threshold / source_model.threshold)
            ),
            "tau_ratio_mean": float(np.mean(best_model.tau_mem / source_model.tau_mem)),
            "output_weight_scale_mean": float(
                np.mean(
                    np.linalg.norm(best_model.output_weights, axis=1)
                    / (np.linalg.norm(source_model.output_weights, axis=1) + 1e-12)
                )
            ),
        }
    else:
        initialization = "source" if method == "per_platform_qat" else "random"
        module = build_supervised_target_srnn(
            source_model, target, initialization=initialization
        ).to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=config.supervised_learning_rate,
            weight_decay=config.weight_decay,
        )
        rng = np.random.default_rng(seed)
        label_budget = len(calibration_indices)
        selection_criterion = "labeled calibration accuracy"
        best_accuracy = -1.0
        best_loss = float("inf")
        best_state = copy.deepcopy(module.state_dict())
        for epoch in range(config.epochs):
            order = rng.permutation(calibration_indices)
            total_loss = 0.0
            seen = 0
            for start in range(0, len(order), config.batch_size):
                sample_indices = order[start : start + config.batch_size]
                events = torch.as_tensor(
                    train_store.frames(sample_indices), device=device
                )
                targets = torch.as_tensor(
                    train_store.labels[sample_indices], dtype=torch.long, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                logits, _, _, _ = module(events)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(), config.gradient_clip
                )
                optimizer.step()
                optimization_steps += 1
                total_loss += float(loss.detach()) * len(sample_indices)
                seen += len(sample_indices)
            module.eval()
            calibration_predictions: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(calibration_indices), config.batch_size):
                    sample_indices = calibration_indices[
                        start : start + config.batch_size
                    ]
                    events = torch.as_tensor(
                        train_store.frames(sample_indices), device=device
                    )
                    logits, _, _, _ = module(events)
                    calibration_predictions.append(
                        torch.argmax(logits, dim=1).cpu().numpy()
                    )
            calibration_predictions_array = np.concatenate(calibration_predictions)
            accuracy = float(
                np.mean(
                    calibration_predictions_array
                    == train_store.labels[calibration_indices]
                )
            )
            epoch_loss = total_loss / seen
            record = {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "calibration_accuracy": accuracy,
            }
            history.append(record)
            print(
                f"repair {method} {condition} epoch={epoch + 1}/{config.epochs} "
                f"loss={epoch_loss:.4f} accuracy={accuracy:.4f}",
                flush=True,
            )
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and epoch_loss < best_loss
            ):
                best_accuracy = accuracy
                best_loss = epoch_loss
                best_state = copy.deepcopy(module.state_dict())
            module.train()
        module.load_state_dict(best_state)
        best_model = _export_supervised(
            module, source_model, f"{source_model.name}-{method}-{condition}"
        )
        selected = {
            "initialization": initialization,
            "best_calibration_accuracy": best_accuracy,
            "best_calibration_loss": best_loss,
        }

    best_model.save(model_output_path)
    after = _evaluate_candidate(
        best_model,
        target,
        train_store,
        test_store,
        audit_indices,
        test_indices,
        reference_audit,
        reference_test,
        confidence_alpha,
        device,
    )
    before_disagreements = int(
        np.count_nonzero(target_audit_before != reference_audit)
    )
    reference_accuracy = float(
        np.mean(reference_test == test_store.labels[test_indices])
    )
    before_accuracy = float(
        np.mean(target_test_before == test_store.labels[test_indices])
    )
    before_loss = reference_accuracy - before_accuracy
    after_loss = after["accuracy_loss"]
    recovery = (
        (before_loss - after_loss) / before_loss if before_loss > 0 else float("nan")
    )
    with predictions_output_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            audit_predictions=after.pop("audit_predictions"),
            test_predictions=after.pop("test_predictions"),
            audit_indices=audit_indices,
            test_indices=test_indices,
        )
    if method == "global_threshold":
        permitted_changes = ["global_threshold_scale"]
    elif method in {"certificate_directed", "logit_only"}:
        permitted_changes = [
            "per_neuron_threshold",
            "per_neuron_tau_mem",
            "per_neuron_bias",
            "per_neuron_incoming_weight_scale",
            "per_neuron_output_weight_scale",
        ]
    else:
        permitted_changes = [
            "all_input_weights",
            "all_recurrent_weights",
            "all_output_weights",
            "per_neuron_threshold",
            "per_neuron_tau_mem",
            "per_neuron_bias",
        ]
    quantization_active = bool(
        target.state_format.is_fixed or target.weight_format.is_fixed
    )
    report = {
        "schema_version": "SHDRepairExperiment/v2",
        "method": method,
        "condition": condition,
        "target_semantics_hash": target.semantics_hash,
        "source_model_hash": source_model.model_hash,
        "repaired_model_hash": best_model.model_hash,
        "source_model_artifact_hash": sha256_file(model_path),
        "repaired_model_artifact_hash": sha256_file(model_output_path),
        "predictions_artifact_hash": sha256_file(predictions_output_path),
        "calibration_samples": len(calibration_indices),
        "audit_samples": len(audit_indices),
        "label_budget": label_budget,
        "labels_used": "none" if label_budget == 0 else "repair calibration labels",
        "selection_criterion": selection_criterion,
        "test_labels_used_for_selection": False,
        "permitted_parameter_changes": permitted_changes,
        "quantization_active": quantization_active,
        "baseline_definition": (
            "source-initialized target-semantics supervised fine-tuning with STE; "
            "a QAT baseline when the target declares fixed-point formats"
            if method == "per_platform_qat"
            else "random-initialized target-semantics supervised training"
            if method == "supervised_target_retraining"
            else "label-free reference imitation and calibration"
            if method in {"certificate_directed", "logit_only"}
            else "label-free hard-semantics grid search"
        ),
        "optimization_steps": optimization_steps,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in module.parameters())
        )
        if method != "global_threshold"
        else 1,
        "random_seed": seed,
        "calibration_indices_hash": array_hash(calibration_indices),
        "audit_indices_hash": array_hash(audit_indices),
        "test_indices_hash": array_hash(test_indices),
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "split_indices_artifact_hash": sha256_file(split_indices_path),
        "semantic_predictions_artifact_hash": sha256_file(
            semantic_predictions_path
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
        "after": after,
        "accuracy_recovery_fraction": recovery,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "calibration_audit_disjoint": True,
        "conditional_on_emulator": True,
        "device": device,
        "torch_version": torch.__version__,
        "code_revision": code_revision(repository_root),
    }
    write_json_immutable(report_path, report)
    return report
