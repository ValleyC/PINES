from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .artifacts import code_revision
from .certificates import CertificateEngine, SemanticsFamily, exact_identity_mapping
from .differentiable_repair import DifferentiableSurrogateRepair
from .emulator import ExecutionTrace, VectorizedEmulator
from .models import DenseRecurrentSNN
from .reports import CertificateReport, RepairReport
from .semantics import ExecutionSemantics


@dataclass(frozen=True)
class RepairOutcome:
    model: DenseRecurrentSNN
    report: RepairReport
    post_repair_certificate: CertificateReport


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0)


def uncertified_mass_surrogate(reference_logits: np.ndarray, target_logits: np.ndarray) -> float:
    reference_prediction = np.argmax(reference_logits, axis=1)
    rows = np.arange(reference_logits.shape[0])
    reference_chosen = reference_logits[rows, reference_prediction]
    reference_masked = reference_logits.copy()
    reference_masked[rows, reference_prediction] = -np.inf
    reference_margin = reference_chosen - np.max(reference_masked, axis=1)
    target_chosen = target_logits[rows, reference_prediction]
    target_masked = target_logits.copy()
    target_masked[rows, reference_prediction] = -np.inf
    target_margin = target_chosen - np.max(target_masked, axis=1)
    importance = 1.0 / (1.0 + np.exp(-reference_margin))
    return float(np.mean(importance * _softplus(-target_margin)))


def certificate_directed_objective(
    reference: ExecutionTrace, target: ExecutionTrace
) -> float:
    """Label-free margin objective with state, spike, and logit trace terms."""

    margin_mass = uncertified_mass_surrogate(
        reference.final_logits, target.final_logits
    )
    spike_distance = float(np.mean(np.abs(reference.spikes - target.spikes)))
    state_scale = float(np.mean(np.abs(reference.membrane))) + 1e-9
    state_distance = float(
        np.mean(np.abs(reference.membrane - target.membrane)) / state_scale
    )
    logit_scale = float(np.mean(np.abs(reference.logits_over_time))) + 1e-9
    logit_distance = float(
        np.mean(np.abs(reference.logits_over_time - target.logits_over_time))
        / logit_scale
    )
    return margin_mass + 0.10 * spike_distance + 0.02 * state_distance + 0.02 * logit_distance


class CertificateDirectedRepair:
    """Label-free coordinate calibration followed by untouched-audit certification."""

    def __init__(self, emulator: VectorizedEmulator | None = None) -> None:
        self.emulator = emulator or VectorizedEmulator()

    def repair(
        self,
        model: DenseRecurrentSNN,
        calibration_inputs: np.ndarray,
        audit_inputs: np.ndarray,
        calibration_ids: Iterable[str],
        audit_ids: Iterable[str],
        reference: ExecutionSemantics,
        target: ExecutionSemantics,
        source_certificate: CertificateReport,
        *,
        delta: float = 0.05,
        decision_budget: float = 0.05,
        threshold_scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
        tau_scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
        bias_offsets: tuple[float, ...] = (-0.1, -0.05, 0.0, 0.05, 0.1),
        incoming_weight_scales: tuple[float, ...] = (0.9, 1.0, 1.1),
        differentiable_steps: int = 50,
        repository_root: str | None = None,
    ) -> RepairOutcome:
        calibration_ids = tuple(str(item) for item in calibration_ids)
        audit_ids = tuple(str(item) for item in audit_ids)
        overlap = set(calibration_ids).intersection(audit_ids)
        if overlap:
            raise ValueError(f"calibration and audit IDs overlap: {sorted(overlap)[:3]}")
        if len(calibration_ids) != len(calibration_inputs):
            raise ValueError("calibration IDs must match calibration samples")
        if len(audit_ids) != len(audit_inputs):
            raise ValueError("audit IDs must match audit samples")
        identity = exact_identity_mapping(reference, target)
        reference_trace = self.emulator.run(model, calibration_inputs, reference)
        initial_trace = self.emulator.run(model, calibration_inputs, target)
        initial_objective = certificate_directed_objective(reference_trace, initial_trace)
        best_model = model
        best_objective = initial_objective
        changes: list[dict[str, float | int | str]] = []
        evaluations = 1
        started = time.perf_counter()

        if not identity.equivalent:
            if differentiable_steps > 0:
                differentiable = DifferentiableSurrogateRepair().calibrate(
                    model,
                    calibration_inputs,
                    reference,
                    target,
                    steps=differentiable_steps,
                )
                differentiable_trace = self.emulator.run(
                    differentiable.model, calibration_inputs, target
                )
                differentiable_objective = certificate_directed_objective(
                    reference_trace, differentiable_trace
                )
                evaluations += differentiable.steps
                if differentiable_objective + 1e-12 < best_objective:
                    best_model = differentiable.model
                    best_objective = differentiable_objective
                    changes.append(
                        {
                            "neuron": -1,
                            "parameter": "differentiable_surrogate",
                            "value": float(differentiable.loss_history[-1]),
                        }
                    )
            stages = (
                ("threshold_scale", threshold_scales),
                ("tau_scale", tau_scales),
                ("bias_offset", bias_offsets),
                ("incoming_weight_scale", incoming_weight_scales),
            )
            for parameter, values_to_try in stages:
                for neuron in range(model.hidden_size):
                    candidates: list[tuple[float, DenseRecurrentSNN]] = []
                    for value in values_to_try:
                        if parameter == "threshold_scale":
                            values = best_model.threshold.copy()
                            values[neuron] = best_model.threshold[neuron] * value
                            candidate = best_model.with_parameters(threshold=values)
                        elif parameter == "tau_scale":
                            values = best_model.tau_mem.copy()
                            values[neuron] = best_model.tau_mem[neuron] * value
                            candidate = best_model.with_parameters(tau_mem=values)
                        elif parameter == "bias_offset":
                            values = best_model.bias.copy()
                            values[neuron] = best_model.bias[neuron] + value
                            candidate = best_model.with_parameters(bias=values)
                        else:
                            input_weights = best_model.input_weights.copy()
                            recurrent_weights = best_model.recurrent_weights.copy()
                            input_weights[:, neuron] = best_model.input_weights[:, neuron] * value
                            recurrent_weights[:, neuron] = (
                                best_model.recurrent_weights[:, neuron] * value
                            )
                            candidate = best_model.with_parameters(
                                input_weights=input_weights,
                                recurrent_weights=recurrent_weights,
                            )
                        candidates.append((value, candidate))
                    selected = None
                    for value, candidate in candidates:
                        trace = self.emulator.run(candidate, calibration_inputs, target)
                        objective = certificate_directed_objective(
                            reference_trace, trace
                        )
                        evaluations += 1
                        if objective + 1e-12 < best_objective:
                            best_objective = objective
                            selected = (value, candidate)
                    if selected is not None:
                        value, best_model = selected
                        changes.append(
                            {
                                "neuron": neuron,
                                "parameter": parameter,
                                "value": float(value),
                            }
                        )

        family = SemanticsFamily((target,), name="post-repair-target")
        post_certificate = CertificateEngine(self.emulator).build_report(
            best_model,
            audit_inputs,
            reference,
            family,
            delta=delta,
            decision_budget=decision_budget,
            dataset_split="certificate-audit",
            checkpoint_file=best_model.model_description,
            seed_manifest={"reference": reference.randomness.seed, "target": target.randomness.seed},
            repository_root=repository_root,
        )
        elapsed = time.perf_counter() - started
        report = RepairReport(
            schema_version="RepairReport/v1",
            source_certificate=source_certificate.report_description,
            permitted_parameter_changes=(
                "threshold",
                "tau_mem",
                "bias",
                "incoming_weight_scale",
            ),
            calibration_split="repair-calibration",
            audit_split="certificate-audit",
            data_budget=len(calibration_inputs),
            label_budget=0,
            optimization_evaluations=evaluations,
            optimization_seconds=elapsed,
            original_model=model.model_description,
            repaired_model=best_model.model_description,
            pre_repair_objective=initial_objective,
            post_repair_objective=best_objective,
            post_repair_certificate=post_certificate.report_description,
            selected_changes=tuple(changes),
            code_revision=code_revision(repository_root),
        )
        return RepairOutcome(best_model, report, post_certificate)
