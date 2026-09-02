from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .artifacts import array_description, code_revision, config_description
from .emulator import VectorizedEmulator, _decay_and_drive
from .models import DenseRecurrentSNN
from .reports import CertificateReport
from .semantics import ExecutionSemantics
from .statistics import (
    bonferroni_alpha,
    compose_physical_bound,
    disagreement_bound,
)


@dataclass(frozen=True)
class SemanticsFamily:
    members: tuple[ExecutionSemantics, ...]
    name: str = "bounded-semantics-family"

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("a semantics family must have at least one member")
        descriptions = [member.semantics_description for member in self.members]
        if len(descriptions) != len(set(descriptions)):
            raise ValueError("semantics family members must be unique")

    @property
    def family_description(self) -> str:
        return config_description(
            {"name": self.name, "members": [m.semantics_description for m in self.members]}
        )


@dataclass(frozen=True)
class FamilyCertificateResult:
    certified: np.ndarray
    certified_fraction: float
    reference_predictions: np.ndarray
    member_predictions: np.ndarray
    logit_lower: np.ndarray
    logit_upper: np.ndarray
    reference_margin: np.ndarray
    worst_logit_perturbation: np.ndarray


@dataclass(frozen=True)
class EquivalenceResult:
    equivalent: bool
    scope: str
    proof_kind: str
    mapping: dict[str, float]
    counterexample: tuple[float, ...] | None = None


def exact_identity_mapping(
    source: ExecutionSemantics, target: ExecutionSemantics
) -> EquivalenceResult:
    if source.semantics_description == target.semantics_description:
        return EquivalenceResult(True, "all supported models and inputs", "structural identity", {})
    return EquivalenceResult(
        False,
        "unrestricted",
        "no general mapping proved",
        {},
    )


def exhaustive_trace_equivalence(
    model: DenseRecurrentSNN,
    source: ExecutionSemantics,
    target: ExecutionSemantics,
    input_alphabet: Sequence[float],
    horizon: int,
    compare: str = "trajectory",
) -> EquivalenceResult:
    """Exhaust all scalar input traces in the declared finite domain."""

    if model.input_size != 1:
        raise ValueError("small-state checker currently requires one input channel")
    if horizon <= 0 or not input_alphabet:
        raise ValueError("horizon and input alphabet must be non-empty")
    emulator = VectorizedEmulator()
    for sequence in itertools.product(input_alphabet, repeat=horizon):
        inputs = np.asarray(sequence, dtype=np.float64).reshape(1, horizon, 1)
        left = emulator.run(model, inputs, source)
        right = emulator.run(model, inputs, target)
        if compare == "prediction":
            equal = np.array_equal(left.predictions, right.predictions)
        elif compare == "trajectory":
            equal = np.array_equal(left.spikes, right.spikes) and np.allclose(
                left.membrane, right.membrane, atol=0.0, rtol=0.0
            )
        else:
            raise ValueError("compare must be prediction or trajectory")
        if not equal:
            return EquivalenceResult(
                False,
                f"alphabet={tuple(input_alphabet)}, horizon={horizon}",
                "exhaustive finite-domain counterexample",
                {},
                tuple(float(value) for value in sequence),
            )
    return EquivalenceResult(
        True,
        f"alphabet={tuple(input_alphabet)}, horizon={horizon}",
        "exhaustive finite-domain proof",
        {},
    )


def _one_neuron_transition(
    model: DenseRecurrentSNN,
    semantics: ExecutionSemantics,
    state: float,
    input_value: float,
    previous_spike: float,
) -> tuple[float, float, tuple[float, ...]]:
    from .semantics import ResetRule, ThresholdTiming

    if model.hidden_size != 1 or model.input_size != 1:
        raise ValueError("finite-state transition checker requires one input and neuron")
    if semantics.synaptic_delay_steps or semantics.output_delay_steps:
        raise ValueError("finite-state transition checker requires delay-free semantics")
    rng = np.random.default_rng(semantics.randomness.seed)
    weight = semantics.weight_format
    numeric = semantics.state_format
    w_in = float(weight.quantize(model.input_weights[0, 0], rng))
    w_rec = float(weight.quantize(model.recurrent_weights[0, 0], rng))
    w_out = np.asarray(weight.quantize(model.output_weights[0], rng))
    current = float(
        numeric.quantize(
            input_value * w_in + previous_spike * w_rec + model.bias[0], rng
        )
    )
    voltage = float(numeric.quantize(state, rng))
    threshold = float(model.threshold[0])

    def apply_reset(value: float) -> float:
        if semantics.reset_rule is ResetRule.SUBTRACTIVE:
            return value - threshold
        return float(model.reset_value[0])

    if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
        spike = float(voltage >= threshold)
        if spike:
            voltage = apply_reset(voltage)
        voltage = float(
            _decay_and_drive(voltage, current, model.tau_mem[0], semantics)
        )
    else:
        voltage = float(
            _decay_and_drive(voltage, current, model.tau_mem[0], semantics)
        )
        voltage = float(numeric.quantize(voltage, rng))
        spike = float(voltage >= threshold)
        if spike:
            voltage = apply_reset(voltage)
    voltage = float(numeric.quantize(voltage, rng))
    output = tuple(float(value) for value in np.asarray(numeric.quantize(spike * w_out, rng)))
    return voltage, spike, output


def exhaustive_state_equivalence(
    model: DenseRecurrentSNN,
    source: ExecutionSemantics,
    target: ExecutionSemantics,
    state_alphabet: Sequence[float],
    input_alphabet: Sequence[float],
    previous_spike_alphabet: Sequence[float] = (0.0, 1.0),
) -> EquivalenceResult:
    """Prove one-step equality over every declared finite machine state/input."""

    if not state_alphabet or not input_alphabet or not previous_spike_alphabet:
        raise ValueError("finite alphabets must be non-empty")
    scope = (
        f"states={tuple(state_alphabet)}, inputs={tuple(input_alphabet)}, "
        f"previous_spikes={tuple(previous_spike_alphabet)}"
    )
    for state, input_value, previous_spike in itertools.product(
        state_alphabet, input_alphabet, previous_spike_alphabet
    ):
        left = _one_neuron_transition(
            model, source, state, input_value, previous_spike
        )
        right = _one_neuron_transition(
            model, target, state, input_value, previous_spike
        )
        if left != right:
            return EquivalenceResult(
                False,
                scope,
                "exhaustive finite-state counterexample",
                {},
                (float(state), float(input_value), float(previous_spike)),
            )
    return EquivalenceResult(
        True,
        scope,
        "exhaustive finite-state transition proof",
        {},
    )


class CertificateEngine:
    def __init__(self, emulator: VectorizedEmulator | None = None) -> None:
        self.emulator = emulator or VectorizedEmulator()

    def certify_family(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        family: SemanticsFamily,
    ) -> FamilyCertificateResult:
        reference_trace = self.emulator.run(model, inputs, reference)
        member_traces = [
            self.emulator.run(model, inputs, semantics) for semantics in family.members
        ]
        member_predictions = np.stack(
            [trace.predictions for trace in member_traces], axis=0
        )
        all_logits = np.stack(
            [reference_trace.final_logits]
            + [trace.final_logits for trace in member_traces],
            axis=0,
        )
        lower = np.min(all_logits, axis=0)
        upper = np.max(all_logits, axis=0)
        certified = np.all(
            member_predictions == reference_trace.predictions[None, :], axis=0
        )
        sorted_logits = np.sort(reference_trace.final_logits, axis=1)
        margins = sorted_logits[:, -1] - sorted_logits[:, -2]
        perturbation = np.max(
            np.abs(all_logits - reference_trace.final_logits[None, :, :]), axis=(0, 2)
        )
        return FamilyCertificateResult(
            certified=certified,
            certified_fraction=float(np.mean(certified)),
            reference_predictions=reference_trace.predictions,
            member_predictions=member_predictions,
            logit_lower=lower,
            logit_upper=upper,
            reference_margin=margins,
            worst_logit_perturbation=perturbation,
        )

    def build_report(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        family: SemanticsFamily,
        *,
        delta: float,
        decision_budget: float,
        dataset_split: str,
        checkpoint_file: str,
        seed_manifest: object,
        hardware_predictions: np.ndarray | None = None,
        hardware_target_index: int = 0,
        firmware_version: str | None = None,
        bitstream_file: str | None = None,
        repository_root: str | Path | None = None,
        static_certified_fraction: float | None = None,
        static_family: str | None = None,
    ) -> CertificateReport:
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0,1)")
        if not 0 <= decision_budget <= 1:
            raise ValueError("decision_budget must be in [0,1]")
        result = self.certify_family(model, inputs, reference, family)
        comparisons = len(family.members)
        semantic_alpha = delta if hardware_predictions is None else delta / 2.0
        member_alpha = bonferroni_alpha(semantic_alpha, comparisons)
        member_bounds = []
        member_errors = []
        for predictions in result.member_predictions:
            bound = disagreement_bound(
                result.reference_predictions, predictions, member_alpha
            )
            member_bounds.append(bound.upper_bound)
            member_errors.append(bound.errors)
        semantic_bound = max(member_bounds)
        conformance_count: int | None = None
        conformance_bound: float | None = None
        if hardware_predictions is not None:
            if not 0 <= hardware_target_index < len(family.members):
                raise ValueError("hardware_target_index is outside the semantics family")
            conformance = disagreement_bound(
                result.member_predictions[hardware_target_index],
                np.asarray(hardware_predictions),
                delta / 2.0,
            )
            conformance_count = conformance.errors
            conformance_bound = conformance.upper_bound
            total_bound = compose_physical_bound(semantic_bound, conformance_bound)
        else:
            total_bound = semantic_bound
        return CertificateReport(
            schema_version="CertificateReport/v1",
            model_description=model.model_description,
            data_description=array_description(np.asarray(inputs)),
            dataset_split=dataset_split,
            checkpoint_file=checkpoint_file,
            reference_semantics=reference.semantics_description,
            target_semantics=tuple(
                member.semantics_description for member in family.members
            ),
            static_family=static_family or family.family_description,
            sample_count=int(np.asarray(inputs).shape[0]),
            confidence_level=1.0 - delta,
            certified_input_fraction=(
                result.certified_fraction
                if static_certified_fraction is None
                else static_certified_fraction
            ),
            semantic_disagreement_count=max(member_errors),
            semantic_bound=semantic_bound,
            conformance_disagreement_count=conformance_count,
            conformance_bound=conformance_bound,
            total_bound=total_bound,
            decision_budget=decision_budget,
            budget_verdict="accept" if total_bound <= decision_budget else "reject",
            conditional_on_emulator=hardware_predictions is None,
            code_revision=code_revision(repository_root),
            random_seed=config_description(seed_manifest),
            firmware_version=firmware_version,
            bitstream_file=bitstream_file,
            assumptions=(
                "audit examples are independent draws from the deployment distribution",
                "predictions are paired on identical inputs",
                "hardware pairs use one independent run per sampled input",
                "accuracy-change magnitude is bounded by prediction disagreement",
            ),
            member_bounds=tuple(member_bounds),
        )
