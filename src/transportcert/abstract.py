from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .artifacts import sha256_json
from .emulator import VectorizedEmulator, _validate_inputs
from .models import DenseRecurrentSNN
from .semantics import (
    ExecutionSemantics,
    IntegrationRule,
    NumericFormat,
    OverflowMode,
    ResetRule,
    RoundingMode,
    ThresholdTiming,
)


@dataclass(frozen=True)
class SemanticsBox:
    """A bounded family with discrete semantic axes and continuous ranges."""

    base: ExecutionSemantics
    timestep_bounds: tuple[float, float]
    threshold_scale_bounds: tuple[float, float] = (1.0, 1.0)
    integration_rules: tuple[IntegrationRule, ...] = (IntegrationRule.FORWARD_EULER,)
    threshold_timings: tuple[ThresholdTiming, ...] = (ThresholdTiming.POST_INTEGRATION,)
    reset_rules: tuple[ResetRule, ...] = (ResetRule.SUBTRACTIVE,)
    synaptic_delays: tuple[int, ...] = (0,)
    output_delays: tuple[int, ...] = (0,)
    name: str = "semantics-box"

    def __post_init__(self) -> None:
        dt_lower, dt_upper = self.timestep_bounds
        threshold_lower, threshold_upper = self.threshold_scale_bounds
        if not (0 < dt_lower <= dt_upper and math.isfinite(dt_upper)):
            raise ValueError("invalid timestep bounds")
        if not (0 < threshold_lower <= threshold_upper):
            raise ValueError("invalid threshold scale bounds")
        collections = (
            self.integration_rules,
            self.threshold_timings,
            self.reset_rules,
            self.synaptic_delays,
            self.output_delays,
        )
        if any(not collection for collection in collections):
            raise ValueError("every bounded semantic axis must be non-empty")
        if any(delay < 0 for delay in self.synaptic_delays + self.output_delays):
            raise ValueError("delays must be non-negative")
        if self.base.state_format.overflow is OverflowMode.WRAP:
            # Wrapping remains supported, but may force the full numeric range.
            pass

    @property
    def box_hash(self) -> str:
        return sha256_json(
            {
                "name": self.name,
                "base": self.base.semantics_hash,
                "timestep_bounds": self.timestep_bounds,
                "threshold_scale_bounds": self.threshold_scale_bounds,
                "integration_rules": self.integration_rules,
                "threshold_timings": self.threshold_timings,
                "reset_rules": self.reset_rules,
                "synaptic_delays": self.synaptic_delays,
                "output_delays": self.output_delays,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SemanticsBox":
        return cls(
            base=ExecutionSemantics.from_dict(dict(data["base"])),
            timestep_bounds=tuple(float(value) for value in data["timestep_bounds"]),
            threshold_scale_bounds=tuple(
                float(value)
                for value in data.get("threshold_scale_bounds", (1.0, 1.0))
            ),
            integration_rules=tuple(
                IntegrationRule(value)
                for value in data.get("integration_rules", ("forward_euler",))
            ),
            threshold_timings=tuple(
                ThresholdTiming(value)
                for value in data.get("threshold_timings", ("post_integration",))
            ),
            reset_rules=tuple(
                ResetRule(value)
                for value in data.get("reset_rules", ("subtractive",))
            ),
            synaptic_delays=tuple(
                int(value) for value in data.get("synaptic_delays", (0,))
            ),
            output_delays=tuple(
                int(value) for value in data.get("output_delays", (0,))
            ),
            name=str(data.get("name", "semantics-box")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticsBox":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class IntervalCertificateResult:
    certified: np.ndarray
    certified_fraction: float
    reference_predictions: np.ndarray
    target_logit_lower: np.ndarray
    target_logit_upper: np.ndarray


@dataclass(frozen=True)
class DecisionMarginCertificateResult:
    """Sound pairwise decision-margin bounds for floating-point execution."""

    certified: np.ndarray
    certified_fraction: float
    reference_predictions: np.ndarray
    target_margin_lower: np.ndarray
    target_margin_upper: np.ndarray


def partition_semantics_box(
    box: SemanticsBox,
    timestep_partitions: int,
    threshold_partitions: int,
) -> tuple[SemanticsBox, ...]:
    """Cover a continuous box by a Cartesian grid of closed sub-boxes."""

    if timestep_partitions <= 0 or threshold_partitions <= 0:
        raise ValueError("partition counts must be positive")
    timestep_edges = np.linspace(
        box.timestep_bounds[0], box.timestep_bounds[1], timestep_partitions + 1
    )
    threshold_edges = np.linspace(
        box.threshold_scale_bounds[0],
        box.threshold_scale_bounds[1],
        threshold_partitions + 1,
    )
    return tuple(
        replace(
            box,
            timestep_bounds=(
                float(timestep_edges[timestep_index]),
                float(timestep_edges[timestep_index + 1]),
            ),
            threshold_scale_bounds=(
                float(threshold_edges[threshold_index]),
                float(threshold_edges[threshold_index + 1]),
            ),
            name=(
                f"{box.name}-dt-{timestep_index + 1}-of-{timestep_partitions}"
                f"-threshold-{threshold_index + 1}-of-{threshold_partitions}"
            ),
        )
        for timestep_index, threshold_index in itertools.product(
            range(timestep_partitions), range(threshold_partitions)
        )
    )


def _linear_interval(
    lower: np.ndarray, upper: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(weights):
        shape = (*lower.shape[:-1], weights.shape[1])
        zeros = np.zeros(shape, dtype=np.result_type(lower, upper, weights))
        return zeros, zeros.copy()
    positive = np.maximum(weights, 0.0)
    negative = np.minimum(weights, 0.0)
    return lower @ positive + upper @ negative, upper @ positive + lower @ negative


def _multiply_interval(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    products = np.stack(
        (
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        ),
        axis=0,
    )
    return np.min(products, axis=0), np.max(products, axis=0)


def _quantize_interval(
    lower: np.ndarray, upper: np.ndarray, numeric: NumericFormat
) -> tuple[np.ndarray, np.ndarray]:
    if not numeric.is_fixed:
        return lower, upper
    assert numeric.total_bits is not None
    assert numeric.fractional_bits is not None
    scale = float(1 << numeric.fractional_bits)
    minimum = -(1 << (numeric.total_bits - 1)) / scale
    maximum = ((1 << (numeric.total_bits - 1)) - 1) / scale
    if numeric.overflow is OverflowMode.WRAP and (
        np.any(lower < minimum) or np.any(upper > maximum)
    ):
        return np.full_like(lower, minimum), np.full_like(upper, maximum)
    if numeric.rounding is RoundingMode.STOCHASTIC:
        quantized_lower = np.floor(lower * scale) / scale
        quantized_upper = np.ceil(upper * scale) / scale
        return (
            np.clip(quantized_lower, minimum, maximum),
            np.clip(quantized_upper, minimum, maximum),
        )
    q_lower = np.asarray(numeric.quantize(lower))
    q_upper = np.asarray(numeric.quantize(upper))
    return np.minimum(q_lower, q_upper), np.maximum(q_lower, q_upper)


def _integrate_interval(
    voltage_lower: np.ndarray,
    voltage_upper: np.ndarray,
    current_lower: np.ndarray,
    current_upper: np.ndarray,
    tau: np.ndarray,
    dt_bounds: tuple[float, float],
    rule: IntegrationRule,
) -> tuple[np.ndarray, np.ndarray]:
    dt_lower, dt_upper = dt_bounds
    if rule is IntegrationRule.FORWARD_EULER:
        derivative_lower = (-voltage_upper + current_lower) / tau
        derivative_upper = (-voltage_lower + current_upper) / tau
        product_lower, product_upper = _multiply_interval(
            np.full_like(derivative_lower, dt_lower),
            np.full_like(derivative_upper, dt_upper),
            derivative_lower,
            derivative_upper,
        )
        return voltage_lower + product_lower, voltage_upper + product_upper
    alpha_lower = np.exp(-dt_upper / tau)
    alpha_upper = np.exp(-dt_lower / tau)
    retained_lower, retained_upper = _multiply_interval(
        voltage_lower, voltage_upper, alpha_lower, alpha_upper
    )
    drive_factor_lower = 1.0 - alpha_upper
    drive_factor_upper = 1.0 - alpha_lower
    driven_lower, driven_upper = _multiply_interval(
        current_lower,
        current_upper,
        drive_factor_lower,
        drive_factor_upper,
    )
    return retained_lower + driven_lower, retained_upper + driven_upper


def _threshold_and_reset_interval(
    lower: np.ndarray,
    upper: np.ndarray,
    threshold_lower: np.ndarray,
    threshold_upper: np.ndarray,
    reset_value: np.ndarray,
    rule: ResetRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    definitely_quiet = upper < threshold_lower
    definitely_spiking = lower >= threshold_upper
    uncertain = ~(definitely_quiet | definitely_spiking)
    spike_lower = definitely_spiking.astype(np.float64)
    spike_upper = (~definitely_quiet).astype(np.float64)
    if rule is ResetRule.SUBTRACTIVE:
        reset_lower = lower - threshold_upper
        reset_upper = upper - threshold_lower
    else:
        reset_lower = np.broadcast_to(reset_value, lower.shape)
        reset_upper = reset_lower
    state_lower = np.where(definitely_spiking, reset_lower, lower)
    state_upper = np.where(definitely_spiking, reset_upper, upper)
    state_lower = np.where(uncertain, np.minimum(lower, reset_lower), state_lower)
    state_upper = np.where(uncertain, np.maximum(upper, reset_upper), state_upper)
    return state_lower, state_upper, spike_lower, spike_upper


class IntervalFamilyCertifier:
    """Sound interval propagation with discrete-member separation.

    Continuous uncertainty is merged within each discrete semantics member.  The
    argmax condition is checked before taking the conjunction across members, so
    logits from mutually exclusive execution rules are never mixed.
    """

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
    ) -> IntervalCertificateResult:
        return self.certify_union(model, inputs, reference, (box,))

    def certify_partitioned(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        timestep_partitions: int,
        threshold_partitions: int,
    ) -> IntervalCertificateResult:
        boxes = partition_semantics_box(
            box, timestep_partitions, threshold_partitions
        )
        return self.certify_union(model, inputs, reference, boxes)

    def certify_union(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        boxes: tuple[SemanticsBox, ...],
    ) -> IntervalCertificateResult:
        if not boxes:
            raise ValueError("at least one semantics box is required")
        events = _validate_inputs(model, inputs)
        reference_trace = VectorizedEmulator().run(model, events, reference)
        first = boxes[0]
        if any(
            box.base.weight_format != first.base.weight_format
            or box.base.state_format != first.base.state_format
            for box in boxes[1:]
        ):
            raise ValueError("all union boxes must share weight and state formats")
        weight = first.base.weight_format
        numeric = first.base.state_format
        w_in = np.asarray(weight.quantize(model.input_weights))
        bias = np.asarray(numeric.quantize(model.bias))
        input_drive = events @ w_in + bias
        prediction = reference_trace.predictions
        rows = np.arange(events.shape[0])
        certified = np.ones(events.shape[0], dtype=bool)
        lower = np.full((events.shape[0], model.output_size), np.inf)
        upper = np.full((events.shape[0], model.output_size), -np.inf)
        for box in boxes:
            for integration, timing, reset, synaptic_delay, output_delay in itertools.product(
                box.integration_rules,
                box.threshold_timings,
                box.reset_rules,
                box.synaptic_delays,
                box.output_delays,
            ):
                member_lower, member_upper = self._propagate_member(
                    model,
                    events,
                    box,
                    integration,
                    timing,
                    reset,
                    synaptic_delay,
                    output_delay,
                    input_drive,
                )
                chosen_lower = member_lower[rows, prediction]
                competing_upper = member_upper.copy()
                competing_upper[rows, prediction] = -np.inf
                certified &= chosen_lower > np.max(competing_upper, axis=1)
                lower = np.minimum(lower, member_lower)
                upper = np.maximum(upper, member_upper)
        return IntervalCertificateResult(
            certified=certified,
            certified_fraction=float(np.mean(certified)),
            reference_predictions=prediction,
            target_logit_lower=lower,
            target_logit_upper=upper,
        )

    def _propagate_member(
        self,
        model: DenseRecurrentSNN,
        events: np.ndarray,
        box: SemanticsBox,
        integration: IntegrationRule,
        timing: ThresholdTiming,
        reset: ResetRule,
        synaptic_delay: int,
        output_delay: int,
        input_drive: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric = box.base.state_format
        weight = box.base.weight_format
        w_rec = np.asarray(weight.quantize(model.recurrent_weights))
        w_out = np.asarray(weight.quantize(model.output_weights))
        batch, horizon, _ = events.shape
        voltage_lower = np.zeros((batch, model.hidden_size))
        voltage_upper = voltage_lower.copy()
        spike_lower = np.zeros_like(voltage_lower)
        spike_upper = np.zeros_like(voltage_lower)
        logit_lower = np.zeros((batch, model.output_size))
        logit_upper = logit_lower.copy()
        current_queue = [
            (np.zeros_like(voltage_lower), np.zeros_like(voltage_upper))
            for _ in range(synaptic_delay)
        ]
        output_queue = [
            (np.zeros_like(logit_lower), np.zeros_like(logit_upper))
            for _ in range(output_delay)
        ]
        threshold_lower = model.threshold * box.threshold_scale_bounds[0]
        threshold_upper = model.threshold * box.threshold_scale_bounds[1]

        for step in range(horizon):
            recurrent_lower, recurrent_upper = _linear_interval(
                spike_lower, spike_upper, w_rec
            )
            current_lower = input_drive[:, step, :] + recurrent_lower
            current_upper = input_drive[:, step, :] + recurrent_upper
            current_lower, current_upper = _quantize_interval(
                current_lower, current_upper, numeric
            )
            if current_queue:
                current_queue.append((current_lower, current_upper))
                current_lower, current_upper = current_queue.pop(0)

            if timing is ThresholdTiming.PRE_INTEGRATION:
                voltage_lower, voltage_upper, spike_lower, spike_upper = (
                    _threshold_and_reset_interval(
                        voltage_lower,
                        voltage_upper,
                        threshold_lower,
                        threshold_upper,
                        model.reset_value,
                        reset,
                    )
                )
                voltage_lower, voltage_upper = _integrate_interval(
                    voltage_lower,
                    voltage_upper,
                    current_lower,
                    current_upper,
                    model.tau_mem,
                    box.timestep_bounds,
                    integration,
                )
            else:
                voltage_lower, voltage_upper = _integrate_interval(
                    voltage_lower,
                    voltage_upper,
                    current_lower,
                    current_upper,
                    model.tau_mem,
                    box.timestep_bounds,
                    integration,
                )
                voltage_lower, voltage_upper = _quantize_interval(
                    voltage_lower, voltage_upper, numeric
                )
                voltage_lower, voltage_upper, spike_lower, spike_upper = (
                    _threshold_and_reset_interval(
                        voltage_lower,
                        voltage_upper,
                        threshold_lower,
                        threshold_upper,
                        model.reset_value,
                        reset,
                    )
                )
            voltage_lower, voltage_upper = _quantize_interval(
                voltage_lower, voltage_upper, numeric
            )
            contribution_lower, contribution_upper = _linear_interval(
                spike_lower, spike_upper, w_out
            )
            contribution_lower, contribution_upper = _quantize_interval(
                contribution_lower, contribution_upper, numeric
            )
            if output_queue:
                output_queue.append((contribution_lower, contribution_upper))
                contribution_lower, contribution_upper = output_queue.pop(0)
            logit_lower, logit_upper = _quantize_interval(
                logit_lower + contribution_lower,
                logit_upper + contribution_upper,
                numeric,
            )
        return logit_lower, logit_upper


class DecisionMarginFamilyCertifier:
    """Sound decision-level interval propagation for bounded semantics families.

    Unlike :class:`IntervalFamilyCertifier`, this domain accumulates each
    reference-versus-competitor logit difference directly. Shared uncertain
    spikes therefore cannot independently maximize one logit and minimize the
    other. Hidden state and spike reachability remain interval abstractions.

    The current implementation deliberately accepts only floating-point state
    execution. Separate fixed-point logit rounding can destroy a bound expressed
    only in terms of the pre-rounded pairwise difference; supporting it requires
    an additional correlated accumulator state.
    """

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
    ) -> DecisionMarginCertificateResult:
        return self.certify_union(model, inputs, reference, (box,))

    def certify_partitioned(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        timestep_partitions: int,
        threshold_partitions: int,
    ) -> DecisionMarginCertificateResult:
        return self.certify_union(
            model,
            inputs,
            reference,
            partition_semantics_box(
                box, timestep_partitions, threshold_partitions
            ),
        )

    def certify_union(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        boxes: tuple[SemanticsBox, ...],
    ) -> DecisionMarginCertificateResult:
        if not boxes:
            raise ValueError("at least one semantics box is required")
        events = _validate_inputs(model, inputs)
        first = boxes[0]
        if first.base.state_format.is_fixed:
            raise NotImplementedError(
                "decision-margin propagation currently requires floating-point state"
            )
        if any(
            box.base.weight_format != first.base.weight_format
            or box.base.state_format != first.base.state_format
            for box in boxes[1:]
        ):
            raise ValueError("all union boxes must share weight and state formats")
        reference_trace = VectorizedEmulator().run(model, events, reference)
        prediction = reference_trace.predictions
        weight = first.base.weight_format
        numeric = first.base.state_format
        w_in = np.asarray(weight.quantize(model.input_weights))
        bias = np.asarray(numeric.quantize(model.bias))
        input_drive = events @ w_in + bias
        certified = np.ones(events.shape[0], dtype=bool)
        lower = np.full((events.shape[0], model.output_size), np.inf)
        upper = np.full((events.shape[0], model.output_size), -np.inf)
        rows = np.arange(events.shape[0])
        for box in boxes:
            if box.base.state_format.is_fixed:
                raise NotImplementedError(
                    "decision-margin propagation currently requires floating-point state"
                )
            for integration, timing, reset, synaptic_delay, output_delay in itertools.product(
                box.integration_rules,
                box.threshold_timings,
                box.reset_rules,
                box.synaptic_delays,
                box.output_delays,
            ):
                member_lower, member_upper = self._propagate_member_margins(
                    model,
                    events,
                    prediction,
                    box,
                    integration,
                    timing,
                    reset,
                    synaptic_delay,
                    output_delay,
                    input_drive,
                )
                competing_lower = member_lower.copy()
                competing_lower[rows, prediction] = np.inf
                certified &= np.all(competing_lower > 0.0, axis=1)
                lower = np.minimum(lower, member_lower)
                upper = np.maximum(upper, member_upper)
        return DecisionMarginCertificateResult(
            certified=certified,
            certified_fraction=float(np.mean(certified)),
            reference_predictions=prediction,
            target_margin_lower=lower,
            target_margin_upper=upper,
        )

    @staticmethod
    def _margin_contribution_interval(
        spike_lower: np.ndarray,
        spike_upper: np.ndarray,
        positive_differences: np.ndarray,
        negative_differences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.sum(
            spike_lower[:, :, None] * positive_differences
            + spike_upper[:, :, None] * negative_differences,
            axis=1,
        )
        upper = np.sum(
            spike_upper[:, :, None] * positive_differences
            + spike_lower[:, :, None] * negative_differences,
            axis=1,
        )
        return lower, upper

    def _propagate_member_margins(
        self,
        model: DenseRecurrentSNN,
        events: np.ndarray,
        reference_predictions: np.ndarray,
        box: SemanticsBox,
        integration: IntegrationRule,
        timing: ThresholdTiming,
        reset: ResetRule,
        synaptic_delay: int,
        output_delay: int,
        input_drive: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric = box.base.state_format
        weight = box.base.weight_format
        w_rec = np.asarray(weight.quantize(model.recurrent_weights))
        w_out = np.asarray(weight.quantize(model.output_weights))
        batch, horizon, _ = events.shape
        voltage_lower = np.zeros((batch, model.hidden_size))
        voltage_upper = voltage_lower.copy()
        spike_lower = np.zeros_like(voltage_lower)
        spike_upper = np.zeros_like(voltage_lower)
        margin_lower = np.zeros((batch, model.output_size))
        margin_upper = margin_lower.copy()
        current_queue = [
            (np.zeros_like(voltage_lower), np.zeros_like(voltage_upper))
            for _ in range(synaptic_delay)
        ]
        output_queue = [
            (np.zeros_like(margin_lower), np.zeros_like(margin_upper))
            for _ in range(output_delay)
        ]
        threshold_lower = model.threshold * box.threshold_scale_bounds[0]
        threshold_upper = model.threshold * box.threshold_scale_bounds[1]
        chosen_weights = w_out[:, reference_predictions].T
        margin_weights = chosen_weights[:, :, None] - w_out[None, :, :]
        positive_margin_weights = np.maximum(margin_weights, 0.0)
        negative_margin_weights = np.minimum(margin_weights, 0.0)

        for step in range(horizon):
            recurrent_lower, recurrent_upper = _linear_interval(
                spike_lower, spike_upper, w_rec
            )
            current_lower = input_drive[:, step, :] + recurrent_lower
            current_upper = input_drive[:, step, :] + recurrent_upper
            current_lower, current_upper = _quantize_interval(
                current_lower, current_upper, numeric
            )
            if current_queue:
                current_queue.append((current_lower, current_upper))
                current_lower, current_upper = current_queue.pop(0)

            if timing is ThresholdTiming.PRE_INTEGRATION:
                voltage_lower, voltage_upper, spike_lower, spike_upper = (
                    _threshold_and_reset_interval(
                        voltage_lower,
                        voltage_upper,
                        threshold_lower,
                        threshold_upper,
                        model.reset_value,
                        reset,
                    )
                )
                voltage_lower, voltage_upper = _integrate_interval(
                    voltage_lower,
                    voltage_upper,
                    current_lower,
                    current_upper,
                    model.tau_mem,
                    box.timestep_bounds,
                    integration,
                )
            else:
                voltage_lower, voltage_upper = _integrate_interval(
                    voltage_lower,
                    voltage_upper,
                    current_lower,
                    current_upper,
                    model.tau_mem,
                    box.timestep_bounds,
                    integration,
                )
                voltage_lower, voltage_upper = _quantize_interval(
                    voltage_lower, voltage_upper, numeric
                )
                voltage_lower, voltage_upper, spike_lower, spike_upper = (
                    _threshold_and_reset_interval(
                        voltage_lower,
                        voltage_upper,
                        threshold_lower,
                        threshold_upper,
                        model.reset_value,
                        reset,
                    )
                )
            voltage_lower, voltage_upper = _quantize_interval(
                voltage_lower, voltage_upper, numeric
            )
            contribution_lower, contribution_upper = (
                self._margin_contribution_interval(
                    spike_lower,
                    spike_upper,
                    positive_margin_weights,
                    negative_margin_weights,
                )
            )
            if output_queue:
                output_queue.append((contribution_lower, contribution_upper))
                contribution_lower, contribution_upper = output_queue.pop(0)
            margin_lower += contribution_lower
            margin_upper += contribution_upper
        return margin_lower, margin_upper
