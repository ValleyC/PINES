from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .abstract import SemanticsBox
from .emulator import VectorizedEmulator, _validate_inputs
from .models import DenseRecurrentSNN
from .semantics import ExecutionSemantics, IntegrationRule, ResetRule, ThresholdTiming


@dataclass(frozen=True)
class AffineGuardCertificateResult:
    certified: np.ndarray
    certified_fraction: float
    reference_predictions: np.ndarray
    target_margin_lower: np.ndarray
    target_margin_upper: np.ndarray
    split_axis_scores: np.ndarray


@dataclass(frozen=True)
class _HybridAffine:
    """Affine forms over two shared semantic generators plus box residuals."""

    center: np.ndarray
    generators: np.ndarray
    radius: np.ndarray

    def __post_init__(self) -> None:
        if self.generators.shape != (*self.center.shape, 2):
            raise ValueError("hybrid affine forms require two shared generators")
        if self.radius.shape != self.center.shape or np.any(self.radius < 0):
            raise ValueError("hybrid affine radius must be nonnegative and shape-aligned")

    @classmethod
    def exact(cls, values: np.ndarray) -> "_HybridAffine":
        center = np.asarray(values, dtype=np.float64)
        return cls(
            center=center,
            generators=np.zeros((*center.shape, 2), dtype=np.float64),
            radius=np.zeros_like(center),
        )

    @classmethod
    def semantic_interval(
        cls,
        lower: np.ndarray,
        upper: np.ndarray,
        generator_index: int,
    ) -> "_HybridAffine":
        lower_values, upper_values = np.broadcast_arrays(
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
        center = (lower_values + upper_values) / 2.0
        generators = np.zeros((*center.shape, 2), dtype=np.float64)
        generators[..., generator_index] = (upper_values - lower_values) / 2.0
        return cls(center, generators, np.zeros_like(center))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        deviation = np.sum(np.abs(self.generators), axis=-1) + self.radius
        return self.center - deviation, self.center + deviation

    def add(self, other: "_HybridAffine") -> "_HybridAffine":
        return _HybridAffine(
            self.center + other.center,
            self.generators + other.generators,
            self.radius + other.radius,
        )

    def negate(self) -> "_HybridAffine":
        return _HybridAffine(-self.center, -self.generators, self.radius.copy())

    def subtract(self, other: "_HybridAffine") -> "_HybridAffine":
        return self.add(other.negate())

    def scale(self, values: np.ndarray) -> "_HybridAffine":
        scale = np.asarray(values, dtype=np.float64)
        return _HybridAffine(
            self.center * scale,
            self.generators * scale[..., None],
            self.radius * np.abs(scale),
        )

    def linear(self, weights: np.ndarray) -> "_HybridAffine":
        matrix = np.asarray(weights, dtype=np.float64)
        return _HybridAffine(
            self.center @ matrix,
            np.einsum("bfi,fo->boi", self.generators, matrix),
            self.radius @ np.abs(matrix),
        )

    def batch_linear(self, weights: np.ndarray) -> "_HybridAffine":
        matrix = np.asarray(weights, dtype=np.float64)
        return _HybridAffine(
            np.einsum("bf,bfo->bo", self.center, matrix),
            np.einsum("bfi,bfo->boi", self.generators, matrix),
            np.einsum("bf,bfo->bo", self.radius, np.abs(matrix)),
        )

    def product(self, other: "_HybridAffine") -> "_HybridAffine":
        center = self.center * other.center
        generators = (
            self.center[..., None] * other.generators
            + other.center[..., None] * self.generators
        )
        left_shared_radius = np.sum(np.abs(self.generators), axis=-1)
        right_shared_radius = np.sum(np.abs(other.generators), axis=-1)
        radius = (
            left_shared_radius * right_shared_radius
            + np.abs(self.center) * other.radius
            + np.abs(other.center) * self.radius
            + left_shared_radius * other.radius
            + right_shared_radius * self.radius
            + self.radius * other.radius
        )
        return _HybridAffine(center, generators, radius)


def _spike_relaxation(guard: _HybridAffine) -> _HybridAffine:
    lower, upper = guard.bounds()
    definitely_quiet = upper < 0.0
    definitely_spiking = lower >= 0.0
    uncertain = ~(definitely_quiet | definitely_spiking)
    center = definitely_spiking.astype(np.float64) + 0.5 * uncertain
    return _HybridAffine(
        center=center,
        generators=np.zeros((*center.shape, 2), dtype=np.float64),
        radius=0.5 * uncertain.astype(np.float64),
    )


class AffineGuardFamilyCertifier:
    """Sound shared-semantics affine propagation through recurrent SNN guards.

    The domain retains affine dependence on timestep and threshold scale. Each
    nonlinear product is enclosed with an independent residual, and each
    threshold-uncertain spike is relaxed to ``[0,1]``. It currently supports
    deterministic forward-Euler execution with floating-point states.
    """

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
    ) -> AffineGuardCertificateResult:
        events = _validate_inputs(model, inputs)
        prediction = VectorizedEmulator().run(model, events, reference).predictions
        return self._certify_with_predictions(
            model, events, prediction, box
        )

    def certify_with_reference_predictions(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        reference_predictions: np.ndarray,
    ) -> AffineGuardCertificateResult:
        del reference
        events = _validate_inputs(model, inputs)
        prediction = np.asarray(reference_predictions, dtype=np.int64)
        if prediction.shape != (len(events),):
            raise ValueError("reference predictions must have one entry per input")
        if np.any(prediction < 0) or np.any(prediction >= model.output_size):
            raise ValueError("reference prediction is outside the output range")
        return self._certify_with_predictions(model, events, prediction, box)

    def _certify_with_predictions(
        self,
        model: DenseRecurrentSNN,
        events: np.ndarray,
        prediction: np.ndarray,
        box: SemanticsBox,
    ) -> AffineGuardCertificateResult:
        if box.base.state_format.is_fixed:
            raise NotImplementedError("affine guards require floating-point state")
        if not box.base.randomness.deterministic:
            raise NotImplementedError("affine guards require deterministic execution")
        certified = np.ones(len(events), dtype=bool)
        lower = np.full((len(events), model.output_size), np.inf)
        upper = np.full((len(events), model.output_size), -np.inf)
        split_axis_scores = np.zeros((len(events), 2), dtype=np.float64)
        rows = np.arange(len(events))
        for integration, timing, reset, synaptic_delay, output_delay in itertools.product(
            box.integration_rules,
            box.threshold_timings,
            box.reset_rules,
            box.synaptic_delays,
            box.output_delays,
        ):
            if integration is not IntegrationRule.FORWARD_EULER:
                raise NotImplementedError(
                    "affine guards currently support forward Euler only"
                )
            member_lower, member_upper, member_scores = self._propagate_member(
                model,
                events,
                prediction,
                box,
                timing,
                reset,
                synaptic_delay,
                output_delay,
            )
            competing = member_lower.copy()
            competing[rows, prediction] = np.inf
            certified &= np.all(competing > 0.0, axis=1)
            lower = np.minimum(lower, member_lower)
            upper = np.maximum(upper, member_upper)
            split_axis_scores += member_scores
        return AffineGuardCertificateResult(
            certified=certified,
            certified_fraction=float(np.mean(certified)),
            reference_predictions=prediction,
            target_margin_lower=lower,
            target_margin_upper=upper,
            split_axis_scores=split_axis_scores,
        )

    @staticmethod
    def _reset(
        voltage: _HybridAffine,
        spikes: _HybridAffine,
        threshold: _HybridAffine,
        reset_value: np.ndarray,
        rule: ResetRule,
    ) -> _HybridAffine:
        if rule is ResetRule.SUBTRACTIVE:
            return voltage.subtract(spikes.product(threshold))
        one_minus_spikes = _HybridAffine.exact(
            np.ones_like(spikes.center)
        ).subtract(spikes)
        reset = _HybridAffine.exact(
            np.broadcast_to(reset_value, voltage.center.shape)
        )
        return one_minus_spikes.product(voltage).add(spikes.product(reset))

    @staticmethod
    def _integrate(
        voltage: _HybridAffine,
        current: _HybridAffine,
        timestep: _HybridAffine,
        tau: np.ndarray,
    ) -> _HybridAffine:
        derivative = current.subtract(voltage).scale(1.0 / tau)
        return voltage.add(timestep.product(derivative))

    def _propagate_member(
        self,
        model: DenseRecurrentSNN,
        events: np.ndarray,
        reference_predictions: np.ndarray,
        box: SemanticsBox,
        timing: ThresholdTiming,
        reset: ResetRule,
        synaptic_delay: int,
        output_delay: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weight = box.base.weight_format
        w_in = np.asarray(weight.quantize(model.input_weights), dtype=np.float64)
        w_rec = np.asarray(weight.quantize(model.recurrent_weights), dtype=np.float64)
        w_out = np.asarray(weight.quantize(model.output_weights), dtype=np.float64)
        batch, horizon, _ = events.shape
        base_drive = events @ w_in + model.bias
        voltage = _HybridAffine.exact(np.zeros((batch, model.hidden_size)))
        spikes = _HybridAffine.exact(np.zeros_like(voltage.center))
        margins = _HybridAffine.exact(np.zeros((batch, model.output_size)))
        threshold = _HybridAffine.semantic_interval(
            np.broadcast_to(
                model.threshold * box.threshold_scale_bounds[0],
                voltage.center.shape,
            ),
            np.broadcast_to(
                model.threshold * box.threshold_scale_bounds[1],
                voltage.center.shape,
            ),
            1,
        )
        timestep = _HybridAffine.semantic_interval(
            np.full_like(voltage.center, box.timestep_bounds[0]),
            np.full_like(voltage.center, box.timestep_bounds[1]),
            0,
        )
        current_queue = [
            _HybridAffine.exact(np.zeros_like(voltage.center))
            for _ in range(synaptic_delay)
        ]
        output_queue = [
            _HybridAffine.exact(np.zeros_like(margins.center))
            for _ in range(output_delay)
        ]
        chosen_weights = w_out[:, reference_predictions].T
        margin_weights = chosen_weights[:, :, None] - w_out[None, :, :]
        split_axis_scores = np.zeros((batch, 2), dtype=np.float64)

        def relax_guard(guard: _HybridAffine) -> _HybridAffine:
            guard_lower, guard_upper = guard.bounds()
            uncertain = ~((guard_upper < 0.0) | (guard_lower >= 0.0))
            split_axis_scores[:] += np.sum(
                np.abs(guard.generators) * uncertain[..., None], axis=1
            )
            return _spike_relaxation(guard)

        for step in range(horizon):
            current = _HybridAffine.exact(base_drive[:, step, :]).add(
                spikes.linear(w_rec)
            )
            if current_queue:
                current_queue.append(current)
                current = current_queue.pop(0)
            if timing is ThresholdTiming.PRE_INTEGRATION:
                spikes = relax_guard(voltage.subtract(threshold))
                voltage = self._integrate(
                    self._reset(
                        voltage,
                        spikes,
                        threshold,
                        model.reset_value,
                        reset,
                    ),
                    current,
                    timestep,
                    model.tau_mem,
                )
            else:
                voltage = self._integrate(
                    voltage,
                    current,
                    timestep,
                    model.tau_mem,
                )
                spikes = relax_guard(voltage.subtract(threshold))
                voltage = self._reset(
                    voltage,
                    spikes,
                    threshold,
                    model.reset_value,
                    reset,
                )
            contribution = spikes.batch_linear(margin_weights)
            if output_queue:
                output_queue.append(contribution)
                contribution = output_queue.pop(0)
            margins = margins.add(contribution)
        margin_lower, margin_upper = margins.bounds()
        return margin_lower, margin_upper, split_axis_scores
