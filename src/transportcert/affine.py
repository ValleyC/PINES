from __future__ import annotations

import itertools
import heapq
from dataclasses import dataclass, replace

import numpy as np

from .abstract import SemanticsBox
from .emulator import VectorizedEmulator, _validate_inputs
from .models import DenseRecurrentSNN
from .semantics import (
    ExecutionSemantics,
    IntegrationRule,
    NumericFormat,
    ResetRule,
    ThresholdTiming,
)


_ANALYSIS_EPSILON = np.finfo(np.float64).eps
_ANALYSIS_MINIMUM = float(np.nextafter(0.0, 1.0))


def _analysis_gamma(operation_count: int) -> float:
    product = operation_count * _ANALYSIS_EPSILON
    if operation_count < 1 or product >= 1.0:
        raise ValueError("invalid binary64 error operation count")
    return product / (1.0 - product)


def _outward_radius(radius: np.ndarray, error: np.ndarray) -> np.ndarray:
    inflated = np.asarray(radius, dtype=np.float64) + np.asarray(
        error, dtype=np.float64
    )
    return np.nextafter(inflated + _ANALYSIS_MINIMUM, np.inf)


@dataclass(frozen=True)
class AffineGuardCertificateResult:
    certified: np.ndarray
    certified_fraction: float
    reference_predictions: np.ndarray
    target_margin_lower: np.ndarray
    target_margin_upper: np.ndarray
    split_axis_scores: np.ndarray
    guard_cut_valid: np.ndarray
    guard_cut_center: np.ndarray
    guard_cut_generators: np.ndarray
    guard_cut_radius: np.ndarray


@dataclass(frozen=True)
class AdaptiveGuardCutCertificateResult:
    certified: bool
    certified_parameter_fraction: float
    unresolved_parameter_fraction: float
    analyzed_polygons: int
    final_leaves: int
    certified_leaves: int
    unresolved_leaves: int
    maximum_depth: int
    maximum_polygon_vertices: int
    guard_band_splits: int
    axis_fallback_splits: int
    unresolved_polygons: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class FixedTraceAffineResult:
    trace_robust: bool
    certified: bool
    reference_prediction: int
    trace_prediction: int
    total_guard_count: int
    uncertain_guard_count: int
    uncertain_timestep_count: int
    first_uncertain_timestep: int | None
    minimum_signed_guard_margin: float


@dataclass(frozen=True)
class PolygonBranchCertificateResult:
    certified: bool
    complete: bool
    reference_prediction: int
    possible_predictions: tuple[int, ...]
    final_branch_count: int
    maximum_active_branches: int
    total_branch_splits: int
    maximum_uncertain_neurons_at_step: int
    first_cap_timestep: int | None


@dataclass(frozen=True)
class AdaptiveHybridCertificateResult:
    certified: bool
    certified_parameter_fraction: float
    unresolved_parameter_fraction: float
    analyzed_polygons: int
    final_leaves: int
    branch_certified_leaves: int
    affine_certified_leaves: int
    unresolved_leaves: int
    branch_attempts: int
    branch_cap_hits: int
    branch_prediction_rejections: int
    maximum_completed_branches: int
    guard_band_splits: int
    axis_fallback_splits: int


@dataclass(frozen=True)
class _PolygonBranch:
    voltage: _HybridAffine
    previous_spikes: np.ndarray
    logits: np.ndarray
    current_queue: tuple[_HybridAffine, ...]
    output_queue: tuple[np.ndarray, ...]


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
        construction_scale = np.abs(lower_values) + np.abs(upper_values)
        radius = _outward_radius(
            np.zeros_like(center),
            _analysis_gamma(8) * construction_scale,
        )
        return cls(center, generators, radius)

    def bounds(
        self, parameter_vertices: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if parameter_vertices is None:
            shared_radius = np.sum(np.abs(self.generators), axis=-1)
            magnitude = np.abs(self.center) + shared_radius + self.radius
            evaluation_error = _analysis_gamma(8) * magnitude
            return (
                np.nextafter(
                    self.center
                    - shared_radius
                    - self.radius
                    - evaluation_error,
                    -np.inf,
                ),
                np.nextafter(
                    self.center
                    + shared_radius
                    + self.radius
                    + evaluation_error,
                    np.inf,
                ),
            )
        vertices = np.asarray(parameter_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 1:
            raise ValueError("parameter vertices must have shape [vertices, 2]")
        shared_values = np.einsum("bfi,vi->bfv", self.generators, vertices)
        shared_magnitude = np.einsum(
            "bfi,vi->bfv", np.abs(self.generators), np.abs(vertices)
        )
        evaluation_error = _analysis_gamma(8) * (
            np.abs(self.center)[..., None]
            + shared_magnitude
            + self.radius[..., None]
        )
        return (
            np.nextafter(
                np.min(
                    self.center[..., None]
                    + shared_values
                    - self.radius[..., None]
                    - evaluation_error,
                    axis=-1,
                ),
                -np.inf,
            ),
            np.nextafter(
                np.max(
                    self.center[..., None]
                    + shared_values
                    + self.radius[..., None]
                    + evaluation_error,
                    axis=-1,
                ),
                np.inf,
            ),
        )

    def add(self, other: "_HybridAffine") -> "_HybridAffine":
        center = self.center + other.center
        generators = self.generators + other.generators
        radius = self.radius + other.radius
        operation_scale = (
            np.abs(self.center)
            + np.abs(other.center)
            + np.sum(
                np.abs(self.generators) + np.abs(other.generators), axis=-1
            )
            + self.radius
            + other.radius
        )
        return _HybridAffine(
            center,
            generators,
            _outward_radius(
                radius, _analysis_gamma(8) * operation_scale
            ),
        )

    def negate(self) -> "_HybridAffine":
        return _HybridAffine(-self.center, -self.generators, self.radius.copy())

    def subtract(self, other: "_HybridAffine") -> "_HybridAffine":
        return self.add(other.negate())

    def scale(self, values: np.ndarray) -> "_HybridAffine":
        scale = np.asarray(values, dtype=np.float64)
        center = self.center * scale
        generators = self.generators * scale[..., None]
        radius = self.radius * np.abs(scale)
        operation_scale = (
            np.abs(center) + np.sum(np.abs(generators), axis=-1) + radius
        )
        return _HybridAffine(
            center,
            generators,
            _outward_radius(
                radius, _analysis_gamma(8) * operation_scale
            ),
        )

    def linear(self, weights: np.ndarray) -> "_HybridAffine":
        matrix = np.asarray(weights, dtype=np.float64)
        center = self.center @ matrix
        generators = np.einsum("bfi,fo->boi", self.generators, matrix)
        radius = self.radius @ np.abs(matrix)
        absolute_center = np.abs(self.center) @ np.abs(matrix)
        absolute_generators = np.einsum(
            "bfi,fo->boi", np.abs(self.generators), np.abs(matrix)
        )
        operation_scale = (
            absolute_center
            + np.sum(absolute_generators, axis=-1)
            + radius
        )
        return _HybridAffine(
            center,
            generators,
            _outward_radius(
                radius,
                _analysis_gamma(2 * matrix.shape[0] + 8) * operation_scale,
            ),
        )

    def batch_linear(self, weights: np.ndarray) -> "_HybridAffine":
        matrix = np.asarray(weights, dtype=np.float64)
        center = np.einsum("bf,bfo->bo", self.center, matrix)
        generators = np.einsum("bfi,bfo->boi", self.generators, matrix)
        radius = np.einsum("bf,bfo->bo", self.radius, np.abs(matrix))
        absolute_center = np.einsum(
            "bf,bfo->bo", np.abs(self.center), np.abs(matrix)
        )
        absolute_generators = np.einsum(
            "bfi,bfo->boi", np.abs(self.generators), np.abs(matrix)
        )
        operation_scale = (
            absolute_center
            + np.sum(absolute_generators, axis=-1)
            + radius
        )
        return _HybridAffine(
            center,
            generators,
            _outward_radius(
                radius,
                _analysis_gamma(2 * matrix.shape[1] + 8) * operation_scale,
            ),
        )

    def product(
        self,
        other: "_HybridAffine",
        parameter_vertices: np.ndarray | None = None,
    ) -> "_HybridAffine":
        if parameter_vertices is None:
            center = self.center * other.center
            generators = (
                self.center[..., None] * other.generators
                + other.center[..., None] * self.generators
            )
            left_expansion_center = self.center
            right_expansion_center = other.center
            left_shared_radius = np.sum(np.abs(self.generators), axis=-1)
            right_shared_radius = np.sum(np.abs(other.generators), axis=-1)
        else:
            vertices = np.asarray(parameter_vertices, dtype=np.float64)
            domain_center = np.mean(vertices, axis=0)
            centered_vertices = vertices - domain_center[None, :]
            left_expansion_center = self.center + np.einsum(
                "bfi,i->bf", self.generators, domain_center
            )
            right_expansion_center = other.center + np.einsum(
                "bfi,i->bf", other.generators, domain_center
            )
            generators = (
                left_expansion_center[..., None] * other.generators
                + right_expansion_center[..., None] * self.generators
            )
            center = (
                left_expansion_center * right_expansion_center
                - np.einsum("bfi,i->bf", generators, domain_center)
            )
            left_shared_radius = np.max(
                np.abs(
                    np.einsum(
                        "bfi,vi->bfv", self.generators, centered_vertices
                    )
                ),
                axis=-1,
            )
            right_shared_radius = np.max(
                np.abs(
                    np.einsum(
                        "bfi,vi->bfv", other.generators, centered_vertices
                    )
                ),
                axis=-1,
            )
        radius = (
            left_shared_radius * right_shared_radius
            + np.abs(left_expansion_center) * other.radius
            + np.abs(right_expansion_center) * self.radius
            + left_shared_radius * other.radius
            + right_shared_radius * self.radius
            + self.radius * other.radius
        )
        product_scale = (
            np.abs(left_expansion_center)
            + left_shared_radius
            + self.radius
        ) * (
            np.abs(right_expansion_center)
            + right_shared_radius
            + other.radius
        )
        return _HybridAffine(
            center,
            generators,
            _outward_radius(
                radius, _analysis_gamma(256) * product_scale
            ),
        )

    def float_quantize(
        self,
        numeric_format: NumericFormat,
        parameter_vertices: np.ndarray | None = None,
    ) -> "_HybridAffine":
        """Add a conservative envelope for a declared floating conversion."""

        if numeric_format.is_fixed:
            raise NotImplementedError("hybrid affine fixed-point transitions")
        lower, upper = self.bounds(parameter_vertices)
        rounding_radius = _floating_rounding_radius(
            numeric_format, lower, upper
        )
        radius = self.radius + rounding_radius
        return _HybridAffine(
            self.center,
            self.generators,
            _outward_radius(
                radius,
                _analysis_gamma(8) * (self.radius + rounding_radius),
            ),
        )

    def exp_negative(
        self, parameter_vertices: np.ndarray | None = None
    ) -> "_HybridAffine":
        """Sound first-order enclosure of ``exp(-self)`` on the domain."""

        if parameter_vertices is None:
            domain_center = np.zeros(2, dtype=np.float64)
            shared_deviation = np.sum(np.abs(self.generators), axis=-1)
        else:
            vertices = np.asarray(parameter_vertices, dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 1:
                raise ValueError("parameter vertices must have shape [vertices, 2]")
            domain_center = np.mean(vertices, axis=0)
            centered_vertices = vertices - domain_center[None, :]
            shared_deviation = np.max(
                np.abs(
                    np.einsum(
                        "bfi,vi->bfv", self.generators, centered_vertices
                    )
                ),
                axis=-1,
            )
        expansion_center = self.center + np.einsum(
            "bfi,i->bf", self.generators, domain_center
        )
        value = np.exp(-expansion_center)
        derivative = -value
        generators = derivative[..., None] * self.generators
        center = value - np.einsum("bfi,i->bf", generators, domain_center)
        total_deviation = shared_deviation + self.radius
        maximum_second_derivative = np.exp(
            -expansion_center + total_deviation
        )
        radius = (
            np.abs(derivative) * self.radius
            + 0.5 * maximum_second_derivative * total_deviation**2
        )
        exponential_scale = np.exp(
            -expansion_center + total_deviation
        )
        return _HybridAffine(
            center,
            generators,
            _outward_radius(
                radius, _analysis_gamma(256) * exponential_scale
            ),
        )


def _floating_rounding_radius(
    numeric_format: NumericFormat,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if numeric_format.is_fixed:
        raise NotImplementedError("floating rounding radius requires float format")
    dtype = np.float32 if numeric_format.kind == "float32" else np.float64
    maximum_magnitude = np.maximum(np.abs(lower), np.abs(upper))
    if np.any(maximum_magnitude > np.finfo(dtype).max):
        raise OverflowError("floating affine enclosure exceeds numeric range")
    minimum_subnormal = float(np.nextafter(dtype(0.0), dtype(1.0)))
    radius = np.finfo(dtype).eps * maximum_magnitude + minimum_subnormal
    return np.nextafter(
        radius + _analysis_gamma(8) * radius + _ANALYSIS_MINIMUM,
        np.inf,
    )


def _spike_relaxation(
    guard: _HybridAffine,
    parameter_vertices: np.ndarray | None = None,
) -> _HybridAffine:
    lower, upper = guard.bounds(parameter_vertices)
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
    deterministic forward or exponential Euler with floating-point states.
    """

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        parameter_vertices: np.ndarray | None = None,
    ) -> AffineGuardCertificateResult:
        events = _validate_inputs(model, inputs)
        prediction = VectorizedEmulator().run(model, events, reference).predictions
        return self._certify_with_predictions(
            model, events, prediction, box, parameter_vertices
        )

    def certify_with_reference_predictions(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        reference_predictions: np.ndarray,
        parameter_vertices: np.ndarray | None = None,
    ) -> AffineGuardCertificateResult:
        del reference
        events = _validate_inputs(model, inputs)
        prediction = np.asarray(reference_predictions, dtype=np.int64)
        if prediction.shape != (len(events),):
            raise ValueError("reference predictions must have one entry per input")
        if np.any(prediction < 0) or np.any(prediction >= model.output_size):
            raise ValueError("reference prediction is outside the output range")
        return self._certify_with_predictions(
            model, events, prediction, box, parameter_vertices
        )

    def _certify_with_predictions(
        self,
        model: DenseRecurrentSNN,
        events: np.ndarray,
        prediction: np.ndarray,
        box: SemanticsBox,
        parameter_vertices: np.ndarray | None,
    ) -> AffineGuardCertificateResult:
        if box.base.state_format.is_fixed:
            raise NotImplementedError("affine guards require floating-point state")
        if not box.base.randomness.deterministic:
            raise NotImplementedError("affine guards require deterministic execution")
        certified = np.ones(len(events), dtype=bool)
        lower = np.full((len(events), model.output_size), np.inf)
        upper = np.full((len(events), model.output_size), -np.inf)
        split_axis_scores = np.zeros((len(events), 2), dtype=np.float64)
        guard_cut_valid = np.zeros(len(events), dtype=bool)
        guard_cut_center = np.full(len(events), np.nan, dtype=np.float64)
        guard_cut_generators = np.full((len(events), 2), np.nan, dtype=np.float64)
        guard_cut_radius = np.full(len(events), np.nan, dtype=np.float64)
        guard_cut_strength = np.full(len(events), -np.inf, dtype=np.float64)
        rows = np.arange(len(events))
        for integration, timing, reset, synaptic_delay, output_delay in itertools.product(
            box.integration_rules,
            box.threshold_timings,
            box.reset_rules,
            box.synaptic_delays,
            box.output_delays,
        ):
            (
                member_lower,
                member_upper,
                member_scores,
                member_cut_valid,
                member_cut_center,
                member_cut_generators,
                member_cut_radius,
                member_cut_strength,
            ) = self._propagate_member(
                model,
                events,
                prediction,
                box,
                integration,
                timing,
                reset,
                synaptic_delay,
                output_delay,
                parameter_vertices,
            )
            competing = member_lower.copy()
            competing[rows, prediction] = np.inf
            certified &= np.all(competing > 0.0, axis=1)
            lower = np.minimum(lower, member_lower)
            upper = np.maximum(upper, member_upper)
            split_axis_scores += member_scores
            replace_cut = member_cut_valid & (
                member_cut_strength > guard_cut_strength
            )
            guard_cut_valid[replace_cut] = True
            guard_cut_center[replace_cut] = member_cut_center[replace_cut]
            guard_cut_generators[replace_cut] = member_cut_generators[replace_cut]
            guard_cut_radius[replace_cut] = member_cut_radius[replace_cut]
            guard_cut_strength[replace_cut] = member_cut_strength[replace_cut]
        return AffineGuardCertificateResult(
            certified=certified,
            certified_fraction=float(np.mean(certified)),
            reference_predictions=prediction,
            target_margin_lower=lower,
            target_margin_upper=upper,
            split_axis_scores=split_axis_scores,
            guard_cut_valid=guard_cut_valid,
            guard_cut_center=guard_cut_center,
            guard_cut_generators=guard_cut_generators,
            guard_cut_radius=guard_cut_radius,
        )

    @staticmethod
    def _reset(
        voltage: _HybridAffine,
        spikes: _HybridAffine,
        threshold: _HybridAffine,
        reset_value: np.ndarray,
        rule: ResetRule,
        parameter_vertices: np.ndarray | None,
    ) -> _HybridAffine:
        if rule is ResetRule.SUBTRACTIVE:
            return voltage.subtract(
                spikes.product(threshold, parameter_vertices)
            )
        one_minus_spikes = _HybridAffine.exact(
            np.ones_like(spikes.center)
        ).subtract(spikes)
        reset = _HybridAffine.exact(
            np.broadcast_to(reset_value, voltage.center.shape)
        )
        return one_minus_spikes.product(voltage, parameter_vertices).add(
            spikes.product(reset, parameter_vertices)
        )

    @staticmethod
    def _integrate(
        voltage: _HybridAffine,
        current: _HybridAffine,
        timestep: _HybridAffine,
        tau: np.ndarray,
        parameter_vertices: np.ndarray | None,
        integration: IntegrationRule = IntegrationRule.FORWARD_EULER,
    ) -> _HybridAffine:
        if integration is IntegrationRule.EXPONENTIAL_EULER:
            alpha = timestep.scale(1.0 / tau).exp_negative(parameter_vertices)
            return current.add(
                alpha.product(voltage.subtract(current), parameter_vertices)
            )
        derivative = current.subtract(voltage).scale(1.0 / tau)
        return voltage.add(timestep.product(derivative, parameter_vertices))

    def _propagate_member(
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
        parameter_vertices: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        weight = box.base.weight_format
        state = box.base.state_format
        w_in = np.asarray(weight.quantize(model.input_weights), dtype=np.float64)
        w_rec = np.asarray(weight.quantize(model.recurrent_weights), dtype=np.float64)
        w_out = np.asarray(weight.quantize(model.output_weights), dtype=np.float64)
        batch, horizon, _ = events.shape
        bias = np.asarray(state.quantize(model.bias), dtype=np.float64)
        base_drive = events @ w_in + bias
        voltage = _HybridAffine.exact(np.zeros((batch, model.hidden_size)))
        spikes = _HybridAffine.exact(np.zeros_like(voltage.center))
        margins = _HybridAffine.exact(np.zeros((batch, model.output_size)))
        class_logits = _HybridAffine.exact(
            np.zeros((batch, model.output_size))
        )
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
        margin_output_queue = [
            _HybridAffine.exact(np.zeros_like(margins.center))
            for _ in range(output_delay)
        ]
        class_output_queue = [
            _HybridAffine.exact(np.zeros_like(class_logits.center))
            for _ in range(output_delay)
        ]
        chosen_weights = w_out[:, reference_predictions].T
        margin_weights = chosen_weights[:, :, None] - w_out[None, :, :]
        split_axis_scores = np.zeros((batch, 2), dtype=np.float64)
        cut_valid = np.zeros(batch, dtype=bool)
        cut_center = np.full(batch, np.nan, dtype=np.float64)
        cut_generators = np.full((batch, 2), np.nan, dtype=np.float64)
        cut_radius = np.full(batch, np.nan, dtype=np.float64)
        cut_strength = np.full(batch, -np.inf, dtype=np.float64)
        vertices = (
            np.asarray(parameter_vertices, dtype=np.float64)
            if parameter_vertices is not None
            else np.asarray(
                [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                dtype=np.float64,
            )
        )

        def relax_guard(guard: _HybridAffine) -> _HybridAffine:
            guard_lower, guard_upper = guard.bounds(vertices)
            uncertain = ~((guard_upper < 0.0) | (guard_lower >= 0.0))
            split_axis_scores[:] += np.sum(
                np.abs(guard.generators) * uncertain[..., None], axis=1
            )
            shared_values = guard.center[..., None] + np.einsum(
                "bfi,vi->bfv", guard.generators, vertices
            )
            shared_lower = np.min(shared_values, axis=-1)
            shared_upper = np.max(shared_values, axis=-1)
            outside_band = np.maximum(
                shared_upper - guard.radius,
                -guard.radius - shared_lower,
            )
            outside_band = np.where(uncertain, outside_band, -np.inf)
            for input_index in range(batch):
                neuron = int(np.argmax(outside_band[input_index]))
                strength = float(outside_band[input_index, neuron])
                if strength > cut_strength[input_index] and strength > 0.0:
                    cut_valid[input_index] = True
                    cut_center[input_index] = guard.center[input_index, neuron]
                    cut_generators[input_index] = guard.generators[
                        input_index, neuron
                    ]
                    cut_radius[input_index] = guard.radius[input_index, neuron]
                    cut_strength[input_index] = strength
            return _spike_relaxation(guard, vertices)

        for step in range(horizon):
            current = _HybridAffine.exact(base_drive[:, step, :]).add(
                spikes.linear(w_rec)
            ).float_quantize(state, vertices)
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
                        vertices,
                    ),
                    current,
                    timestep,
                    model.tau_mem,
                    vertices,
                    integration,
                ).float_quantize(state, vertices)
            else:
                voltage = self._integrate(
                    voltage,
                    current,
                    timestep,
                    model.tau_mem,
                    vertices,
                    integration,
                ).float_quantize(state, vertices)
                spikes = relax_guard(voltage.subtract(threshold))
                voltage = self._reset(
                    voltage,
                    spikes,
                    threshold,
                    model.reset_value,
                    reset,
                    vertices,
                ).float_quantize(state, vertices)
            raw_class_contribution = spikes.linear(w_out)
            class_lower, class_upper = raw_class_contribution.bounds(vertices)
            class_contribution_error = _floating_rounding_radius(
                state, class_lower, class_upper
            )
            class_contribution = raw_class_contribution.float_quantize(
                state, vertices
            )
            margin_contribution = spikes.batch_linear(margin_weights)
            chosen_error = class_contribution_error[
                np.arange(batch), reference_predictions, None
            ]
            margin_contribution = _HybridAffine(
                margin_contribution.center,
                margin_contribution.generators,
                _outward_radius(
                    margin_contribution.radius
                    + chosen_error
                    + class_contribution_error,
                    _analysis_gamma(8)
                    * (
                        margin_contribution.radius
                        + chosen_error
                        + class_contribution_error
                    ),
                ),
            )
            if margin_output_queue:
                margin_output_queue.append(margin_contribution)
                margin_contribution = margin_output_queue.pop(0)
                class_output_queue.append(class_contribution)
                class_contribution = class_output_queue.pop(0)
            raw_class_logits = class_logits.add(class_contribution)
            logit_lower, logit_upper = raw_class_logits.bounds(vertices)
            logit_error = _floating_rounding_radius(
                state, logit_lower, logit_upper
            )
            class_logits = raw_class_logits.float_quantize(state, vertices)
            chosen_logit_error = logit_error[
                np.arange(batch), reference_predictions, None
            ]
            raw_margins = margins.add(margin_contribution)
            margins = _HybridAffine(
                raw_margins.center,
                raw_margins.generators,
                _outward_radius(
                    raw_margins.radius + chosen_logit_error + logit_error,
                    _analysis_gamma(8)
                    * (
                        raw_margins.radius
                        + chosen_logit_error
                        + logit_error
                    ),
                ),
            )
        margin_lower, margin_upper = margins.bounds(vertices)
        return (
            margin_lower,
            margin_upper,
            split_axis_scores,
            cut_valid,
            cut_center,
            cut_generators,
            cut_radius,
            cut_strength,
        )


class FixedTraceAffineAnalyzer:
    """Prove or diagnose one center trace over a convex semantics polygon."""

    def analyze(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        parameter_vertices: np.ndarray,
    ) -> FixedTraceAffineResult:
        events = _validate_inputs(model, inputs)
        if len(events) != 1:
            raise ValueError("fixed-trace analysis currently accepts one input")
        discrete_sizes = (
            len(box.integration_rules),
            len(box.threshold_timings),
            len(box.reset_rules),
            len(box.synaptic_delays),
            len(box.output_delays),
        )
        if discrete_sizes != (1, 1, 1, 1, 1):
            raise ValueError("fixed-trace analysis requires one discrete member")
        if box.base.state_format.is_fixed:
            raise NotImplementedError("fixed-trace affine analysis requires float state")
        vertices = np.asarray(parameter_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("parameter polygon must have shape [vertices, 2]")
        centroid = np.mean(vertices, axis=0)
        timestep_center = sum(box.timestep_bounds) / 2.0
        timestep_radius = (box.timestep_bounds[1] - box.timestep_bounds[0]) / 2.0
        threshold_center = sum(box.threshold_scale_bounds) / 2.0
        threshold_radius = (
            box.threshold_scale_bounds[1] - box.threshold_scale_bounds[0]
        ) / 2.0
        center_timestep = timestep_center + timestep_radius * centroid[0]
        center_threshold_scale = threshold_center + threshold_radius * centroid[1]
        center_semantics = replace(box.base, timestep=float(center_timestep))
        center_model = model.with_parameters(
            threshold=model.threshold * float(center_threshold_scale)
        )
        center_trace = VectorizedEmulator().run(
            center_model, events, center_semantics
        )
        forced_spikes = center_trace.spikes[0]
        trace_prediction = int(center_trace.predictions[0])
        reference_prediction = int(
            VectorizedEmulator().run(model, events, reference).predictions[0]
        )

        state = box.base.state_format
        weight = box.base.weight_format
        w_in = np.asarray(weight.quantize(model.input_weights), dtype=np.float64)
        w_rec = np.asarray(weight.quantize(model.recurrent_weights), dtype=np.float64)
        bias = np.asarray(state.quantize(model.bias), dtype=np.float64)
        input_drive = events @ w_in
        voltage = _HybridAffine.exact(np.zeros((1, model.hidden_size)))
        threshold = _HybridAffine.semantic_interval(
            model.threshold[None, :] * box.threshold_scale_bounds[0],
            model.threshold[None, :] * box.threshold_scale_bounds[1],
            1,
        )
        timestep = _HybridAffine.semantic_interval(
            np.full((1, model.hidden_size), box.timestep_bounds[0]),
            np.full((1, model.hidden_size), box.timestep_bounds[1]),
            0,
        )
        synaptic_delay = box.synaptic_delays[0]
        current_queue = [
            _HybridAffine.exact(np.zeros_like(voltage.center))
            for _ in range(synaptic_delay)
        ]
        previous_spikes = np.zeros((1, model.hidden_size), dtype=np.float64)
        uncertain_guard_count = 0
        uncertain_timesteps: set[int] = set()
        first_uncertain: int | None = None
        minimum_signed_margin = np.inf
        timing = box.threshold_timings[0]
        reset = box.reset_rules[0]

        def inspect_guard(step: int, guard: _HybridAffine, spikes: np.ndarray) -> None:
            nonlocal uncertain_guard_count, first_uncertain, minimum_signed_margin
            lower, upper = guard.bounds(vertices)
            signed = np.where(spikes > 0.0, lower, -upper)
            minimum_signed_margin = min(
                minimum_signed_margin, float(np.min(signed))
            )
            robust = np.where(spikes > 0.0, lower >= 0.0, upper < 0.0)
            count = int(np.count_nonzero(~robust))
            uncertain_guard_count += count
            if count:
                uncertain_timesteps.add(step)
                if first_uncertain is None:
                    first_uncertain = step

        for step in range(events.shape[1]):
            exact_current = (
                input_drive[:, step, :] + previous_spikes @ w_rec + bias
            )
            current = _HybridAffine.exact(
                np.asarray(state.quantize(exact_current), dtype=np.float64)
            )
            if current_queue:
                current_queue.append(current)
                current = current_queue.pop(0)
            spikes_array = forced_spikes[step][None, :]
            spikes = _HybridAffine.exact(spikes_array)
            if timing is ThresholdTiming.PRE_INTEGRATION:
                inspect_guard(step, voltage.subtract(threshold), spikes_array)
                voltage = AffineGuardFamilyCertifier._integrate(
                    AffineGuardFamilyCertifier._reset(
                        voltage,
                        spikes,
                        threshold,
                        model.reset_value,
                        reset,
                        vertices,
                    ),
                    current,
                    timestep,
                    model.tau_mem,
                    vertices,
                    box.integration_rules[0],
                ).float_quantize(state, vertices)
            else:
                voltage = AffineGuardFamilyCertifier._integrate(
                    voltage,
                    current,
                    timestep,
                    model.tau_mem,
                    vertices,
                    box.integration_rules[0],
                ).float_quantize(state, vertices)
                inspect_guard(step, voltage.subtract(threshold), spikes_array)
                voltage = AffineGuardFamilyCertifier._reset(
                    voltage,
                    spikes,
                    threshold,
                    model.reset_value,
                    reset,
                    vertices,
                ).float_quantize(state, vertices)
            previous_spikes = spikes_array

        trace_robust = uncertain_guard_count == 0
        return FixedTraceAffineResult(
            trace_robust=trace_robust,
            certified=trace_robust and trace_prediction == reference_prediction,
            reference_prediction=reference_prediction,
            trace_prediction=trace_prediction,
            total_guard_count=events.shape[1] * model.hidden_size,
            uncertain_guard_count=uncertain_guard_count,
            uncertain_timestep_count=len(uncertain_timesteps),
            first_uncertain_timestep=first_uncertain,
            minimum_signed_guard_margin=float(minimum_signed_margin),
        )


class PolygonBranchCertifier:
    """Soundly enumerate only guard-uncertain traces inside one polygon."""

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        parameter_vertices: np.ndarray,
        *,
        max_branches: int = 1024,
    ) -> PolygonBranchCertificateResult:
        events = _validate_inputs(model, inputs)
        if len(events) != 1:
            raise ValueError("polygon branch certification accepts one input")
        if max_branches < 1:
            raise ValueError("max_branches must be positive")
        discrete_sizes = (
            len(box.integration_rules),
            len(box.threshold_timings),
            len(box.reset_rules),
            len(box.synaptic_delays),
            len(box.output_delays),
        )
        if discrete_sizes != (1, 1, 1, 1, 1):
            member_results = []
            for integration, timing, reset, synaptic_delay, output_delay in (
                itertools.product(
                    box.integration_rules,
                    box.threshold_timings,
                    box.reset_rules,
                    box.synaptic_delays,
                    box.output_delays,
                )
            ):
                member_box = replace(
                    box,
                    integration_rules=(integration,),
                    threshold_timings=(timing,),
                    reset_rules=(reset,),
                    synaptic_delays=(synaptic_delay,),
                    output_delays=(output_delay,),
                    name=(
                        f"{box.name}:{integration.value}:{timing.value}:"
                        f"{reset.value}:sd{synaptic_delay}:od{output_delay}"
                    ),
                )
                member_result = self.certify(
                    model,
                    events,
                    reference,
                    member_box,
                    parameter_vertices,
                    max_branches=max_branches,
                )
                member_results.append(member_result)
                if not member_result.complete:
                    break
            complete = all(result.complete for result in member_results)
            possible_predictions = tuple(
                sorted(
                    {
                        prediction
                        for result in member_results
                        for prediction in result.possible_predictions
                    }
                )
            )
            reference_prediction = member_results[0].reference_prediction
            first_cap = next(
                (
                    result.first_cap_timestep
                    for result in member_results
                    if not result.complete
                ),
                None,
            )
            return PolygonBranchCertificateResult(
                certified=(
                    complete and possible_predictions == (reference_prediction,)
                ),
                complete=complete,
                reference_prediction=reference_prediction,
                possible_predictions=possible_predictions,
                final_branch_count=sum(
                    result.final_branch_count for result in member_results
                ),
                maximum_active_branches=max(
                    result.maximum_active_branches for result in member_results
                ),
                total_branch_splits=sum(
                    result.total_branch_splits for result in member_results
                ),
                maximum_uncertain_neurons_at_step=max(
                    result.maximum_uncertain_neurons_at_step
                    for result in member_results
                ),
                first_cap_timestep=first_cap,
            )
        if box.base.state_format.is_fixed:
            raise NotImplementedError("polygon branches require floating state")
        if not box.base.randomness.deterministic:
            raise NotImplementedError("polygon branches require determinism")
        vertices = np.asarray(parameter_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("parameter polygon must have shape [vertices, 2]")

        state = box.base.state_format
        weight = box.base.weight_format
        w_in = np.asarray(weight.quantize(model.input_weights), dtype=np.float64)
        w_rec = np.asarray(weight.quantize(model.recurrent_weights), dtype=np.float64)
        w_out = np.asarray(weight.quantize(model.output_weights), dtype=np.float64)
        bias = np.asarray(state.quantize(model.bias), dtype=np.float64)
        input_drive = events @ w_in
        threshold = _HybridAffine.semantic_interval(
            model.threshold[None, :] * box.threshold_scale_bounds[0],
            model.threshold[None, :] * box.threshold_scale_bounds[1],
            1,
        )
        timestep = _HybridAffine.semantic_interval(
            np.full((1, model.hidden_size), box.timestep_bounds[0]),
            np.full((1, model.hidden_size), box.timestep_bounds[1]),
            0,
        )
        timing = box.threshold_timings[0]
        reset = box.reset_rules[0]
        reference_prediction = int(
            VectorizedEmulator().run(model, events, reference).predictions[0]
        )
        branches = [
            _PolygonBranch(
                voltage=_HybridAffine.exact(
                    np.zeros((1, model.hidden_size), dtype=np.float64)
                ),
                previous_spikes=np.zeros(model.hidden_size, dtype=np.float64),
                logits=np.zeros(model.output_size, dtype=np.float64),
                current_queue=tuple(
                    _HybridAffine.exact(
                        np.zeros((1, model.hidden_size), dtype=np.float64)
                    )
                    for _ in range(box.synaptic_delays[0])
                ),
                output_queue=tuple(
                    np.zeros(model.output_size, dtype=np.float64)
                    for _ in range(box.output_delays[0])
                ),
            )
        ]
        maximum_active = 1
        total_splits = 0
        maximum_uncertain = 0

        for step in range(events.shape[1]):
            next_branches: list[_PolygonBranch] = []
            for branch in branches:
                exact_current = (
                    input_drive[0, step, :]
                    + branch.previous_spikes @ w_rec
                    + bias
                )
                raw_current = _HybridAffine.exact(
                    np.asarray(
                        state.quantize(exact_current), dtype=np.float64
                    )[None, :]
                )
                if branch.current_queue:
                    current = branch.current_queue[0]
                    next_current_queue = (
                        *branch.current_queue[1:],
                        raw_current,
                    )
                else:
                    current = raw_current
                    next_current_queue = ()
                if timing is ThresholdTiming.PRE_INTEGRATION:
                    guard_voltage = branch.voltage
                else:
                    guard_voltage = AffineGuardFamilyCertifier._integrate(
                        branch.voltage,
                        current,
                        timestep,
                        model.tau_mem,
                        vertices,
                        box.integration_rules[0],
                    ).float_quantize(state, vertices)
                guard = guard_voltage.subtract(threshold)
                guard_lower, guard_upper = guard.bounds(vertices)
                definitely_spiking = guard_lower[0] >= 0.0
                definitely_quiet = guard_upper[0] < 0.0
                uncertain_indices = np.flatnonzero(
                    ~(definitely_spiking | definitely_quiet)
                )
                maximum_uncertain = max(
                    maximum_uncertain, len(uncertain_indices)
                )
                combination_count = 1 << len(uncertain_indices)
                if len(next_branches) + combination_count > max_branches:
                    return PolygonBranchCertificateResult(
                        certified=False,
                        complete=False,
                        reference_prediction=reference_prediction,
                        possible_predictions=(),
                        final_branch_count=len(branches),
                        maximum_active_branches=maximum_active,
                        total_branch_splits=total_splits,
                        maximum_uncertain_neurons_at_step=maximum_uncertain,
                        first_cap_timestep=step,
                    )
                total_splits += combination_count - 1
                for assignment in itertools.product(
                    (0.0, 1.0), repeat=len(uncertain_indices)
                ):
                    spikes_array = definitely_spiking.astype(np.float64)
                    if len(uncertain_indices):
                        spikes_array = spikes_array.copy()
                        spikes_array[uncertain_indices] = assignment
                    spikes = _HybridAffine.exact(spikes_array[None, :])
                    if timing is ThresholdTiming.PRE_INTEGRATION:
                        next_voltage = AffineGuardFamilyCertifier._integrate(
                            AffineGuardFamilyCertifier._reset(
                                branch.voltage,
                                spikes,
                                threshold,
                                model.reset_value,
                                reset,
                                vertices,
                            ),
                            current,
                            timestep,
                            model.tau_mem,
                            vertices,
                            box.integration_rules[0],
                        ).float_quantize(state, vertices)
                    else:
                        next_voltage = AffineGuardFamilyCertifier._reset(
                            guard_voltage,
                            spikes,
                            threshold,
                            model.reset_value,
                            reset,
                            vertices,
                        ).float_quantize(state, vertices)
                    contribution = np.asarray(
                        state.quantize(spikes_array @ w_out), dtype=np.float64
                    )
                    if branch.output_queue:
                        delivered = branch.output_queue[0]
                        next_output_queue = (
                            *branch.output_queue[1:],
                            contribution,
                        )
                    else:
                        delivered = contribution
                        next_output_queue = ()
                    logits = np.asarray(
                        state.quantize(branch.logits + delivered),
                        dtype=np.float64,
                    )
                    next_branches.append(
                        _PolygonBranch(
                            voltage=next_voltage,
                            previous_spikes=spikes_array,
                            logits=logits,
                            current_queue=next_current_queue,
                            output_queue=next_output_queue,
                        )
                    )
            branches = next_branches
            maximum_active = max(maximum_active, len(branches))

        predictions = tuple(
            sorted({int(np.argmax(branch.logits)) for branch in branches})
        )
        return PolygonBranchCertificateResult(
            certified=predictions == (reference_prediction,),
            complete=True,
            reference_prediction=reference_prediction,
            possible_predictions=predictions,
            final_branch_count=len(branches),
            maximum_active_branches=maximum_active,
            total_branch_splits=total_splits,
            maximum_uncertain_neurons_at_step=maximum_uncertain,
            first_cap_timestep=None,
        )


class AdaptiveHybridPolygonCertifier:
    """Refine polygons only when local spike-branch closure cannot certify."""

    def __init__(
        self,
        *,
        max_branches: int = 64,
        max_guard_band_splits: int | None = None,
        member_certifier: AffineGuardFamilyCertifier | None = None,
        branch_certifier: PolygonBranchCertifier | None = None,
    ) -> None:
        if max_branches < 1:
            raise ValueError("max_branches must be positive")
        if max_guard_band_splits is not None and max_guard_band_splits < 0:
            raise ValueError("maximum guard-band splits must be nonnegative")
        self.max_branches = max_branches
        self.max_guard_band_splits = max_guard_band_splits
        self.member_certifier = member_certifier or AffineGuardFamilyCertifier()
        self.branch_certifier = branch_certifier or PolygonBranchCertifier()

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        *,
        max_leaves: int = 256,
    ) -> AdaptiveHybridCertificateResult:
        events = _validate_inputs(model, inputs)
        if len(events) != 1:
            raise ValueError("adaptive hybrid certification accepts one input")
        if max_leaves < 1:
            raise ValueError("max_leaves must be positive")
        reference_predictions = VectorizedEmulator().run(
            model, events, reference
        ).predictions
        root_polygon = np.asarray(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float64,
        )
        root_area = polygon_area(root_polygon)
        queue: list[tuple[float, int, int, np.ndarray]] = []
        counter = itertools.count()
        heapq.heappush(queue, (-root_area, 0, next(counter), root_polygon))
        leaf_count = 1
        analyzed = 0
        certified_area = 0.0
        unresolved_area = 0.0
        branch_certified_leaves = 0
        affine_certified_leaves = 0
        unresolved_leaves = 0
        branch_attempts = 0
        branch_cap_hits = 0
        branch_prediction_rejections = 0
        maximum_completed_branches = 0
        guard_band_splits = 0
        axis_fallback_splits = 0

        while queue:
            _, depth, _, polygon = heapq.heappop(queue)
            del depth
            area = polygon_area(polygon)
            branch = self.branch_certifier.certify(
                model,
                events,
                reference,
                box,
                polygon,
                max_branches=self.max_branches,
            )
            branch_attempts += 1
            if branch.complete:
                maximum_completed_branches = max(
                    maximum_completed_branches, branch.final_branch_count
                )
                if branch.certified:
                    certified_area += area
                    branch_certified_leaves += 1
                    analyzed += 1
                    continue
                branch_prediction_rejections += 1
            else:
                branch_cap_hits += 1

            result = self.member_certifier.certify_with_reference_predictions(
                model,
                events,
                reference,
                box,
                reference_predictions,
                polygon,
            )
            analyzed += 1
            if bool(result.certified[0]):
                certified_area += area
                affine_certified_leaves += 1
                continue
            children: tuple[np.ndarray, ...] = (polygon,)
            split_kind: str | None = None
            guard_budget_available = (
                self.max_guard_band_splits is None
                or guard_band_splits < self.max_guard_band_splits
            )
            if (
                bool(result.guard_cut_valid[0])
                and leaf_count < max_leaves
                and guard_budget_available
            ):
                children = split_polygon_guard_band(
                    polygon,
                    float(result.guard_cut_center[0]),
                    result.guard_cut_generators[0],
                    float(result.guard_cut_radius[0]),
                )
                split_kind = "guard"

            def is_useful(candidate: tuple[np.ndarray, ...]) -> bool:
                candidate_area = sum(polygon_area(child) for child in candidate)
                return (
                    len(candidate) >= 2
                    and leaf_count + len(candidate) - 1 <= max_leaves
                    and np.isclose(candidate_area, area, rtol=1e-8, atol=1e-12)
                    and max(polygon_area(child) for child in candidate)
                    < area * (1.0 - 1e-12)
                )

            useful_split = is_useful(children)
            if not useful_split and leaf_count < max_leaves:
                coordinate_widths = np.ptp(polygon, axis=0)
                scores = np.asarray(result.split_axis_scores[0]) * coordinate_widths
                axis = int(np.argmax(scores)) if np.any(scores > 0.0) else 0
                lower_coordinate = float(np.min(polygon[:, axis]))
                upper_coordinate = float(np.max(polygon[:, axis]))
                if upper_coordinate > lower_coordinate:
                    midpoint = (lower_coordinate + upper_coordinate) / 2.0
                    normal = np.zeros(2, dtype=np.float64)
                    normal[axis] = 1.0
                    children = (
                        clip_polygon_halfspace(polygon, normal, midpoint),
                        clip_polygon_halfspace(polygon, -normal, -midpoint),
                    )
                    split_kind = "axis"
                    useful_split = is_useful(children)
            if not useful_split:
                unresolved_area += area
                unresolved_leaves += 1
                continue
            leaf_count += len(children) - 1
            if split_kind == "guard":
                guard_band_splits += 1
            else:
                axis_fallback_splits += 1
            for child in children:
                heapq.heappush(
                    queue,
                    (-polygon_area(child), 0, next(counter), child),
                )

        certified_fraction = certified_area / root_area
        unresolved_fraction = unresolved_area / root_area
        if not np.isclose(certified_fraction + unresolved_fraction, 1.0):
            raise AssertionError("hybrid polygons do not cover the root domain")
        return AdaptiveHybridCertificateResult(
            certified=unresolved_leaves == 0,
            certified_parameter_fraction=certified_fraction,
            unresolved_parameter_fraction=unresolved_fraction,
            analyzed_polygons=analyzed,
            final_leaves=(
                branch_certified_leaves
                + affine_certified_leaves
                + unresolved_leaves
            ),
            branch_certified_leaves=branch_certified_leaves,
            affine_certified_leaves=affine_certified_leaves,
            unresolved_leaves=unresolved_leaves,
            branch_attempts=branch_attempts,
            branch_cap_hits=branch_cap_hits,
            branch_prediction_rejections=branch_prediction_rejections,
            maximum_completed_branches=maximum_completed_branches,
            guard_band_splits=guard_band_splits,
            axis_fallback_splits=axis_fallback_splits,
        )


def polygon_area(vertices: np.ndarray) -> float:
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        return 0.0
    x = values[:, 0]
    y = values[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def clip_polygon_halfspace(
    vertices: np.ndarray,
    normal: np.ndarray,
    bound: float,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Clip a convex polygon by ``normal @ point <= bound``."""

    polygon = np.asarray(vertices, dtype=np.float64)
    direction = np.asarray(normal, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon vertices must have shape [vertices, 2]")
    if direction.shape != (2,):
        raise ValueError("halfspace normal must have two entries")
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=np.float64)
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(direction @ previous - bound)
    previous_inside = previous_value <= tolerance
    for current in polygon:
        current_value = float(direction @ current - bound)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > tolerance:
                fraction = previous_value / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current.copy())
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    deduplicated = [output[0]]
    for point in output[1:]:
        if not np.allclose(point, deduplicated[-1], atol=tolerance, rtol=0.0):
            deduplicated.append(point)
    if len(deduplicated) > 1 and np.allclose(
        deduplicated[0], deduplicated[-1], atol=tolerance, rtol=0.0
    ):
        deduplicated.pop()
    return np.asarray(deduplicated, dtype=np.float64)


def split_polygon_guard_band(
    vertices: np.ndarray,
    center: float,
    generators: np.ndarray,
    radius: float,
    *,
    area_tolerance: float = 1e-14,
) -> tuple[np.ndarray, ...]:
    """Split a polygon into quiet, residual guard band, and spiking regions."""

    normal = np.asarray(generators, dtype=np.float64)
    if normal.shape != (2,) or np.linalg.norm(normal) <= 1e-15:
        return (np.asarray(vertices, dtype=np.float64),)
    quiet = clip_polygon_halfspace(vertices, normal, -radius - center)
    band = clip_polygon_halfspace(vertices, normal, radius - center)
    band = clip_polygon_halfspace(band, -normal, radius + center)
    spiking = clip_polygon_halfspace(vertices, -normal, center - radius)
    children = tuple(
        polygon
        for polygon in (quiet, band, spiking)
        if polygon_area(polygon) > area_tolerance
    )
    return children or (np.asarray(vertices, dtype=np.float64),)


class AdaptiveAffineGuardCutCertifier:
    """Sound polygonal branch-and-bound using affine recurrent guard bands."""

    def __init__(
        self,
        member_certifier: AffineGuardFamilyCertifier | None = None,
        *,
        max_guard_band_splits: int | None = None,
    ) -> None:
        if max_guard_band_splits is not None and max_guard_band_splits < 0:
            raise ValueError("maximum guard-band splits must be nonnegative")
        self.member_certifier = member_certifier or AffineGuardFamilyCertifier()
        self.max_guard_band_splits = max_guard_band_splits

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        box: SemanticsBox,
        *,
        max_leaves: int = 4096,
        retain_unresolved_polygons: bool = False,
    ) -> AdaptiveGuardCutCertificateResult:
        events = _validate_inputs(model, inputs)
        if len(events) != 1:
            raise ValueError("adaptive guard cuts currently accept one input")
        if max_leaves < 1:
            raise ValueError("max_leaves must be positive")
        reference_predictions = VectorizedEmulator().run(
            model, events, reference
        ).predictions
        root_polygon = np.asarray(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float64,
        )
        root_area = polygon_area(root_polygon)
        queue: list[tuple[float, int, int, np.ndarray]] = []
        counter = itertools.count()
        heapq.heappush(queue, (-root_area, 0, next(counter), root_polygon))
        leaf_count = 1
        analyzed = 0
        certified_area = 0.0
        unresolved_area = 0.0
        certified_leaves = 0
        unresolved_leaves = 0
        maximum_depth = 0
        maximum_vertices = len(root_polygon)
        guard_band_splits = 0
        axis_fallback_splits = 0
        unresolved_polygons: list[np.ndarray] = []

        while queue:
            _, depth, _, polygon = heapq.heappop(queue)
            maximum_depth = max(maximum_depth, depth)
            maximum_vertices = max(maximum_vertices, len(polygon))
            result = self.member_certifier.certify_with_reference_predictions(
                model,
                events,
                reference,
                box,
                reference_predictions,
                polygon,
            )
            analyzed += 1
            area = polygon_area(polygon)
            if bool(result.certified[0]):
                certified_area += area
                certified_leaves += 1
                continue
            children: tuple[np.ndarray, ...] = (polygon,)
            split_kind: str | None = None
            guard_budget_available = (
                self.max_guard_band_splits is None
                or guard_band_splits < self.max_guard_band_splits
            )
            if (
                bool(result.guard_cut_valid[0])
                and leaf_count < max_leaves
                and guard_budget_available
            ):
                children = split_polygon_guard_band(
                    polygon,
                    float(result.guard_cut_center[0]),
                    result.guard_cut_generators[0],
                    float(result.guard_cut_radius[0]),
                )
                split_kind = "guard"

            def is_useful(candidate: tuple[np.ndarray, ...]) -> bool:
                candidate_area = sum(polygon_area(child) for child in candidate)
                return (
                    len(candidate) >= 2
                    and leaf_count + len(candidate) - 1 <= max_leaves
                    and np.isclose(candidate_area, area, rtol=1e-8, atol=1e-12)
                    and max(polygon_area(child) for child in candidate)
                    < area * (1.0 - 1e-12)
                )

            useful_split = is_useful(children)
            if not useful_split and leaf_count < max_leaves:
                coordinate_widths = np.ptp(polygon, axis=0)
                scores = (
                    np.asarray(result.split_axis_scores[0]) * coordinate_widths
                )
                axis = int(np.argmax(scores)) if np.any(scores > 0.0) else 0
                lower_coordinate = float(np.min(polygon[:, axis]))
                upper_coordinate = float(np.max(polygon[:, axis]))
                if upper_coordinate > lower_coordinate:
                    midpoint = (lower_coordinate + upper_coordinate) / 2.0
                    normal = np.zeros(2, dtype=np.float64)
                    normal[axis] = 1.0
                    children = (
                        clip_polygon_halfspace(polygon, normal, midpoint),
                        clip_polygon_halfspace(polygon, -normal, -midpoint),
                    )
                    split_kind = "axis"
                    useful_split = is_useful(children)
            if not useful_split:
                unresolved_area += area
                unresolved_leaves += 1
                if retain_unresolved_polygons:
                    unresolved_polygons.append(polygon.copy())
                continue
            leaf_count += len(children) - 1
            if split_kind == "guard":
                guard_band_splits += 1
            else:
                axis_fallback_splits += 1
            for child in children:
                child_area_value = polygon_area(child)
                heapq.heappush(
                    queue,
                    (-child_area_value, depth + 1, next(counter), child),
                )

        certified_fraction = certified_area / root_area
        unresolved_fraction = unresolved_area / root_area
        if not np.isclose(certified_fraction + unresolved_fraction, 1.0):
            raise AssertionError("guard-cut polygons do not cover the root domain")
        return AdaptiveGuardCutCertificateResult(
            certified=unresolved_leaves == 0,
            certified_parameter_fraction=certified_fraction,
            unresolved_parameter_fraction=unresolved_fraction,
            analyzed_polygons=analyzed,
            final_leaves=certified_leaves + unresolved_leaves,
            certified_leaves=certified_leaves,
            unresolved_leaves=unresolved_leaves,
            maximum_depth=maximum_depth,
            maximum_polygon_vertices=maximum_vertices,
            guard_band_splits=guard_band_splits,
            axis_fallback_splits=axis_fallback_splits,
            unresolved_polygons=tuple(unresolved_polygons),
        )
