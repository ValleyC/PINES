from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .emulator import VectorizedEmulator, _decay_and_drive, _validate_inputs
from .models import DenseRecurrentSNN
from .semantics import ExecutionSemantics, ResetRule, RoundingMode, ThresholdTiming


@dataclass(frozen=True)
class ThresholdTraceCell:
    """A contiguous set of binary64 threshold scales with one spike trace."""

    lower_scale: float
    upper_scale: float
    prediction: int
    trace_key: str


@dataclass(frozen=True)
class ExactThresholdCertificateResult:
    """Exact fixed-timestep result over all binary64 scales in a closed range."""

    certified: bool
    conclusive: bool
    reference_prediction: int
    observed_predictions: tuple[int, ...]
    threshold_scale_bounds: tuple[float, float]
    cells: tuple[ThresholdTraceCell, ...]
    unique_trace_count: int
    execution_count: int
    uncovered_representable_values: int
    counterexample_scale: float | None
    reason: str


@dataclass(frozen=True)
class _ThresholdTrace:
    prediction: int
    spikes: np.ndarray
    guard_voltage: np.ndarray


def _positive_float_order(value: float) -> int:
    scalar = np.float64(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("threshold scales must be finite and positive")
    return int(scalar.view(np.uint64))


def _float_from_positive_order(order: int) -> float:
    return float(np.asarray(order, dtype=np.uint64).view(np.float64))


def _first_true_order(
    predicate,
    lower_order: int,
    upper_order: int,
) -> int:
    if predicate(_float_from_positive_order(lower_order)):
        return lower_order
    if not predicate(_float_from_positive_order(upper_order)):
        raise AssertionError("monotone predicate is false at its required endpoint")
    lower = lower_order
    upper = upper_order
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if predicate(_float_from_positive_order(middle)):
            upper = middle
        else:
            lower = middle
    return upper


def _last_true_order(
    predicate,
    lower_order: int,
    upper_order: int,
) -> int:
    if predicate(_float_from_positive_order(upper_order)):
        return upper_order
    if not predicate(_float_from_positive_order(lower_order)):
        raise AssertionError("monotone predicate is false at its required endpoint")
    lower = lower_order
    upper = upper_order
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if predicate(_float_from_positive_order(middle)):
            lower = middle
        else:
            upper = middle
    return lower


class ExactThresholdBoundaryOracle:
    """Exhaust spike-trace cells for a shared threshold scale at fixed timestep.

    The proof is exact for the implemented binary64 scale parameter: every
    representable value in the requested closed interval is assigned to a cell.
    Reset-to-value is required because, conditional on a spike trace, threshold
    does not enter the state update. The guard comparisons are evaluated with
    the same NumPy multiplication and comparison used by the emulator.
    """

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        target: ExecutionSemantics,
        threshold_scale_bounds: tuple[float, float],
        *,
        max_cells: int = 100_000,
    ) -> ExactThresholdCertificateResult:
        events = _validate_inputs(model, inputs)
        if len(events) != 1:
            raise ValueError("exact threshold oracle currently accepts one input")
        if target.reset_rule is not ResetRule.TO_VALUE:
            raise NotImplementedError(
                "exact threshold cells require reset-to-value execution"
            )
        if not target.randomness.deterministic:
            raise NotImplementedError("exact threshold cells require determinism")
        if (
            target.state_format.rounding is RoundingMode.STOCHASTIC
            or target.weight_format.rounding is RoundingMode.STOCHASTIC
        ):
            raise NotImplementedError("stochastic rounding is not supported")
        if max_cells < 1:
            raise ValueError("max_cells must be positive")
        lower_scale, upper_scale = map(float, threshold_scale_bounds)
        lower_order = _positive_float_order(lower_scale)
        upper_order = _positive_float_order(upper_scale)
        if lower_order > upper_order:
            raise ValueError("threshold scale bounds must be ordered")

        reference_prediction = int(
            VectorizedEmulator().run(model, events, reference).predictions[0]
        )
        pending: list[tuple[int, int]] = [(lower_order, upper_order)]
        cells: list[ThresholdTraceCell] = []
        trace_keys: dict[bytes, str] = {}
        observed_predictions: set[int] = set()
        execution_count = 0

        while pending:
            if len(cells) >= max_cells:
                uncovered = sum(end - start + 1 for start, end in pending)
                return ExactThresholdCertificateResult(
                    certified=False,
                    conclusive=False,
                    reference_prediction=reference_prediction,
                    observed_predictions=tuple(sorted(observed_predictions)),
                    threshold_scale_bounds=(lower_scale, upper_scale),
                    cells=tuple(cells),
                    unique_trace_count=len(trace_keys),
                    execution_count=execution_count,
                    uncovered_representable_values=uncovered,
                    counterexample_scale=None,
                    reason="cell budget exhausted",
                )
            start, end = heapq.heappop(pending)
            query_order = (start + end) // 2
            query_scale = _float_from_positive_order(query_order)
            trace = self._execute(model, events[0], target, query_scale)
            execution_count += 1
            valid_start, valid_end = self._valid_scale_orders(
                model,
                trace,
                lower_order,
                upper_order,
            )
            if not valid_start <= query_order <= valid_end:
                raise AssertionError("executed trace does not contain its query scale")
            covered_start = max(start, valid_start)
            covered_end = min(end, valid_end)
            packed_trace = np.packbits(trace.spikes != 0).tobytes()
            trace_key = trace_keys.setdefault(
                packed_trace, f"trace-{len(trace_keys) + 1:06d}"
            )
            observed_predictions.add(trace.prediction)
            cell = ThresholdTraceCell(
                lower_scale=_float_from_positive_order(covered_start),
                upper_scale=_float_from_positive_order(covered_end),
                prediction=trace.prediction,
                trace_key=trace_key,
            )
            cells.append(cell)
            if trace.prediction != reference_prediction:
                uncovered = sum(
                    pending_end - pending_start + 1
                    for pending_start, pending_end in pending
                )
                uncovered += covered_start - start
                uncovered += end - covered_end
                return ExactThresholdCertificateResult(
                    certified=False,
                    conclusive=True,
                    reference_prediction=reference_prediction,
                    observed_predictions=tuple(sorted(observed_predictions)),
                    threshold_scale_bounds=(lower_scale, upper_scale),
                    cells=tuple(cells),
                    unique_trace_count=len(trace_keys),
                    execution_count=execution_count,
                    uncovered_representable_values=uncovered,
                    counterexample_scale=query_scale,
                    reason="exact prediction counterexample",
                )
            if start < covered_start:
                heapq.heappush(pending, (start, covered_start - 1))
            if covered_end < end:
                heapq.heappush(pending, (covered_end + 1, end))

        cells.sort(key=lambda cell: _positive_float_order(cell.lower_scale))
        return ExactThresholdCertificateResult(
            certified=True,
            conclusive=True,
            reference_prediction=reference_prediction,
            observed_predictions=tuple(sorted(observed_predictions)),
            threshold_scale_bounds=(lower_scale, upper_scale),
            cells=tuple(cells),
            unique_trace_count=len(trace_keys),
            execution_count=execution_count,
            uncovered_representable_values=0,
            counterexample_scale=None,
            reason="all representable threshold scales preserve the prediction",
        )

    @staticmethod
    def _valid_scale_orders(
        model: DenseRecurrentSNN,
        trace: _ThresholdTrace,
        lower_order: int,
        upper_order: int,
    ) -> tuple[int, int]:
        thresholds = np.broadcast_to(model.threshold, trace.guard_voltage.shape)
        quiet_voltage = trace.guard_voltage[trace.spikes == 0.0]
        quiet_threshold = thresholds[trace.spikes == 0.0]
        spike_voltage = trace.guard_voltage[trace.spikes == 1.0]
        spike_threshold = thresholds[trace.spikes == 1.0]

        valid_lower = lower_order
        if len(quiet_voltage):
            def all_quiet(scale: float) -> bool:
                return bool(np.all(quiet_voltage < quiet_threshold * scale))

            valid_lower = _first_true_order(
                all_quiet,
                lower_order,
                upper_order,
            )

        valid_upper = upper_order
        if len(spike_voltage):
            def all_spiking(scale: float) -> bool:
                return bool(np.all(spike_voltage >= spike_threshold * scale))

            valid_upper = _last_true_order(
                all_spiking,
                lower_order,
                upper_order,
            )
        return valid_lower, valid_upper

    @staticmethod
    def _execute(
        model: DenseRecurrentSNN,
        events: np.ndarray,
        semantics: ExecutionSemantics,
        threshold_scale: float,
    ) -> _ThresholdTrace:
        rng = np.random.default_rng(semantics.randomness.seed)
        q_state = semantics.state_format
        q_weight = semantics.weight_format
        w_in = np.asarray(q_weight.quantize(model.input_weights, rng))
        w_rec = np.asarray(q_weight.quantize(model.recurrent_weights, rng))
        w_out = np.asarray(q_weight.quantize(model.output_weights, rng))
        bias = np.asarray(q_state.quantize(model.bias, rng))
        threshold = model.threshold * np.float64(threshold_scale)
        voltage = np.zeros(model.hidden_size, dtype=np.float64)
        previous_spikes = np.zeros_like(voltage)
        logits = np.zeros(model.output_size, dtype=np.float64)
        current_queue = [
            np.zeros_like(voltage) for _ in range(semantics.synaptic_delay_steps)
        ]
        output_queue = [
            np.zeros_like(logits) for _ in range(semantics.output_delay_steps)
        ]
        guard_history: list[np.ndarray] = []
        spike_history: list[np.ndarray] = []

        for step in range(events.shape[0]):
            raw_current = events[step] @ w_in + previous_spikes @ w_rec + bias
            raw_current = np.asarray(q_state.quantize(raw_current, rng))
            if current_queue:
                current_queue.append(raw_current)
                current = current_queue.pop(0)
            else:
                current = raw_current
            if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                guard_voltage = voltage.copy()
                spikes = (guard_voltage >= threshold).astype(np.float64)
                voltage = np.where(spikes > 0.0, model.reset_value, voltage)
                voltage = np.asarray(
                    _decay_and_drive(
                        voltage,
                        current,
                        model.tau_mem,
                        semantics,
                    )
                )
            else:
                voltage = np.asarray(
                    _decay_and_drive(
                        voltage,
                        current,
                        model.tau_mem,
                        semantics,
                    )
                )
                voltage = np.asarray(q_state.quantize(voltage, rng))
                guard_voltage = voltage.copy()
                spikes = (guard_voltage >= threshold).astype(np.float64)
                voltage = np.where(spikes > 0.0, model.reset_value, voltage)
            voltage = np.asarray(q_state.quantize(voltage, rng))
            contribution = np.asarray(q_state.quantize(spikes @ w_out, rng))
            if output_queue:
                output_queue.append(contribution)
                delivered = output_queue.pop(0)
            else:
                delivered = contribution
            logits = np.asarray(q_state.quantize(logits + delivered, rng))
            previous_spikes = spikes
            guard_history.append(guard_voltage)
            spike_history.append(spikes)
        return _ThresholdTrace(
            prediction=int(np.argmax(logits)),
            spikes=np.stack(spike_history),
            guard_voltage=np.stack(guard_history),
        )
