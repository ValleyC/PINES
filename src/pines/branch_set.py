from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .abstract import (
    SemanticsBox,
    _integrate_interval,
    _quantize_interval,
)
from .emulator import _validate_inputs
from .models import DenseRecurrentSNN
from .semantics import (
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
)


@dataclass(frozen=True)
class BranchSetCertificateResult:
    complete: bool
    certified: bool
    reachable_predictions: tuple[int, ...]
    final_state_branches: int
    maximum_state_branches: int
    aborted_step: int | None


def _repeat_rows(values: np.ndarray, parents: np.ndarray) -> np.ndarray:
    return np.asarray(values)[parents]


class BranchSetMemberCertifier:
    """Sound path-separated interval analysis for one semantics member and input.

    The implementation keeps each possible spike vector separate and merges only
    states with identical current spikes and accumulated logits. It currently
    supports zero-delay deterministic members; delayed members require queue-aware
    merge keys and are deliberately rejected rather than approximated unsafely.
    """

    def certify(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        box: SemanticsBox,
        reference_prediction: int,
        *,
        max_branches: int = 4096,
        merge_equivalent: bool = True,
    ) -> BranchSetCertificateResult:
        events = _validate_inputs(model, inputs)
        if events.shape[0] != 1:
            raise ValueError("branch-set certification currently accepts one input")
        if max_branches < 1:
            raise ValueError("max_branches must be positive")
        discrete_axes = (
            box.integration_rules,
            box.threshold_timings,
            box.reset_rules,
            box.synaptic_delays,
            box.output_delays,
        )
        if any(len(axis) != 1 for axis in discrete_axes):
            raise ValueError("a branch-set member must have singleton discrete axes")
        if box.synaptic_delays != (0,) or box.output_delays != (0,):
            raise NotImplementedError("branch-set queue separation is not implemented")
        if not box.base.randomness.deterministic:
            raise NotImplementedError("branch-set analysis requires deterministic execution")

        integration = box.integration_rules[0]
        timing = box.threshold_timings[0]
        reset = box.reset_rules[0]
        numeric = box.base.state_format
        weight = box.base.weight_format
        w_in = np.asarray(weight.quantize(model.input_weights))
        w_rec = np.asarray(weight.quantize(model.recurrent_weights))
        w_out = np.asarray(weight.quantize(model.output_weights))
        bias = np.asarray(numeric.quantize(model.bias))
        base_drive = events[0] @ w_in + bias
        threshold_lower = model.threshold * box.threshold_scale_bounds[0]
        threshold_upper = model.threshold * box.threshold_scale_bounds[1]

        lower = np.zeros((1, model.hidden_size), dtype=np.float64)
        upper = lower.copy()
        previous_spikes = np.zeros_like(lower)
        logits = np.zeros((1, model.output_size), dtype=np.float64)
        maximum_seen = 1

        for step in range(events.shape[1]):
            current = base_drive[step][None, :] + previous_spikes @ w_rec
            current_lower, current_upper = _quantize_interval(current, current, numeric)
            if timing is ThresholdTiming.PRE_INTEGRATION:
                (
                    branch_lower,
                    branch_upper,
                    branch_spikes,
                    branch_logits,
                    complete,
                ) = self._split_threshold(
                    lower,
                    upper,
                    logits,
                    threshold_lower,
                    threshold_upper,
                    model.reset_value,
                    reset,
                    max_branches,
                )
                if not complete:
                    return BranchSetCertificateResult(
                        complete=False,
                        certified=False,
                        reachable_predictions=(),
                        final_state_branches=0,
                        maximum_state_branches=maximum_seen,
                        aborted_step=step,
                    )
                parent_counts = self._parent_counts(
                    lower, upper, threshold_lower, threshold_upper
                )
                parents = np.repeat(np.arange(len(lower)), parent_counts)
                current_lower = _repeat_rows(current_lower, parents)
                current_upper = _repeat_rows(current_upper, parents)
                lower, upper = _integrate_interval(
                    branch_lower,
                    branch_upper,
                    current_lower,
                    current_upper,
                    model.tau_mem,
                    box.timestep_bounds,
                    integration,
                )
                logits = branch_logits
                previous_spikes = branch_spikes
            else:
                lower, upper = _integrate_interval(
                    lower,
                    upper,
                    current_lower,
                    current_upper,
                    model.tau_mem,
                    box.timestep_bounds,
                    integration,
                )
                lower, upper = _quantize_interval(lower, upper, numeric)
                (
                    lower,
                    upper,
                    previous_spikes,
                    logits,
                    complete,
                ) = self._split_threshold(
                    lower,
                    upper,
                    logits,
                    threshold_lower,
                    threshold_upper,
                    model.reset_value,
                    reset,
                    max_branches,
                )
                if not complete:
                    return BranchSetCertificateResult(
                        complete=False,
                        certified=False,
                        reachable_predictions=(),
                        final_state_branches=0,
                        maximum_state_branches=maximum_seen,
                        aborted_step=step,
                    )
            lower, upper = _quantize_interval(lower, upper, numeric)
            contribution = np.asarray(
                numeric.quantize(previous_spikes @ w_out), dtype=np.float64
            )
            logits = np.asarray(numeric.quantize(logits + contribution), dtype=np.float64)
            if merge_equivalent:
                lower, upper, previous_spikes, logits = self._merge_equivalent(
                    lower, upper, previous_spikes, logits
                )
            maximum_seen = max(maximum_seen, len(lower))
            if len(lower) > max_branches:
                return BranchSetCertificateResult(
                    complete=False,
                    certified=False,
                    reachable_predictions=(),
                    final_state_branches=len(lower),
                    maximum_state_branches=maximum_seen,
                    aborted_step=step,
                )

        predictions = tuple(sorted({int(value) for value in np.argmax(logits, axis=1)}))
        return BranchSetCertificateResult(
            complete=True,
            certified=predictions == (int(reference_prediction),),
            reachable_predictions=predictions,
            final_state_branches=len(lower),
            maximum_state_branches=maximum_seen,
            aborted_step=None,
        )

    @staticmethod
    def _parent_counts(
        lower: np.ndarray,
        upper: np.ndarray,
        threshold_lower: np.ndarray,
        threshold_upper: np.ndarray,
    ) -> np.ndarray:
        definite_quiet = upper < threshold_lower
        definite_spiking = lower >= threshold_upper
        uncertain_counts = np.count_nonzero(
            ~(definite_quiet | definite_spiking), axis=1
        )
        return np.left_shift(np.ones_like(uncertain_counts), uncertain_counts)

    @staticmethod
    def _split_threshold(
        lower: np.ndarray,
        upper: np.ndarray,
        logits: np.ndarray,
        threshold_lower: np.ndarray,
        threshold_upper: np.ndarray,
        reset_value: np.ndarray,
        reset: ResetRule,
        max_branches: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        definite_quiet = upper < threshold_lower
        definite_spiking = lower >= threshold_upper
        uncertain = ~(definite_quiet | definite_spiking)
        uncertain_counts = np.count_nonzero(uncertain, axis=1)
        if np.any(uncertain_counts > int(math.floor(math.log2(max_branches)))):
            empty_state = np.empty((0, lower.shape[1]), dtype=np.float64)
            empty_logits = np.empty((0, logits.shape[1]), dtype=np.float64)
            return empty_state, empty_state.copy(), empty_state.copy(), empty_logits, False
        branch_counts = np.left_shift(np.ones_like(uncertain_counts), uncertain_counts)
        if int(np.sum(branch_counts)) > max_branches:
            empty_state = np.empty((0, lower.shape[1]), dtype=np.float64)
            empty_logits = np.empty((0, logits.shape[1]), dtype=np.float64)
            return empty_state, empty_state.copy(), empty_state.copy(), empty_logits, False

        lower_rows: list[np.ndarray] = []
        upper_rows: list[np.ndarray] = []
        spike_rows: list[np.ndarray] = []
        logit_rows: list[np.ndarray] = []
        for parent in range(len(lower)):
            uncertain_indices = np.flatnonzero(uncertain[parent])
            for bits in range(1 << len(uncertain_indices)):
                spikes = definite_spiking[parent].astype(np.float64)
                for offset, neuron in enumerate(uncertain_indices):
                    spikes[neuron] = float((bits >> offset) & 1)
                next_lower = lower[parent].copy()
                next_upper = upper[parent].copy()
                active = spikes.astype(bool)
                if reset is ResetRule.SUBTRACTIVE:
                    next_lower[active] -= threshold_upper[active]
                    next_upper[active] -= threshold_lower[active]
                else:
                    next_lower[active] = reset_value[active]
                    next_upper[active] = reset_value[active]
                lower_rows.append(next_lower)
                upper_rows.append(next_upper)
                spike_rows.append(spikes)
                logit_rows.append(logits[parent].copy())
        return (
            np.stack(lower_rows),
            np.stack(upper_rows),
            np.stack(spike_rows),
            np.stack(logit_rows),
            True,
        )

    @staticmethod
    def _merge_equivalent(
        lower: np.ndarray,
        upper: np.ndarray,
        spikes: np.ndarray,
        logits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        groups: dict[bytes, list[int]] = {}
        for index in range(len(lower)):
            key = spikes[index].tobytes() + logits[index].tobytes()
            groups.setdefault(key, []).append(index)
        merged_lower = []
        merged_upper = []
        merged_spikes = []
        merged_logits = []
        for indices in groups.values():
            merged_lower.append(np.min(lower[indices], axis=0))
            merged_upper.append(np.max(upper[indices], axis=0))
            merged_spikes.append(spikes[indices[0]])
            merged_logits.append(logits[indices[0]])
        return (
            np.stack(merged_lower),
            np.stack(merged_upper),
            np.stack(merged_spikes),
            np.stack(merged_logits),
        )
