from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from transportcert.emulator import ScalarInterpreter, VectorizedEmulator
from transportcert.semantics import (
    ExecutionSemantics,
    IntegrationRule,
    NumericFormat,
    OverflowMode,
    ResetRule,
    RoundingMode,
    ThresholdTiming,
    UpdateOrdering,
)


@pytest.mark.parametrize("integration", list(IntegrationRule))
@pytest.mark.parametrize("reset", list(ResetRule))
@pytest.mark.parametrize("delay", [0, 1, 2])
def test_scalar_vector_differential_float(
    small_model, event_batch, integration, reset, delay
) -> None:
    semantics = ExecutionSemantics(
        integration_rule=integration,
        reset_rule=reset,
        synaptic_delay_steps=delay,
        output_delay_steps=min(delay, 1),
    )
    scalar = ScalarInterpreter().run(small_model, event_batch[:5], semantics)
    vector = VectorizedEmulator().run(small_model, event_batch[:5], semantics)
    np.testing.assert_allclose(scalar.membrane, vector.membrane, atol=1e-12)
    np.testing.assert_array_equal(scalar.spikes, vector.spikes)
    np.testing.assert_allclose(scalar.final_logits, vector.final_logits, atol=1e-12)


def test_scalar_vector_differential_fixed(small_model, event_batch) -> None:
    semantics = ExecutionSemantics(
        state_format=NumericFormat(
            "fixed", 12, 6, RoundingMode.NEAREST_EVEN, OverflowMode.SATURATE
        ),
        weight_format=NumericFormat(
            "fixed", 8, 5, RoundingMode.NEAREST_EVEN, OverflowMode.SATURATE
        ),
    )
    scalar = ScalarInterpreter().run(small_model, event_batch[:8], semantics)
    vector = VectorizedEmulator().run(small_model, event_batch[:8], semantics)
    np.testing.assert_array_equal(scalar.membrane, vector.membrane)
    np.testing.assert_array_equal(scalar.spikes, vector.spikes)
    np.testing.assert_array_equal(scalar.final_logits, vector.final_logits)


def test_pre_integration_order_is_operational(small_model, event_batch) -> None:
    pre = ExecutionSemantics(
        threshold_timing=ThresholdTiming.PRE_INTEGRATION,
        update_ordering=UpdateOrdering.THRESHOLD_RESET_INTEGRATE,
    )
    post = ExecutionSemantics()
    pre_trace = VectorizedEmulator().run(small_model, event_batch, pre)
    post_trace = VectorizedEmulator().run(small_model, event_batch, post)
    assert not np.array_equal(pre_trace.spikes, post_trace.spikes)


def test_delay_is_explicit_state(small_model, event_batch) -> None:
    immediate = VectorizedEmulator().run(small_model, event_batch, ExecutionSemantics())
    delayed = VectorizedEmulator().run(
        small_model, event_batch, ExecutionSemantics(synaptic_delay_steps=1)
    )
    assert np.all(delayed.spikes[:, 0] == 0)
    assert not np.array_equal(immediate.spikes, delayed.spikes)


def test_bad_input_shape_fails(small_model) -> None:
    with pytest.raises(ValueError, match="inputs must be"):
        VectorizedEmulator().run(small_model, np.zeros((4, 5, 3)), ExecutionSemantics())

