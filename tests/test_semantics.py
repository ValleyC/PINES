from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pines.semantics import (
    ExecutionSemantics,
    NumericFormat,
    OverflowMode,
    Randomness,
    ResetRule,
    RoundingMode,
    ThresholdTiming,
    UpdateOrdering,
)


def test_semantics_description_is_stable_and_sensitive() -> None:
    first = ExecutionSemantics()
    second = ExecutionSemantics()
    changed = replace(first, reset_rule=ResetRule.TO_VALUE)
    assert first.semantics_description == second.semantics_description
    assert first.semantics_description != changed.semantics_description


def test_order_and_timing_must_agree() -> None:
    with pytest.raises(ValueError, match="different orders"):
        ExecutionSemantics(
            threshold_timing=ThresholdTiming.PRE_INTEGRATION,
            update_ordering=UpdateOrdering.INTEGRATE_THRESHOLD_RESET,
        )


def test_fixed_saturation_and_wrap() -> None:
    saturated = NumericFormat(
        "fixed", 4, 1, RoundingMode.NEAREST_EVEN, OverflowMode.SATURATE
    )
    wrapped = NumericFormat(
        "fixed", 4, 1, RoundingMode.NEAREST_EVEN, OverflowMode.WRAP
    )
    assert saturated.quantize(9.0) == 3.5
    assert saturated.quantize(-9.0) == -4.0
    assert wrapped.quantize(4.0) == -4.0


def test_nearest_even_is_declared_not_implicit() -> None:
    numeric = NumericFormat("fixed", 8, 2, RoundingMode.NEAREST_EVEN)
    assert numeric.quantize(0.125) == 0.0
    assert numeric.quantize(0.375) == 0.5


def test_float32_format_records_runtime_rounding() -> None:
    numeric = NumericFormat("float32")
    value = np.asarray([1.0 / 3.0], dtype=np.float64)
    result = numeric.quantize(value)
    assert result.dtype == np.float32
    assert float(result[0]) == float(np.float32(1.0 / 3.0))
    assert float(result[0]) != float(value[0])


def test_stochastic_rounding_requires_stochastic_semantics() -> None:
    stochastic = NumericFormat("fixed", 8, 4, RoundingMode.STOCHASTIC)
    with pytest.raises(ValueError, match="deterministic=false"):
        ExecutionSemantics(state_format=stochastic)
    valid = ExecutionSemantics(
        state_format=stochastic, randomness=Randomness(deterministic=False, seed=2)
    )
    assert not valid.randomness.deterministic


@pytest.mark.parametrize("delay", [-1, -2])
def test_negative_delays_rejected(delay: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ExecutionSemantics(synaptic_delay_steps=delay)
