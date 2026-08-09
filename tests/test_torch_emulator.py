from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pines.emulator import VectorizedEmulator
from pines.semantics import ExecutionSemantics, NumericFormat, ResetRule
from pines.torch_emulator import TorchEmulator


@pytest.mark.parametrize("reset", list(ResetRule))
@pytest.mark.parametrize("delay", [0, 1])
def test_torch_matches_numpy_float(small_model, event_batch, reset, delay) -> None:
    semantics = ExecutionSemantics(
        reset_rule=reset, synaptic_delay_steps=delay, output_delay_steps=delay
    )
    expected = VectorizedEmulator().run(small_model, event_batch[:6], semantics)
    actual = TorchEmulator().run(small_model, event_batch[:6], semantics).numpy()
    np.testing.assert_allclose(actual.membrane, expected.membrane, atol=1e-12)
    np.testing.assert_array_equal(actual.spikes, expected.spikes)
    np.testing.assert_allclose(actual.final_logits, expected.final_logits, atol=1e-12)


def test_torch_matches_numpy_fixed(small_model, event_batch) -> None:
    semantics = ExecutionSemantics(
        state_format=NumericFormat("fixed", 12, 6),
        weight_format=NumericFormat("fixed", 8, 5),
    )
    expected = VectorizedEmulator().run(small_model, event_batch[:6], semantics)
    actual = TorchEmulator().run(small_model, event_batch[:6], semantics).numpy()
    np.testing.assert_array_equal(actual.membrane, expected.membrane)
    np.testing.assert_array_equal(actual.final_logits, expected.final_logits)


@pytest.mark.parametrize("reset", list(ResetRule))
def test_torch_matches_numpy_at_explicit_float32_cast_points(
    small_model, event_batch, reset
) -> None:
    float32 = NumericFormat("float32")
    semantics = ExecutionSemantics(
        reset_rule=reset,
        state_format=float32,
        weight_format=float32,
        synaptic_delay_steps=1,
        output_delay_steps=1,
    )
    expected = VectorizedEmulator().run(small_model, event_batch[:6], semantics)
    actual = TorchEmulator(dtype=torch.float64).run(
        small_model, event_batch[:6], semantics
    ).numpy()
    np.testing.assert_allclose(
        actual.membrane,
        expected.membrane,
        rtol=0.0,
        atol=np.finfo(np.float32).eps,
    )
    np.testing.assert_array_equal(actual.spikes, expected.spikes)
    np.testing.assert_allclose(
        actual.final_logits,
        expected.final_logits,
        rtol=0.0,
        atol=np.finfo(np.float32).eps,
    )
    np.testing.assert_array_equal(actual.predictions, expected.predictions)
