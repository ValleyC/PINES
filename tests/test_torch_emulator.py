from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transportcert.emulator import VectorizedEmulator
from transportcert.semantics import ExecutionSemantics, NumericFormat, ResetRule
from transportcert.torch_emulator import TorchEmulator


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
