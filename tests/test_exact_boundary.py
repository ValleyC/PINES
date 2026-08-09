from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pines.emulator import VectorizedEmulator
from pines.exact_boundary import ExactThresholdBoundaryOracle
from pines.models import DenseRecurrentSNN
from pines.semantics import ExecutionSemantics, ResetRule


def _one_neuron_model(output_weights: np.ndarray) -> DenseRecurrentSNN:
    return DenseRecurrentSNN(
        input_weights=np.asarray([[1.0]]),
        recurrent_weights=np.zeros((1, 1)),
        output_weights=np.asarray(output_weights, dtype=np.float64),
        bias=np.zeros(1),
        threshold=np.ones(1),
        tau_mem=np.ones(1),
        reset_value=np.zeros(1),
    )


def test_exact_threshold_oracle_covers_stable_binary64_family() -> None:
    model = _one_neuron_model([[1.0, 0.0]])
    reference = ExecutionSemantics()
    target = replace(reference, reset_rule=ResetRule.TO_VALUE)
    inputs = np.ones((1, 2, 1))
    result = ExactThresholdBoundaryOracle().certify(
        model,
        inputs,
        reference,
        target,
        (0.5, 1.5),
    )
    assert result.certified
    assert result.conclusive
    assert result.uncovered_representable_values == 0
    assert result.observed_predictions == (0,)
    assert len(result.cells) >= 2


def test_exact_threshold_oracle_returns_prediction_counterexample() -> None:
    model = _one_neuron_model([[0.0, 1.0]])
    reference = ExecutionSemantics()
    target = replace(reference, reset_rule=ResetRule.TO_VALUE)
    inputs = np.ones((1, 1, 1))
    result = ExactThresholdBoundaryOracle().certify(
        model,
        inputs,
        reference,
        target,
        (0.5, 1.5),
    )
    assert not result.certified
    assert result.conclusive
    assert result.counterexample_scale is not None
    candidate = model.with_parameters(
        threshold=model.threshold * result.counterexample_scale
    )
    prediction = VectorizedEmulator().run(candidate, inputs, target).predictions[0]
    assert prediction != result.reference_prediction


def test_exact_threshold_oracle_rejects_threshold_dependent_reset() -> None:
    model = _one_neuron_model([[1.0, 0.0]])
    semantics = ExecutionSemantics()
    with pytest.raises(NotImplementedError, match="reset-to-value"):
        ExactThresholdBoundaryOracle().certify(
            model,
            np.ones((1, 1, 1)),
            semantics,
            semantics,
            (0.9, 1.1),
        )


def test_exact_threshold_trace_matches_emulator(small_model, event_batch) -> None:
    semantics = replace(
        ExecutionSemantics(),
        reset_rule=ResetRule.TO_VALUE,
        synaptic_delay_steps=1,
    )
    events = event_batch[0]
    for scale in (0.9, 1.0, 1.1):
        exact = ExactThresholdBoundaryOracle._execute(
            small_model,
            events,
            semantics,
            scale,
        )
        candidate = small_model.with_parameters(
            threshold=small_model.threshold * scale
        )
        emulated = VectorizedEmulator().run(
            candidate,
            events[None, ...],
            semantics,
        )
        np.testing.assert_array_equal(exact.spikes, emulated.spikes[0])
        assert exact.prediction == int(emulated.predictions[0])
