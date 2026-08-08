from __future__ import annotations

from dataclasses import replace

import numpy as np

from transportcert.emulator import VectorizedEmulator
from transportcert.parameter_batch import (
    ReferenceParameterSweepEmulator,
    TorchParameterBatchEmulator,
)
from transportcert.semantics import (
    ExecutionSemantics,
    IntegrationRule,
    NumericFormat,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)
from transportcert.torch_emulator import TorchEmulator, _torch


def test_parameter_batch_matches_individual_torch_runs(small_model, event_batch) -> None:
    torch = _torch()
    semantics = replace(
        ExecutionSemantics(),
        reset_rule=ResetRule.TO_VALUE,
        synaptic_delay_steps=1,
        output_delay_steps=1,
    )
    timesteps = np.asarray([0.91, 0.97, 1.0, 1.04, 1.09])
    threshold_scales = np.asarray([1.08, 1.02, 1.0, 0.96, 0.92])
    inputs = event_batch[:1]
    actual = TorchParameterBatchEmulator(
        dtype=torch.float64
    ).run(
        small_model,
        inputs,
        semantics,
        timesteps,
        threshold_scales,
        batch_size=2,
    )
    expected_logits = []
    expected_predictions = []
    emulator = TorchEmulator(dtype=torch.float64)
    for timestep, threshold_scale in zip(
        timesteps, threshold_scales, strict=True
    ):
        candidate = small_model.with_parameters(
            threshold=small_model.threshold * threshold_scale
        )
        trace = emulator.run(
            candidate,
            inputs,
            replace(semantics, timestep=float(timestep)),
        ).numpy()
        expected_logits.append(trace.final_logits[0])
        expected_predictions.append(trace.predictions[0])
    np.testing.assert_array_equal(actual.final_logits, expected_logits)
    np.testing.assert_array_equal(actual.predictions, expected_predictions)


def test_cartesian_parameter_batch_matches_single_input_batches(
    small_model, event_batch
) -> None:
    torch = _torch()
    semantics = replace(
        ExecutionSemantics(),
        integration_rule=IntegrationRule.EXPONENTIAL_EULER,
        threshold_timing=ThresholdTiming.PRE_INTEGRATION,
        update_ordering=UpdateOrdering.THRESHOLD_RESET_INTEGRATE,
        reset_rule=ResetRule.TO_VALUE,
        synaptic_delay_steps=1,
        output_delay_steps=1,
    )
    timesteps = np.asarray([0.91, 1.0, 1.09])
    threshold_scales = np.asarray([1.08, 1.0, 0.92])
    inputs = event_batch[:2]
    emulator = TorchParameterBatchEmulator(dtype=torch.float64)
    actual = emulator.run_cartesian(
        small_model,
        inputs,
        semantics,
        timesteps,
        threshold_scales,
        input_batch_size=1,
        parameter_batch_size=2,
    )
    expected = [
        emulator.run(
            small_model,
            sample[None, ...],
            semantics,
            timesteps,
            threshold_scales,
            batch_size=2,
        )
        for sample in inputs
    ]
    np.testing.assert_array_equal(
        actual.final_logits,
        np.stack([result.final_logits for result in expected]),
    )
    np.testing.assert_array_equal(
        actual.predictions,
        np.stack([result.predictions for result in expected]),
    )


def test_reference_parameter_sweep_matches_individual_oracle_runs(
    small_model, event_batch
) -> None:
    semantics = ExecutionSemantics(
        state_format=NumericFormat("float32"),
        weight_format=NumericFormat("float32"),
        reset_rule=ResetRule.TO_VALUE,
    )
    timesteps = np.asarray([0.91, 1.0, 1.09])
    threshold_scales = np.asarray([1.08, 1.0, 0.92])
    inputs = event_batch[:2]
    oracle = VectorizedEmulator()
    reference_predictions = oracle.run(
        small_model, inputs, ExecutionSemantics()
    ).predictions
    actual = ReferenceParameterSweepEmulator(oracle).run(
        small_model,
        inputs,
        semantics,
        timesteps,
        threshold_scales,
        reference_predictions,
    )
    expected_predictions = []
    expected_margins = []
    rows = np.arange(len(inputs))
    for timestep, threshold_scale in zip(
        timesteps, threshold_scales, strict=True
    ):
        candidate = small_model.with_parameters(
            threshold=small_model.threshold * threshold_scale
        )
        trace = oracle.run(
            candidate, inputs, replace(semantics, timestep=float(timestep))
        )
        expected_predictions.append(trace.predictions)
        competing = trace.final_logits.copy()
        competing[rows, reference_predictions] = -np.inf
        expected_margins.append(
            trace.final_logits[rows, reference_predictions]
            - np.max(competing, axis=1)
        )
    np.testing.assert_array_equal(
        actual.predictions, np.stack(expected_predictions, axis=1)
    )
    np.testing.assert_array_equal(
        actual.minimum_reference_margin,
        np.min(np.stack(expected_margins, axis=1), axis=1),
    )
