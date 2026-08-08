from __future__ import annotations

from dataclasses import replace

import numpy as np

from transportcert.parameter_batch import TorchParameterBatchEmulator
from transportcert.semantics import ExecutionSemantics, ResetRule
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
