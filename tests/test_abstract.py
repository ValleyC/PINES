from __future__ import annotations

from dataclasses import replace

import numpy as np

from transportcert.abstract import (
    IntervalFamilyCertifier,
    SemanticsBox,
    _linear_interval,
    partition_semantics_box,
)
from transportcert.emulator import VectorizedEmulator
from transportcert.models import DenseRecurrentSNN
from transportcert.semantics import (
    ExecutionSemantics,
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
)


def test_interval_box_contains_sampled_execution(small_model, event_batch) -> None:
    reference = ExecutionSemantics()
    box = SemanticsBox(
        base=reference,
        timestep_bounds=(0.9, 1.1),
        threshold_scale_bounds=(0.9, 1.1),
        integration_rules=(IntegrationRule.FORWARD_EULER, IntegrationRule.EXPONENTIAL_EULER),
        reset_rules=(ResetRule.SUBTRACTIVE, ResetRule.TO_VALUE),
        synaptic_delays=(0, 1),
        output_delays=(0, 1),
    )
    inputs = event_batch[:4]
    result = IntervalFamilyCertifier().certify(small_model, inputs, reference, box)
    for dt in (0.9, 1.0, 1.1):
        for integration in box.integration_rules:
            for reset in box.reset_rules:
                for delay in box.synaptic_delays:
                    semantics = replace(
                        reference,
                        timestep=dt,
                        integration_rule=integration,
                        reset_rule=reset,
                        synaptic_delay_steps=delay,
                    )
                    for threshold_scale in (0.9, 1.0, 1.1):
                        candidate = small_model.with_parameters(
                            threshold=small_model.threshold * threshold_scale
                        )
                        logits = VectorizedEmulator().run(candidate, inputs, semantics).final_logits
                        assert np.all(logits >= result.target_logit_lower - 1e-12)
                        assert np.all(logits <= result.target_logit_upper + 1e-12)


def test_interval_identity_contains_exact_logits(small_model, event_batch) -> None:
    semantics = ExecutionSemantics()
    box = SemanticsBox(
        base=semantics,
        timestep_bounds=(1.0, 1.0),
        integration_rules=(semantics.integration_rule,),
        threshold_timings=(semantics.threshold_timing,),
        reset_rules=(semantics.reset_rule,),
        synaptic_delays=(0,),
        output_delays=(0,),
    )
    inputs = event_batch[:3]
    result = IntervalFamilyCertifier().certify(small_model, inputs, semantics, box)
    exact = VectorizedEmulator().run(small_model, inputs, semantics).final_logits
    assert np.all(exact >= result.target_logit_lower)
    assert np.all(exact <= result.target_logit_upper)


def test_linear_interval_zero_weights_preserves_batch_shape() -> None:
    lower = np.zeros((3, 4))
    upper = np.ones((3, 4))
    weights = np.zeros((4, 5))

    result_lower, result_upper = _linear_interval(lower, upper, weights)

    assert result_lower.shape == (3, 5)
    assert result_upper.shape == (3, 5)
    np.testing.assert_array_equal(result_lower, 0.0)
    np.testing.assert_array_equal(result_upper, 0.0)


def test_discrete_members_are_certified_before_family_logit_merge() -> None:
    model = DenseRecurrentSNN(
        input_weights=np.asarray([[1.0]]),
        recurrent_weights=np.asarray([[0.0]]),
        output_weights=np.asarray([[1.0, 0.9]]),
        bias=np.asarray([0.0]),
        threshold=np.asarray([1.0]),
        tau_mem=np.asarray([2.0]),
        reset_value=np.asarray([0.0]),
    )
    inputs = np.full((1, 2, 1), 3.0)
    reference = ExecutionSemantics()
    box = SemanticsBox(
        base=reference,
        timestep_bounds=(1.0, 1.0),
        integration_rules=(IntegrationRule.FORWARD_EULER,),
        threshold_timings=(
            ThresholdTiming.POST_INTEGRATION,
            ThresholdTiming.PRE_INTEGRATION,
        ),
        reset_rules=(ResetRule.SUBTRACTIVE,),
        synaptic_delays=(0,),
        output_delays=(0,),
    )

    result = IntervalFamilyCertifier().certify(model, inputs, reference, box)

    assert result.certified.tolist() == [True]
    assert result.target_logit_lower[0, 0] == 1.0
    assert result.target_logit_upper[0, 1] == 1.8


def test_partitioned_box_is_a_closed_cover_and_never_loses_certificates(
    small_model, event_batch
) -> None:
    reference = ExecutionSemantics()
    box = SemanticsBox(
        base=reference,
        timestep_bounds=(0.9, 1.1),
        threshold_scale_bounds=(0.9, 1.1),
        integration_rules=(IntegrationRule.FORWARD_EULER,),
        reset_rules=(ResetRule.SUBTRACTIVE,),
        synaptic_delays=(0,),
        output_delays=(0,),
    )
    boxes = partition_semantics_box(box, 2, 4)
    assert len(boxes) == 8
    assert min(item.timestep_bounds[0] for item in boxes) == 0.9
    assert max(item.timestep_bounds[1] for item in boxes) == 1.1
    assert min(item.threshold_scale_bounds[0] for item in boxes) == 0.9
    assert max(item.threshold_scale_bounds[1] for item in boxes) == 1.1

    inputs = event_batch[:4]
    coarse = IntervalFamilyCertifier().certify(small_model, inputs, reference, box)
    partitioned = IntervalFamilyCertifier().certify_partitioned(
        small_model,
        inputs,
        reference,
        box,
        timestep_partitions=2,
        threshold_partitions=4,
    )

    assert np.all(~coarse.certified | partitioned.certified)
