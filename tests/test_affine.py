from __future__ import annotations

from dataclasses import replace

import numpy as np

from transportcert.abstract import IntervalFamilyCertifier, SemanticsBox
from transportcert.affine import AffineGuardFamilyCertifier, _HybridAffine
from transportcert.emulator import VectorizedEmulator
from transportcert.models import DenseRecurrentSNN
from transportcert.semantics import ExecutionSemantics, IntegrationRule, ResetRule


def test_hybrid_affine_product_residual_contains_shared_samples() -> None:
    left = _HybridAffine(
        center=np.asarray([[1.2]]),
        generators=np.asarray([[[0.3, -0.2]]]),
        radius=np.asarray([[0.1]]),
    )
    right = _HybridAffine(
        center=np.asarray([[-0.7]]),
        generators=np.asarray([[[0.4, 0.1]]]),
        radius=np.asarray([[0.05]]),
    )
    product = left.product(right)
    rng = np.random.default_rng(7)
    for _ in range(1000):
        epsilon = rng.uniform(-1.0, 1.0, size=2)
        left_error = rng.uniform(-left.radius, left.radius)
        right_error = rng.uniform(-right.radius, right.radius)
        left_value = left.center + left.generators @ epsilon + left_error
        right_value = right.center + right.generators @ epsilon + right_error
        affine_value = product.center + product.generators @ epsilon
        assert np.all(np.abs(left_value * right_value - affine_value) <= product.radius)


def test_affine_guard_domain_preserves_shared_output_spikes() -> None:
    model = DenseRecurrentSNN(
        input_weights=np.asarray([[2.0, 1.0]]),
        recurrent_weights=np.zeros((2, 2)),
        output_weights=np.asarray([[1.0, 0.9], [1.0, 1.0]]),
        bias=np.zeros(2),
        threshold=np.ones(2),
        tau_mem=np.ones(2),
        reset_value=np.zeros(2),
    )
    inputs = np.ones((1, 1, 1))
    reference = ExecutionSemantics()
    box = SemanticsBox(
        base=reference,
        timestep_bounds=(0.9, 1.1),
        integration_rules=(reference.integration_rule,),
        threshold_timings=(reference.threshold_timing,),
        reset_rules=(reference.reset_rule,),
        synaptic_delays=(0,),
        output_delays=(0,),
    )
    interval = IntervalFamilyCertifier().certify(model, inputs, reference, box)
    affine = AffineGuardFamilyCertifier().certify(model, inputs, reference, box)
    assert not interval.certified[0]
    assert affine.certified[0]


def test_affine_guard_bounds_contain_sampled_recurrent_executions(
    small_model, event_batch
) -> None:
    reference = ExecutionSemantics()
    target = replace(reference, reset_rule=ResetRule.TO_VALUE)
    box = SemanticsBox(
        base=target,
        timestep_bounds=(0.9, 1.1),
        threshold_scale_bounds=(0.9, 1.1),
        integration_rules=(IntegrationRule.FORWARD_EULER,),
        reset_rules=(ResetRule.TO_VALUE,),
        synaptic_delays=(0,),
        output_delays=(0,),
    )
    inputs = event_batch[:4]
    result = AffineGuardFamilyCertifier().certify(
        small_model, inputs, reference, box
    )
    assert result.split_axis_scores.shape == (len(inputs), 2)
    assert np.all(result.split_axis_scores >= 0.0)
    rows = np.arange(len(inputs))
    for timestep in (0.9, 1.0, 1.1):
        semantics = replace(target, timestep=timestep)
        for threshold_scale in (0.9, 1.0, 1.1):
            candidate = small_model.with_parameters(
                threshold=small_model.threshold * threshold_scale
            )
            logits = VectorizedEmulator().run(candidate, inputs, semantics).final_logits
            margins = logits[rows, result.reference_predictions, None] - logits
            assert np.all(margins >= result.target_margin_lower - 1e-12)
            assert np.all(margins <= result.target_margin_upper + 1e-12)
