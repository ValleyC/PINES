from __future__ import annotations

from dataclasses import replace

import numpy as np

from pines.abstract import SemanticsBox
from pines.branch_set import BranchSetMemberCertifier
from pines.emulator import VectorizedEmulator
from pines.semantics import ExecutionSemantics


def _box(semantics: ExecutionSemantics, radius: float) -> SemanticsBox:
    return SemanticsBox(
        base=semantics,
        timestep_bounds=(1.0 - radius, 1.0 + radius),
        threshold_scale_bounds=(1.0 - radius, 1.0 + radius),
        integration_rules=(semantics.integration_rule,),
        threshold_timings=(semantics.threshold_timing,),
        reset_rules=(semantics.reset_rule,),
        synaptic_delays=(0,),
        output_delays=(0,),
    )


def test_branch_set_contains_sampled_predictions(small_model, event_batch) -> None:
    semantics = ExecutionSemantics()
    inputs = event_batch[:1]
    reference = VectorizedEmulator().run(small_model, inputs, semantics).predictions[0]
    result = BranchSetMemberCertifier().certify(
        small_model,
        inputs,
        _box(semantics, 0.05),
        int(reference),
        max_branches=16384,
    )
    assert result.complete
    for timestep in np.linspace(0.95, 1.05, 7):
        for threshold_scale in np.linspace(0.95, 1.05, 7):
            target_semantics = replace(semantics, timestep=float(timestep))
            target_model = small_model.with_parameters(
                threshold=small_model.threshold * threshold_scale
            )
            prediction = int(
                VectorizedEmulator()
                .run(target_model, inputs, target_semantics)
                .predictions[0]
            )
            assert prediction in result.reachable_predictions

    unmerged = BranchSetMemberCertifier().certify(
        small_model,
        inputs,
        _box(semantics, 0.05),
        int(reference),
        max_branches=16384,
        merge_equivalent=False,
    )
    assert unmerged.complete
    assert set(result.reachable_predictions).issuperset(
        unmerged.reachable_predictions
    )


def test_branch_set_rejects_multiple_discrete_members(small_model, event_batch) -> None:
    semantics = ExecutionSemantics()
    box = _box(semantics, 0.01)
    invalid = replace(
        box,
        integration_rules=(
            semantics.integration_rule,
            semantics.integration_rule,
        ),
    )
    try:
        BranchSetMemberCertifier().certify(
            small_model, event_batch[:1], invalid, 0
        )
    except ValueError as error:
        assert "singleton" in str(error)
    else:
        raise AssertionError("expected singleton-axis validation")
