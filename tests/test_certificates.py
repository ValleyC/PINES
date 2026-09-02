from __future__ import annotations

from dataclasses import replace

import numpy as np

from pines.certificates import (
    CertificateEngine,
    SemanticsFamily,
    exact_identity_mapping,
    exhaustive_state_equivalence,
    exhaustive_trace_equivalence,
)
from pines.models import DenseRecurrentSNN
from pines.semantics import (
    ExecutionSemantics,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)


def test_family_certificate_is_per_input(small_model, event_batch) -> None:
    reference = ExecutionSemantics()
    family = SemanticsFamily(
        (
            replace(reference, reset_rule=ResetRule.TO_VALUE),
            replace(reference, synaptic_delay_steps=1),
        )
    )
    result = CertificateEngine().certify_family(small_model, event_batch, reference, family)
    assert result.certified.shape == (len(event_batch),)
    assert 0 <= result.certified_fraction <= 1
    assert np.all(result.logit_lower <= result.logit_upper)
    assert result.member_predictions.shape == (2, len(event_batch))


def test_conditional_and_physical_reports(small_model, event_batch) -> None:
    reference = ExecutionSemantics()
    family = SemanticsFamily((replace(reference, synaptic_delay_steps=1),))
    engine = CertificateEngine()
    conditional = engine.build_report(
        small_model,
        event_batch,
        reference,
        family,
        delta=0.05,
        decision_budget=0.2,
        dataset_split="audit",
        checkpoint_file=small_model.model_description,
        seed_manifest=[1, 2, 3],
    )
    assert conditional.conditional_on_emulator
    emulator_predictions = engine.certify_family(
        small_model, event_batch, reference, family
    ).member_predictions[0]
    hardware = emulator_predictions.copy()
    hardware[0] = 1 - hardware[0]
    physical = engine.build_report(
        small_model,
        event_batch,
        reference,
        family,
        delta=0.05,
        decision_budget=0.5,
        dataset_split="audit",
        checkpoint_file=small_model.model_description,
        seed_manifest=[1, 2, 3],
        hardware_predictions=hardware,
        firmware_version="f" * 64,
    )
    assert not physical.conditional_on_emulator
    assert physical.conformance_disagreement_count == 1
    assert physical.total_bound >= physical.semantic_bound


def test_only_identity_is_claimed_generally() -> None:
    semantics = ExecutionSemantics()
    assert exact_identity_mapping(semantics, semantics).equivalent
    changed = replace(semantics, reset_rule=ResetRule.TO_VALUE)
    assert not exact_identity_mapping(semantics, changed).equivalent


def test_order_swap_counterexample_not_theorem() -> None:
    model = DenseRecurrentSNN(
        input_weights=np.asarray([[1.0]]),
        recurrent_weights=np.asarray([[0.0]]),
        output_weights=np.asarray([[1.0, -1.0]]),
        bias=np.asarray([0.0]),
        threshold=np.asarray([0.5]),
        tau_mem=np.asarray([1.0]),
        reset_value=np.asarray([0.0]),
    )
    post = ExecutionSemantics()
    pre = ExecutionSemantics(
        threshold_timing=ThresholdTiming.PRE_INTEGRATION,
        update_ordering=UpdateOrdering.THRESHOLD_RESET_INTEGRATE,
    )
    result = exhaustive_trace_equivalence(
        model, post, pre, input_alphabet=(0.0, 1.0), horizon=2
    )
    assert not result.equivalent
    assert result.counterexample is not None


def test_finite_domain_identity_proof(small_model) -> None:
    semantics = ExecutionSemantics()
    result = exhaustive_trace_equivalence(
        small_model.with_parameters(
            input_weights=small_model.input_weights[:1]
        ),
        semantics,
        semantics,
        input_alphabet=(0.0, 1.0),
        horizon=3,
    )
    assert result.equivalent
    assert result.proof_kind == "exhaustive finite-domain proof"


def test_exhaustive_machine_state_checker_finds_order_counterexample() -> None:
    model = DenseRecurrentSNN(
        input_weights=np.asarray([[1.0]]),
        recurrent_weights=np.asarray([[0.0]]),
        output_weights=np.asarray([[1.0, -1.0]]),
        bias=np.asarray([0.0]),
        threshold=np.asarray([0.5]),
        tau_mem=np.asarray([1.0]),
        reset_value=np.asarray([0.0]),
    )
    post = ExecutionSemantics()
    pre = ExecutionSemantics(
        threshold_timing=ThresholdTiming.PRE_INTEGRATION,
        update_ordering=UpdateOrdering.THRESHOLD_RESET_INTEGRATE,
    )
    result = exhaustive_state_equivalence(
        model,
        post,
        pre,
        state_alphabet=(0.0, 0.5, 1.0),
        input_alphabet=(0.0, 1.0),
    )
    assert not result.equivalent
    assert result.proof_kind == "exhaustive finite-state counterexample"
