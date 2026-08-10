from __future__ import annotations

import numpy as np
import torch

from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd_repair import (
    _checkpoint_selection_key,
    build_repairable_srnn,
    build_supervised_target_srnn,
    _export_supervised,
)
from pines.torch_emulator import TorchEmulator


def test_checkpoint_selection_preserves_method_specific_ordering() -> None:
    guard_low_disagreement = _checkpoint_selection_key(
        "guard_margin", disagreement=2, selection_score=20.0, margin_deficit=0.0
    )
    guard_low_combined_score = _checkpoint_selection_key(
        "guard_margin", disagreement=3, selection_score=3.0, margin_deficit=0.0
    )
    assert guard_low_combined_score < guard_low_disagreement

    margin_low_disagreement = _checkpoint_selection_key(
        "margin_distilled", disagreement=2, selection_score=20.0, margin_deficit=1.0
    )
    margin_low_deficit = _checkpoint_selection_key(
        "margin_distilled", disagreement=3, selection_score=3.0, margin_deficit=0.0
    )
    assert margin_low_disagreement < margin_low_deficit


def test_repairable_srnn_executes_target_semantics(small_model) -> None:
    target = primary_semantic_conditions()["reset_to_value__delay_1"]
    module = build_repairable_srnn(small_model, target)
    logits, states, spikes, logits_over_time = module(torch.zeros((3, 5, 2)))
    assert logits.shape == (3, 2)
    assert states.shape == (3, 5, 2)
    assert spikes.shape == (3, 5, 2)
    assert logits_over_time.shape == (3, 5, 2)
    assert module.last_guard_trace is not None
    assert module.last_guard_trace.shape == (3, 5, 2)


def test_repairable_srnn_matches_cast_faithful_executor(small_model) -> None:
    rng = np.random.default_rng(19)
    events = rng.normal(size=(4, 7, small_model.input_size))
    for condition in ("reference", "reset_to_value", "floor_rounding_saturation"):
        semantics = primary_semantic_conditions()[condition]
        module = build_repairable_srnn(small_model, semantics)
        logits, states, spikes, logits_over_time = module(torch.as_tensor(events))
        expected = TorchEmulator(dtype=torch.float64).run(
            small_model, events, semantics
        )
        torch.testing.assert_close(states, expected.membrane, rtol=0.0, atol=0.0)
        torch.testing.assert_close(spikes, expected.spikes, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            logits_over_time, expected.logits_over_time, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(logits, expected.final_logits, rtol=0.0, atol=0.0)


def test_repairable_srnn_executes_positive_family_scales(small_model) -> None:
    target = primary_semantic_conditions()["reset_to_value"]
    module = build_repairable_srnn(small_model, target)
    events = torch.ones((2, 5, 2))
    center = module(events)[0]
    corner = module(events, timestep_scale=1.01, threshold_scale=0.99)[0]
    assert center.shape == corner.shape == (2, 2)


def test_repairable_srnn_rejects_nonpositive_family_scales(small_model) -> None:
    target = primary_semantic_conditions()["reset_to_value"]
    module = build_repairable_srnn(small_model, target)
    events = torch.zeros((1, 2, 2))
    try:
        module(events, timestep_scale=0.0)
    except ValueError as error:
        assert "scales" in str(error)
    else:
        raise AssertionError("nonpositive family scale was accepted")


def test_supervised_target_srnn_source_initialization_round_trips(small_model) -> None:
    target = primary_semantic_conditions()["fixed_q8_weights_q16_state"]
    module = build_supervised_target_srnn(
        small_model, target, initialization="source"
    )
    exported = _export_supervised(module, small_model, "round-trip")
    np.testing.assert_allclose(exported.input_weights, small_model.input_weights)
    np.testing.assert_allclose(
        exported.recurrent_weights, small_model.recurrent_weights
    )
    logits, states, spikes, logits_over_time = module(torch.zeros((3, 5, 2)))
    assert logits.shape == (3, 2)
    assert states.shape == (3, 5, 2)
    assert spikes.shape == (3, 5, 2)
    assert logits_over_time.shape == (3, 5, 2)


def test_supervised_target_srnn_matches_cast_faithful_executor(small_model) -> None:
    events = np.random.default_rng(23).normal(
        size=(3, 6, small_model.input_size)
    )
    for condition in ("reference", "reset_to_value", "floor_rounding_saturation"):
        semantics = primary_semantic_conditions()[condition]
        module = build_supervised_target_srnn(
            small_model, semantics, initialization="source"
        )
        logits, states, spikes, logits_over_time = module(torch.as_tensor(events))
        expected = TorchEmulator(dtype=torch.float64).run(
            small_model, events, semantics
        )
        torch.testing.assert_close(states, expected.membrane, rtol=0.0, atol=0.0)
        torch.testing.assert_close(spikes, expected.spikes, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            logits_over_time, expected.logits_over_time, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(logits, expected.final_logits, rtol=0.0, atol=0.0)


def test_supervised_target_srnn_rejects_unknown_initialization(small_model) -> None:
    target = primary_semantic_conditions()["reference"]
    try:
        build_supervised_target_srnn(small_model, target, initialization="unknown")
    except ValueError as error:
        assert "initialization" in str(error)
    else:
        raise AssertionError("unknown initialization was accepted")
