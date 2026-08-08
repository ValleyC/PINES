from __future__ import annotations

import numpy as np
import torch

from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd_repair import (
    build_repairable_srnn,
    build_supervised_target_srnn,
    _export_supervised,
)


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


def test_supervised_target_srnn_rejects_unknown_initialization(small_model) -> None:
    target = primary_semantic_conditions()["reference"]
    try:
        build_supervised_target_srnn(small_model, target, initialization="unknown")
    except ValueError as error:
        assert "initialization" in str(error)
    else:
        raise AssertionError("unknown initialization was accepted")
