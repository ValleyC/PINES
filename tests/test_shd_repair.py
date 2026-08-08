from __future__ import annotations

import torch

from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd_repair import build_repairable_srnn


def test_repairable_srnn_executes_target_semantics(small_model) -> None:
    target = primary_semantic_conditions()["reset_to_value__delay_1"]
    module = build_repairable_srnn(small_model, target)
    logits, states, spikes, logits_over_time = module(torch.zeros((3, 5, 2)))
    assert logits.shape == (3, 2)
    assert states.shape == (3, 5, 2)
    assert spikes.shape == (3, 5, 2)
    assert logits_over_time.shape == (3, 5, 2)
