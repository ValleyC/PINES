from __future__ import annotations

import numpy as np
import pytest

from pines.branch_analysis import (
    nested_grid_flat_indices,
    packed_trace_keys,
    summarize_branch_grid,
    summarize_family_prediction_grid,
)


def test_packed_trace_keys_distinguish_binary_traces() -> None:
    spikes = np.zeros((3, 2, 4), dtype=np.float32)
    spikes[1, 0, 0] = 1
    spikes[2, 0, 0] = 1
    descriptions = packed_trace_keys(spikes)
    assert descriptions.shape == (3,)
    assert descriptions[0] != descriptions[1]
    assert descriptions[1] == descriptions[2]
    with pytest.raises(ValueError):
        packed_trace_keys(np.zeros((2, 4)))


def test_nested_grid_indices_are_centered_subgrids() -> None:
    assert nested_grid_flat_indices(5, 3).tolist() == [
        0,
        2,
        4,
        10,
        12,
        14,
        20,
        22,
        24,
    ]
    with pytest.raises(ValueError):
        nested_grid_flat_indices(6, 4)


def test_branch_summary_separates_trace_and_prediction_stability() -> None:
    traces = np.full((25, 2), b"center", dtype="|S32")
    predictions = np.zeros((25, 2), dtype=np.int16)
    traces[0, 1] = b"different"
    predictions[0, 0] = 1
    summary = summarize_branch_grid(
        traces, predictions, max_resolution=5, resolution=5
    )
    assert summary["mean_unique_spike_traces"] == 1.5
    assert summary["single_spike_trace_fraction"] == 0.5
    assert summary["mean_unique_predictions"] == 1.5
    assert summary["single_prediction_fraction"] == 0.5
    assert summary["all_grid_predictions_match_center_fraction"] == 0.5


def test_family_grid_summary_requires_every_member_and_point() -> None:
    predictions = np.zeros((2, 25, 3), dtype=np.int16)
    reference = np.zeros(3, dtype=np.int16)
    predictions[0, 0, 1] = 1
    predictions[1, 24, 2] = 2
    summary = summarize_family_prediction_grid(
        predictions,
        reference,
        max_resolution=5,
        resolution=5,
    )
    assert summary["total_sampled_semantics"] == 50
    assert summary["full_family_prediction_identity_fraction"] == 1 / 3
    assert summary["full_family_prediction_identity_inputs"] == 1
    assert summary["max_unique_predictions"] == 2
    coarse = summarize_family_prediction_grid(
        predictions,
        reference,
        max_resolution=5,
        resolution=3,
    )
    assert coarse["full_family_prediction_identity_fraction"] == 1 / 3
