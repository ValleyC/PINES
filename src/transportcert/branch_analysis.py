from __future__ import annotations

import hashlib

import numpy as np


def packed_trace_hashes(spikes: np.ndarray) -> np.ndarray:
    """Return one stable binary trace digest per batch element."""

    values = np.asarray(spikes)
    if values.ndim != 3:
        raise ValueError("spikes must have shape [batch, time, neurons]")
    binary = values != 0
    packed = np.packbits(binary.reshape(binary.shape[0], -1), axis=1)
    return np.asarray(
        [hashlib.sha256(row.tobytes()).digest() for row in packed], dtype="|S32"
    )


def nested_grid_flat_indices(
    max_resolution: int, resolution: int
) -> np.ndarray:
    """Indices selecting a nested square grid from a larger square grid."""

    if resolution < 2 or max_resolution < resolution:
        raise ValueError("grid resolutions must satisfy 2 <= resolution <= maximum")
    if (max_resolution - 1) % (resolution - 1) != 0:
        raise ValueError("requested grid is not nested in the maximum grid")
    stride = (max_resolution - 1) // (resolution - 1)
    coordinates = np.arange(0, max_resolution, stride, dtype=np.int64)
    return np.asarray(
        [row * max_resolution + column for row in coordinates for column in coordinates],
        dtype=np.int64,
    )


def summarize_branch_grid(
    trace_hashes: np.ndarray,
    predictions: np.ndarray,
    *,
    max_resolution: int,
    resolution: int,
) -> dict[str, float | int]:
    """Summarize sampled trajectory and decision multiplicity per input."""

    traces = np.asarray(trace_hashes)
    labels = np.asarray(predictions)
    expected_points = max_resolution * max_resolution
    if traces.ndim != 2 or labels.shape != traces.shape:
        raise ValueError("trace hashes and predictions must share [grid point, input]")
    if traces.shape[0] != expected_points:
        raise ValueError("first dimension does not match the maximum square grid")
    selected = nested_grid_flat_indices(max_resolution, resolution)
    selected_traces = traces[selected]
    selected_labels = labels[selected]
    center = (max_resolution // 2) * max_resolution + max_resolution // 2
    center_traces = traces[center]
    center_labels = labels[center]
    unique_trace_counts = np.asarray(
        [len(np.unique(selected_traces[:, index])) for index in range(traces.shape[1])],
        dtype=np.int64,
    )
    unique_prediction_counts = np.asarray(
        [len(np.unique(selected_labels[:, index])) for index in range(labels.shape[1])],
        dtype=np.int64,
    )
    trace_matches_center = selected_traces == center_traces[None, :]
    prediction_matches_center = selected_labels == center_labels[None, :]
    grid_points = resolution * resolution
    return {
        "resolution": resolution,
        "grid_points": grid_points,
        "inputs": traces.shape[1],
        "mean_unique_spike_traces": float(np.mean(unique_trace_counts)),
        "median_unique_spike_traces": float(np.median(unique_trace_counts)),
        "p90_unique_spike_traces": float(np.quantile(unique_trace_counts, 0.9)),
        "max_unique_spike_traces": int(np.max(unique_trace_counts)),
        "single_spike_trace_fraction": float(np.mean(unique_trace_counts == 1)),
        "all_points_unique_trace_fraction": float(
            np.mean(unique_trace_counts == grid_points)
        ),
        "mean_unique_predictions": float(np.mean(unique_prediction_counts)),
        "max_unique_predictions": int(np.max(unique_prediction_counts)),
        "single_prediction_fraction": float(np.mean(unique_prediction_counts == 1)),
        "all_grid_predictions_match_center_fraction": float(
            np.mean(np.all(prediction_matches_center, axis=0))
        ),
        "trace_grid_pair_disagreement_fraction": float(
            1.0 - np.mean(trace_matches_center)
        ),
        "prediction_grid_pair_disagreement_fraction": float(
            1.0 - np.mean(prediction_matches_center)
        ),
    }


def summarize_family_prediction_grid(
    predictions: np.ndarray,
    reference_predictions: np.ndarray,
    *,
    max_resolution: int,
    resolution: int,
) -> dict[str, float | int | list[float]]:
    """Summarize sampled prediction invariance over members and a nested grid."""

    values = np.asarray(predictions)
    reference = np.asarray(reference_predictions)
    expected_points = max_resolution * max_resolution
    if values.ndim != 3:
        raise ValueError("predictions must have shape [member, grid point, input]")
    if values.shape[1] != expected_points or values.shape[2:] != reference.shape:
        raise ValueError("prediction and reference shapes are inconsistent")
    selected = nested_grid_flat_indices(max_resolution, resolution)
    selected_values = values[:, selected, :]
    matches = selected_values == reference[None, None, :]
    per_member_stable = np.all(matches, axis=1)
    family_stable = np.all(per_member_stable, axis=0)
    flattened = selected_values.reshape(-1, values.shape[2])
    unique_predictions = np.asarray(
        [len(np.unique(flattened[:, index])) for index in range(values.shape[2])],
        dtype=np.int64,
    )
    member_fractions = np.mean(per_member_stable, axis=1)
    return {
        "resolution": resolution,
        "grid_points_per_member": resolution * resolution,
        "member_count": values.shape[0],
        "total_sampled_semantics": values.shape[0] * resolution * resolution,
        "inputs": values.shape[2],
        "full_family_prediction_identity_fraction": float(np.mean(family_stable)),
        "full_family_prediction_identity_inputs": int(np.count_nonzero(family_stable)),
        "mean_unique_predictions": float(np.mean(unique_predictions)),
        "p90_unique_predictions": float(np.quantile(unique_predictions, 0.9)),
        "max_unique_predictions": int(np.max(unique_predictions)),
        "sample_input_pair_disagreement_fraction": float(1.0 - np.mean(matches)),
        "per_member_prediction_identity_fractions": member_fractions.tolist(),
        "mean_member_prediction_identity_fraction": float(np.mean(member_fractions)),
        "minimum_member_prediction_identity_fraction": float(np.min(member_fractions)),
        "maximum_member_prediction_identity_fraction": float(np.max(member_fractions)),
    }
