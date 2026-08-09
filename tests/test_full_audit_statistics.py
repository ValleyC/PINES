from __future__ import annotations

import numpy as np

from experiments.aggregate_shd_hybrid_full_audit import (
    _input_cluster_bootstrap_interval,
    _training_seed_t_interval,
)


def test_input_cluster_bootstrap_is_deterministic_and_conditions_on_clusters() -> None:
    values = np.asarray([0.0, 0.2, 0.6, 1.0], dtype=np.float64)
    first = _input_cluster_bootstrap_interval(
        values, confidence=0.95, repetitions=2_000, seed=20260808
    )
    second = _input_cluster_bootstrap_interval(
        values, confidence=0.95, repetitions=2_000, seed=20260808
    )
    assert first == second
    assert 0.0 <= first[0] <= np.mean(values) <= first[1] <= 1.0


def test_training_seed_interval_uses_five_seed_means() -> None:
    values = np.asarray([0.40, 0.27, 0.33, 0.35, 0.33], dtype=np.float64)
    lower, upper = _training_seed_t_interval(values, confidence=0.95)
    assert lower < np.mean(values) < upper
    assert upper - lower > 0.08
