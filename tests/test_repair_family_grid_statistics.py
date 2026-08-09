from __future__ import annotations

import numpy as np

from experiments.aggregate_shd_repair_family_grid import (
    _cluster_bootstrap_interval,
    _training_seed_t_interval,
    _unpack_bool_hex,
)
from experiments.run_shd_repair_family_grid import _packed_bool_hex


def test_repair_family_identity_bitpack_round_trip() -> None:
    values = np.asarray(
        [True, False, True, True, False, False, True, False, True], dtype=bool
    )
    encoded = _packed_bool_hex(values)
    decoded = _unpack_bool_hex(encoded, len(values))
    np.testing.assert_array_equal(decoded, values)


def test_repair_family_cluster_bootstrap_is_deterministic() -> None:
    cluster_means = np.asarray([0.0, 0.2, 0.4, 0.8, 1.0], dtype=np.float64)
    first = _cluster_bootstrap_interval(cluster_means)
    second = _cluster_bootstrap_interval(cluster_means)
    assert first == second
    assert first[0] < float(np.mean(cluster_means)) < first[1]


def test_repair_family_seed_interval_can_represent_signed_difference() -> None:
    differences = np.asarray([-0.03, -0.01, 0.0, 0.01, 0.02], dtype=np.float64)
    interval = _training_seed_t_interval(
        differences, lower_bound=None, upper_bound=None
    )
    assert interval[0] < 0 < interval[1]
