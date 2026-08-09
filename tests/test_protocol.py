from __future__ import annotations

import pytest

from pines.protocol import FrozenSplit, assert_disjoint_splits, deterministic_partition


def test_hash_partition_is_order_independent_and_disjoint() -> None:
    ids = [f"sample-{index}" for index in range(1000)]
    fractions = {"train": 0.8, "repair": 0.1, "audit": 0.1}
    forward = deterministic_partition(ids, fractions, "pines-v1")
    reverse = deterministic_partition(reversed(ids), fractions, "pines-v1")
    assert forward == reverse
    assert_disjoint_splits(forward.values())
    assert sum(len(split.sample_ids) for split in forward.values()) == len(ids)


def test_split_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        assert_disjoint_splits(
            (FrozenSplit("calibration", ("x",)), FrozenSplit("audit", ("x",)))
        )

