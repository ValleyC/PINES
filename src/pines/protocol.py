from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .artifacts import config_description


@dataclass(frozen=True)
class FrozenSplit:
    name: str
    sample_ids: tuple[str, ...]

    @property
    def split_description(self) -> str:
        return config_description({"name": self.name, "sample_ids": self.sample_ids})


def deterministic_partition(
    sample_ids: Iterable[str],
    fractions: Mapping[str, float],
    salt: str,
) -> dict[str, FrozenSplit]:
    """Create a reproducible partition after sorting the public sample IDs."""

    if not salt:
        raise ValueError("partition salt must be non-empty")
    if not fractions or any(value <= 0 for value in fractions.values()):
        raise ValueError("partition fractions must be positive")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-12:
        raise ValueError("partition fractions must sum to one")
    ids = tuple(str(sample_id) for sample_id in sample_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("sample IDs must be unique")
    names = tuple(fractions)
    cumulative = []
    running = 0.0
    for name in names:
        running += fractions[name]
        cumulative.append(running)
    assigned: dict[str, list[str]] = {name: [] for name in names}
    seed = sum((index + 1) * ord(character) for index, character in enumerate(salt))
    rng = np.random.default_rng(seed)
    for sample_id in sorted(ids):
        value = float(rng.random())
        for name, boundary in zip(names, cumulative):
            if value < boundary:
                assigned[name].append(sample_id)
                break
    return {
        name: FrozenSplit(name, tuple(sorted(values)))
        for name, values in assigned.items()
    }


def assert_disjoint_splits(splits: Iterable[FrozenSplit]) -> None:
    seen: set[str] = set()
    for split in splits:
        overlap = seen.intersection(split.sample_ids)
        if overlap:
            raise ValueError(f"split {split.name} overlaps prior splits: {sorted(overlap)[:3]}")
        seen.update(split.sample_ids)
