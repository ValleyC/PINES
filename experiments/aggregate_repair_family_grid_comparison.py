from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, file_reference, write_json


DEFAULT_SEEDS = (1701, 2718, 3141, 5772, 8119)
DEFAULT_CONDITIONS = ("reset_to_value", "floor_rounding_saturation")


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _method_roots(
    values: list[str], repository_root: Path
) -> dict[str, tuple[str, Path]]:
    roots: dict[str, tuple[str, Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("method roots must use [ALIAS@]METHOD=PATH")
        identity, raw_path = value.split("=", 1)
        if "@" in identity:
            alias, report_method = identity.split("@", 1)
        else:
            alias = report_method = identity
        if not alias or not report_method or alias in roots:
            raise ValueError(f"invalid or duplicate method identity: {identity!r}")
        path = Path(raw_path)
        roots[alias] = (
            report_method,
            path if path.is_absolute() else repository_root / path,
        )
    if len(roots) < 2:
        raise ValueError("at least two method roots are required")
    return roots


def _unpack(payload: str, count: int) -> np.ndarray:
    packed = np.frombuffer(bytes.fromhex(payload), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little")[:count].astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-root", action="append", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--result-stem", required=True)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    roots = _method_roots(args.method_root, repository_root)
    input_references: dict[str, str] = {}
    condition_results: list[dict[str, object]] = []
    for condition in args.conditions:
        vectors: dict[str, list[np.ndarray]] = {alias: [] for alias in roots}
        fractions: dict[str, list[float]] = {alias: [] for alias in roots}
        design: tuple[int, int, float] | None = None
        per_seed_index_reference: dict[int, str] = {}
        for seed in args.seeds:
            for alias, (report_method, artifact_root) in roots.items():
                path = artifact_root / f"seed_{seed}" / f"{condition}_family_grid.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                if report["condition"] != condition or int(report["seed"]) != seed:
                    raise ValueError(f"cell identity mismatch: {path}")
                current_design = (
                    int(report["sample_count"]),
                    int(report["grid_resolution"]),
                    float(report["relative_radius"]),
                )
                if design is None:
                    design = current_design
                elif current_design != design:
                    raise ValueError(f"grid design mismatch: {path}")
                index_reference = report["selected_indices_reference"]
                if seed in per_seed_index_reference and per_seed_index_reference[seed] != index_reference:
                    raise ValueError(f"selected inputs differ across methods: {path}")
                per_seed_index_reference[seed] = index_reference
                row = next(
                    item for item in report["rows"] if item["method"] == report_method
                )
                vector = _unpack(
                    row["grid_family_identity_packed_hex"],
                    int(report["sample_count"]),
                )
                if int(np.count_nonzero(vector)) != int(
                    row["grid_family_identity_inputs"]
                ):
                    raise ValueError(f"packed identity mismatch: {path}")
                vectors[alias].append(vector)
                fractions[alias].append(float(np.mean(vector)))
                input_references[f"{condition}__seed_{seed}__{alias}"] = file_reference(path)
        assert design is not None
        paired: dict[str, object] = {}
        for left, right in combinations(roots, 2):
            per_seed = [
                float(np.mean(a) - np.mean(b))
                for a, b in zip(vectors[left], vectors[right], strict=True)
            ]
            paired[f"{left}_minus_{right}"] = {
                "identity_fraction": _stats(per_seed),
                "paired_input_difference": _stats(
                    [
                        float(value)
                        for a, b in zip(vectors[left], vectors[right], strict=True)
                        for value in a.astype(np.int8) - b.astype(np.int8)
                    ]
                ),
            }
        condition_results.append(
            {
                "condition": condition,
                "sample_count_per_seed": design[0],
                "grid_resolution": design[1],
                "relative_radius": design[2],
                "per_method": [
                    {
                        "method": alias,
                        "grid_family_identity_fraction": _stats(fractions[alias]),
                    }
                    for alias in roots
                ],
                "paired_differences": paired,
            }
        )

    result = {
        "schema_version": "RepairFamilyGridComparison/v1",
        "status": "finite-grid diagnostic, not a continuous-family certificate",
        "seeds": list(args.seeds),
        "conditions": list(args.conditions),
        "methods": {
            alias: {
                "report_method": report_method,
                "path": str(path.relative_to(repository_root)),
            }
            for alias, (report_method, path) in roots.items()
        },
        "condition_results": condition_results,
        "input_report_references": input_references,
        "code_revision": code_revision(repository_root),
    }
    output_path = (
        repository_root
        / "results"
        / "shd_v1"
        / f"{args.result_stem}.json"
    )
    write_json(output_path, result)
    print(json.dumps(condition_results, indent=2))


if __name__ == "__main__":
    main()
