from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


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
            raise ValueError(f"duplicate or empty method root: {identity!r}")
        path = Path(raw_path)
        roots[alias] = (
            report_method,
            path if path.is_absolute() else repository_root / path,
        )
    if len(roots) < 2:
        raise ValueError("at least two method roots are required")
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-root", action="append", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--conditions", nargs="+", default=DEFAULT_CONDITIONS
    )
    parser.add_argument("--result-stem", required=True)
    parser.add_argument("--result-dir", default="results/shd_v1")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    method_roots = _method_roots(args.method_root, repository_root)
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for condition in args.conditions:
        for seed in args.seeds:
            for method, (report_method, artifact_root) in method_roots.items():
                report_path = (
                    artifact_root
                    / f"seed_{seed}"
                    / condition
                    / report_method
                    / "repair_report.json"
                )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (
                    report["method"] != report_method
                    or report["condition"] != condition
                ):
                    raise ValueError(f"cell identity mismatch: {report_path}")
                if not report["calibration_audit_disjoint"]:
                    raise ValueError(f"calibration/audit overlap: {report_path}")
                if report["test_labels_used_for_selection"]:
                    raise ValueError(f"test-label selection leakage: {report_path}")
                selected = report.get("selected", {})
                calibration_disagreements = selected.get(
                    "best_calibration_disagreements"
                )
                if calibration_disagreements is None:
                    raise ValueError(
                        f"missing calibration disagreement count: {report_path}"
                    )
                calibration_rate = float(calibration_disagreements) / int(
                    report["calibration_samples"]
                )
                audit_rate = float(report["after"]["audit_disagreement_rate"])
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "method": method,
                        "calibration_disagreement_rate": calibration_rate,
                        "audit_disagreement_rate": audit_rate,
                        "generalization_gap": audit_rate - calibration_rate,
                        "accuracy_recovery_fraction": float(
                            report["accuracy_recovery_fraction"]
                        ),
                        "certificate_upper_bound": float(
                            report["after"]["certificate_upper_bound"]
                        ),
                        "accuracy_loss": float(report["after"]["accuracy_loss"]),
                        "elapsed_seconds": float(report["elapsed_seconds"]),
                    }
                )
                hashes[f"{condition}__seed_{seed}__{method}"] = sha256_file(
                    report_path
                )

    by_condition: list[dict[str, object]] = []
    for condition in args.conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        per_method: list[dict[str, object]] = []
        method_rows: dict[str, list[dict[str, object]]] = {}
        for method in method_roots:
            selected = [row for row in condition_rows if row["method"] == method]
            selected.sort(key=lambda row: int(row["seed"]))
            method_rows[method] = selected
            per_method.append(
                {
                    "method": method,
                    "calibration_disagreement_rate": _stats(
                        [float(row["calibration_disagreement_rate"]) for row in selected]
                    ),
                    "audit_disagreement_rate": _stats(
                        [float(row["audit_disagreement_rate"]) for row in selected]
                    ),
                    "generalization_gap": _stats(
                        [float(row["generalization_gap"]) for row in selected]
                    ),
                    "accuracy_recovery_fraction": _stats(
                        [float(row["accuracy_recovery_fraction"]) for row in selected]
                    ),
                    "certificate_upper_bound": _stats(
                        [float(row["certificate_upper_bound"]) for row in selected]
                    ),
                    "cells_recovering_at_least_70_percent": int(
                        sum(
                            float(row["accuracy_recovery_fraction"]) >= 0.70
                            for row in selected
                        )
                    ),
                }
            )
        paired: dict[str, object] = {}
        for left, right in combinations(method_roots, 2):
            left_rows = method_rows[left]
            right_rows = method_rows[right]
            paired[f"{left}_minus_{right}"] = {
                "accuracy_recovery_fraction": _stats(
                    [
                        float(a["accuracy_recovery_fraction"])
                        - float(b["accuracy_recovery_fraction"])
                        for a, b in zip(left_rows, right_rows, strict=True)
                    ]
                ),
                "audit_disagreement_rate": _stats(
                    [
                        float(a["audit_disagreement_rate"])
                        - float(b["audit_disagreement_rate"])
                        for a, b in zip(left_rows, right_rows, strict=True)
                    ]
                ),
            }
        by_condition.append(
            {
                "condition": condition,
                "per_method": per_method,
                "paired_differences": paired,
            }
        )

    result = {
        "schema_version": "RepairObjectiveDiagnostic/v1",
        "status": "matched label-free repair diagnostic on untouched audit splits",
        "seeds": list(args.seeds),
        "conditions": list(args.conditions),
        "method_roots": {
            method: {
                "report_method": report_method,
                "path": str(path.relative_to(repository_root)),
            }
            for method, (report_method, path) in method_roots.items()
        },
        "by_condition": by_condition,
        "input_report_hashes": hashes,
        "code_revision": code_revision(repository_root),
    }
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = repository_root / result_dir
    output_path = result_dir / f"{args.result_stem}.json"
    write_json_immutable(output_path, result)
    print(json.dumps(by_condition, indent=2))


if __name__ == "__main__":
    main()
