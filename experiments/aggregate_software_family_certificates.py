from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable
from pines.statistics import clopper_pearson_upper


SEEDS = (1701, 2718, 3141, 5772, 8119)
BUDGETS = (0.01, 0.02, 0.05)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _finite_shd(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = []
    hashes = {}
    for seed in SEEDS:
        path = (
            root
            / "artifacts"
            / "shd_v2_memberwise_static_family"
            / f"seed_{seed}"
            / "static_family_report.json"
        )
        report = _load(path)
        family = next(row for row in report["rows"] if row["family"] == "full")
        samples = int(family["samples"])
        certified = int(family["exact_family_agreement_inputs"])
        rows.append(
            {
                "task": "SHD",
                "family": "finite 16",
                "seed": seed,
                "samples": samples,
                "certified": certified,
                "falsified": samples - certified,
                "unknown": 0,
                "seconds_per_input": float(family["exact_seconds"]) / samples,
            }
        )
        hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    return rows, hashes


def _finite_reports(
    root: Path,
    *,
    task: str,
    family: str,
    artifact_root: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = []
    hashes = {}
    for seed in SEEDS:
        path = root / artifact_root / f"seed_{seed}" / "finite_family_report.json"
        report = _load(path)
        samples = int(report["audit_samples"])
        certified = int(report["certified_inputs"])
        rows.append(
            {
                "task": task,
                "family": family,
                "seed": seed,
                "samples": samples,
                "certified": certified,
                "falsified": int(report["falsified_inputs"]),
                "unknown": int(report["unknown_inputs"]),
                "seconds_per_input": float(report["elapsed_seconds"]) / samples,
            }
        )
        hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    return rows, hashes


def _continuous_shd(
    root: Path, summary_path: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    summary = _load(summary_path)
    rows = []
    for seed_row in summary["per_seed"]:
        samples = int(seed_row["sample_count"])
        certified = int(seed_row["certified_input_count"])
        falsified = int(seed_row["counterexample_union_uncertified_count"])
        unknown = samples - certified - falsified
        if unknown < 0:
            raise ValueError("continuous-family partition has negative unknown count")
        rows.append(
            {
                "task": "SHD",
                "family": "reset continuous",
                "seed": int(seed_row["seed"]),
                "samples": samples,
                "certified": certified,
                "falsified": falsified,
                "unknown": unknown,
                "seconds_per_input": float(seed_row["median_seconds"]),
            }
        )
    if tuple(sorted(row["seed"] for row in rows)) != SEEDS:
        raise ValueError("continuous SHD summary does not contain the five fixed seeds")
    return rows, {
        str(summary_path.relative_to(root)).replace("\\", "/"): sha256_file(
            summary_path
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shd-continuous-summary",
        default=(
            "results/shd_v1/"
            "hybrid_family_full_audit_soundness_corrected_v1_summary.json"
        ),
    )
    parser.add_argument(
        "--output", default="results/family_certificates/software_family_summary.json"
    )
    parser.add_argument(
        "--rows", default="results/family_certificates/software_family_rows.csv"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    all_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for loader in (
        lambda: _finite_shd(root),
        lambda: _continuous_shd(root, root / args.shd_continuous_summary),
        lambda: _finite_reports(
            root,
            task="DVS Gesture",
            family="finite 16",
            artifact_root="artifacts/dvs_gesture_v6_finite_family",
        ),
        lambda: _finite_reports(
            root,
            task="N-MNIST",
            family="finite 16",
            artifact_root="artifacts/nmnist_v3_full_audit_finite_family",
        ),
    ):
        rows, hashes = loader()
        all_rows.extend(rows)
        input_hashes.update(hashes)

    comparisons = len(all_rows)
    alpha = 0.05 / comparisons
    for row in all_rows:
        failures = int(row["falsified"]) + int(row["unknown"])
        row["certified_fraction"] = int(row["certified"]) / int(row["samples"])
        row["falsified_fraction"] = int(row["falsified"]) / int(row["samples"])
        row["unknown_fraction"] = int(row["unknown"]) / int(row["samples"])
        row["semantic_upper_bound"] = clopper_pearson_upper(
            failures, int(row["samples"]), alpha
        )
        for budget in BUDGETS:
            row[f"accept_{int(100 * budget)}pp"] = (
                float(row["semantic_upper_bound"]) <= budget
            )

    grouped: list[dict[str, Any]] = []
    keys = []
    for row in all_rows:
        key = (str(row["task"]), str(row["family"]))
        if key not in keys:
            keys.append(key)
    for task, family in keys:
        selected = [
            row
            for row in all_rows
            if row["task"] == task and row["family"] == family
        ]
        grouped.append(
            {
                "task": task,
                "family": family,
                "samples_per_seed": [int(row["samples"]) for row in selected],
                "certified_fraction": _stats(
                    [float(row["certified_fraction"]) for row in selected]
                ),
                "falsified_fraction": _stats(
                    [float(row["falsified_fraction"]) for row in selected]
                ),
                "unknown_fraction": _stats(
                    [float(row["unknown_fraction"]) for row in selected]
                ),
                "semantic_upper_bound": _stats(
                    [float(row["semantic_upper_bound"]) for row in selected]
                ),
                "all_seed_verdicts": {
                    f"{int(100 * budget)}pp": all(
                        bool(row[f"accept_{int(100 * budget)}pp"])
                        for row in selected
                    )
                    for budget in BUDGETS
                },
                "seconds_per_input": _stats(
                    [float(row["seconds_per_input"]) for row in selected]
                ),
            }
        )

    rows_path = root / args.rows
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "schema_version": "SoftwareFamilyCertificateAggregate/v1",
        "status": (
            "five-seed conditional software certificates; no physical "
            "conformance term"
        ),
        "confidence": 0.95,
        "simultaneous_method": "Bonferroni over every task-family-seed cell",
        "simultaneous_comparisons": comparisons,
        "per_cell_alpha": alpha,
        "budgets": list(BUDGETS),
        "rows": grouped,
        "input_report_hashes": input_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
        "interpretation": (
            "Finite families are exactly enumerated. The continuous SHD row "
            "counts every unresolved input as a possible disagreement, so its "
            "population bound remains sound even when the analyzer is incomplete."
        ),
    }
    write_json_immutable(root / args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
