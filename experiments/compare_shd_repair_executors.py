from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


SEEDS = (1701, 2718, 3141, 5772, 8119)
CONDITIONS = ("reset_to_value", "floor_rounding_saturation")
METHODS = (
    "certificate_directed",
    "logit_only",
    "global_threshold",
    "per_platform_qat",
    "supervised_target_retraining",
)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", default="artifacts/shd_v3_repairs_matched")
    parser.add_argument(
        "--corrected-root", default="artifacts/shd_v75_repairs_cast_faithful_v1"
    )
    parser.add_argument(
        "--output",
        default="results/shd_v1/repair_executor_correction_comparison_v2.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cell_rows = []
    source_hashes = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for condition in CONDITIONS:
        for seed in SEEDS:
            for method in METHODS:
                relative = (
                    Path(f"seed_{seed}")
                    / condition
                    / method
                    / "repair_report.json"
                )
                prior_path = root / args.prior_root / relative
                corrected_path = root / args.corrected_root / relative
                with prior_path.open("r", encoding="utf-8") as handle:
                    prior = json.load(handle)
                with corrected_path.open("r", encoding="utf-8") as handle:
                    corrected = json.load(handle)
                for report, label in ((prior, "prior"), (corrected, "corrected")):
                    if report["condition"] != condition or report["method"] != method:
                        raise ValueError(f"{label} repair cell identity mismatch: {relative}")
                    if int(report["calibration_samples"]) != 800:
                        raise ValueError(f"{label} calibration budget mismatch: {relative}")
                    if int(report["optimization_steps"]) != int(
                        prior["optimization_steps"]
                    ):
                        raise ValueError(f"optimization budget mismatch: {relative}")
                source_hashes[
                    str(prior_path.relative_to(root)).replace("\\", "/")
                ] = sha256_file(prior_path)
                source_hashes[
                    str(corrected_path.relative_to(root)).replace("\\", "/")
                ] = sha256_file(corrected_path)
                row = {
                    "seed": seed,
                    "condition": condition,
                    "method": method,
                    "prior_accuracy_recovery_fraction": float(
                        prior["accuracy_recovery_fraction"]
                    ),
                    "corrected_accuracy_recovery_fraction": float(
                        corrected["accuracy_recovery_fraction"]
                    ),
                    "accuracy_recovery_delta": float(
                        corrected["accuracy_recovery_fraction"]
                        - prior["accuracy_recovery_fraction"]
                    ),
                    "prior_post_repair_bound": float(
                        prior["after"]["certificate_upper_bound"]
                    ),
                    "corrected_post_repair_bound": float(
                        corrected["after"]["certificate_upper_bound"]
                    ),
                    "post_repair_bound_delta": float(
                        corrected["after"]["certificate_upper_bound"]
                        - prior["after"]["certificate_upper_bound"]
                    ),
                    "prior_repaired_model_hash": prior["repaired_model_hash"],
                    "corrected_repaired_model_hash": corrected[
                        "repaired_model_hash"
                    ],
                    "repaired_model_changed": (
                        prior["repaired_model_hash"]
                        != corrected["repaired_model_hash"]
                    ),
                }
                cell_rows.append(row)
                grouped[(condition, method)].append(row)

    group_rows = []
    for (condition, method), rows in sorted(grouped.items()):
        group_rows.append(
            {
                "condition": condition,
                "method": method,
                "accuracy_recovery_delta": _stats(
                    [row["accuracy_recovery_delta"] for row in rows]
                ),
                "post_repair_bound_delta": _stats(
                    [row["post_repair_bound_delta"] for row in rows]
                ),
                "repaired_model_change_count": sum(
                    row["repaired_model_changed"] for row in rows
                ),
            }
        )

    report = {
        "schema_version": "SHDRepairExecutorCorrectionComparison/v2",
        "status": (
            "paired matched-budget comparison across implementation revisions; "
            "not a single-factor executor ablation"
        ),
        "cell_count": len(cell_rows),
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "methods": list(METHODS),
        "cells": cell_rows,
        "per_condition_method": group_rows,
        "source_hashes": source_hashes,
        "code_revision": code_revision(root),
        "interpretation": (
            "Calibration samples, labels, epochs, steps, and named methods are "
            "matched. The corrected run uses cast-faithful semantic targets and "
            "evaluation, but its source revision also adds repair diagnostics and "
            "additional optional methods. Therefore paired deltas establish which "
            "manuscript results survive the current implementation, not a pure "
            "causal executor effect."
        ),
    }
    write_json_immutable(root / args.output, report)


if __name__ == "__main__":
    main()
