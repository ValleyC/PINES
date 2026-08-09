from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", default="artifacts/shd_v1_semantics_final")
    parser.add_argument(
        "--corrected-root", default="artifacts/shd_v74_semantics_cast_faithful_v1"
    )
    parser.add_argument(
        "--output",
        default="results/shd_v1/semantic_executor_correction_comparison_v2.json",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1701, 2718, 3141, 5772, 8119]
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    seed_rows = []
    total_predictions = 0
    total_changes = 0
    changed_keys: set[str] = set()
    total_logit_values = 0
    changed_logit_values = 0
    maximum_logit_absolute_delta = 0.0
    source_hashes = {}
    for seed in args.seeds:
        prior_dir = root / args.prior_root / f"seed_{seed}"
        corrected_dir = root / args.corrected_root / f"seed_{seed}"
        prior_predictions_path = prior_dir / "semantic_predictions.npz"
        corrected_predictions_path = corrected_dir / "semantic_predictions.npz"
        prior_report_path = prior_dir / "semantic_matrix.json"
        corrected_report_path = corrected_dir / "semantic_matrix.json"
        source_hashes[str(prior_predictions_path.relative_to(root)).replace("\\", "/")] = sha256_file(prior_predictions_path)
        source_hashes[str(corrected_predictions_path.relative_to(root)).replace("\\", "/")] = sha256_file(corrected_predictions_path)
        source_hashes[str(prior_report_path.relative_to(root)).replace("\\", "/")] = sha256_file(prior_report_path)
        source_hashes[str(corrected_report_path.relative_to(root)).replace("\\", "/")] = sha256_file(corrected_report_path)
        with np.load(prior_predictions_path, allow_pickle=False) as prior, np.load(
            corrected_predictions_path, allow_pickle=False
        ) as corrected:
            if set(prior.files) != set(corrected.files):
                raise ValueError(f"prediction keys differ for seed {seed}")
            key_rows = []
            logit_rows = []
            for key in sorted(prior.files):
                left = np.asarray(prior[key])
                right = np.asarray(corrected[key])
                if left.shape != right.shape:
                    raise ValueError(f"prediction shape differs for {seed} {key}")
                if key.endswith("_indices"):
                    if not np.array_equal(left, right):
                        raise ValueError(f"split indices differ for {seed} {key}")
                    continue
                changes = int(np.count_nonzero(left != right))
                if key.startswith("pred__"):
                    key_rows.append(
                        {
                            "key": key,
                            "sample_count": int(left.size),
                            "prediction_change_count": changes,
                            "prediction_change_fraction": changes / int(left.size),
                        }
                    )
                    total_predictions += int(left.size)
                    total_changes += changes
                    if changes:
                        changed_keys.add(key)
                elif key.startswith("logits__"):
                    maximum_delta = float(np.max(np.abs(left - right)))
                    logit_rows.append(
                        {
                            "key": key,
                            "value_count": int(left.size),
                            "changed_value_count": changes,
                            "maximum_absolute_delta": maximum_delta,
                        }
                    )
                    total_logit_values += int(left.size)
                    changed_logit_values += changes
                    maximum_logit_absolute_delta = max(
                        maximum_logit_absolute_delta, maximum_delta
                    )
                else:
                    raise ValueError(f"unrecognized artifact key: {key}")
        with prior_report_path.open("r", encoding="utf-8") as handle:
            prior_report = json.load(handle)
        with corrected_report_path.open("r", encoding="utf-8") as handle:
            corrected_report = json.load(handle)
        seed_rows.append(
            {
                "seed": seed,
                "prediction_count": sum(row["sample_count"] for row in key_rows),
                "prediction_change_count": sum(
                    row["prediction_change_count"] for row in key_rows
                ),
                "changed_keys": [
                    row for row in key_rows if row["prediction_change_count"]
                ],
                "logit_value_count": sum(
                    row["value_count"] for row in logit_rows
                ),
                "changed_logit_value_count": sum(
                    row["changed_value_count"] for row in logit_rows
                ),
                "maximum_logit_absolute_delta": max(
                    row["maximum_absolute_delta"] for row in logit_rows
                ),
                "prior_reference_test_accuracy": float(
                    prior_report["reference_test_accuracy"]
                ),
                "corrected_reference_test_accuracy": float(
                    corrected_report["reference_test_accuracy"]
                ),
                "prior_conditions_over_five_point_loss": int(
                    prior_report["conditions_over_five_point_loss"]
                ),
                "corrected_conditions_over_five_point_loss": int(
                    corrected_report["conditions_over_five_point_loss"]
                ),
            }
        )

    report = {
        "schema_version": "SHDSemanticExecutorCorrectionComparison/v2",
        "status": "paired prediction comparison on identical model-input-semantics cells",
        "prior_executor": "TorchEmulator all-operations float32",
        "corrected_executor": (
            "TorchEmulator binary64 arithmetic with explicit declared float32 casts"
        ),
        "seeds": args.seeds,
        "prediction_count": total_predictions,
        "prediction_change_count": total_changes,
        "prediction_change_fraction": total_changes / total_predictions,
        "changed_prediction_keys": sorted(changed_keys),
        "logit_value_count": total_logit_values,
        "changed_logit_value_count": changed_logit_values,
        "maximum_logit_absolute_delta": maximum_logit_absolute_delta,
        "per_seed": seed_rows,
        "source_hashes": source_hashes,
        "code_revision": code_revision(root),
        "interpretation": (
            "Any changed prediction invalidates direct reuse of the corresponding "
            "prior metric under the corrected operational semantics. Unchanged "
            "predictions establish exact decision-level equivalence for the recorded "
            "cells, not trajectory equivalence."
        ),
    }
    write_json_immutable(root / args.output, report)


if __name__ == "__main__":
    main()
