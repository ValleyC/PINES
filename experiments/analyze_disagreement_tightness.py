from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.statistics import disagreement_tightness_witness


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    benchmarks = {
        "SHD": root / "artifacts" / "shd_v1_semantics_final",
        "N-MNIST": root / "artifacts" / "nmnist_v1_semantics",
        "DVS Gesture": root / "artifacts" / "dvs_gesture_v3_semantics",
    }
    output = root / "results" / "disagreement_tightness"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "witness_summary.json"
    rows_path = output / "witness_rows.csv"
    if summary_path.exists() or rows_path.exists():
        raise FileExistsError("disagreement tightness aggregate exists")

    rows: list[dict[str, object]] = []
    prediction_artifact_hashes: dict[str, str] = {}
    for benchmark, artifact_root in benchmarks.items():
        for seed in seeds:
            path = artifact_root / f"seed_{seed}" / "semantic_predictions.npz"
            prediction_artifact_hashes[f"{benchmark}/seed_{seed}"] = sha256_file(path)
            with np.load(path, allow_pickle=False) as predictions:
                reference = np.asarray(predictions["pred__audit__reference"])
                target_keys = sorted(
                    key
                    for key in predictions.files
                    if key.startswith("pred__audit__")
                    and key != "pred__audit__reference"
                )
                for key in target_keys:
                    condition = key.removeprefix("pred__audit__")
                    target = np.asarray(predictions[key])
                    witness = disagreement_tightness_witness(reference, target)
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "seed": seed,
                            "condition": condition,
                            "samples": len(reference),
                            "disagreement_rate": witness.disagreement_rate,
                            "reference_favoring_accuracy_change": (
                                witness.reference_minus_target_accuracy
                            ),
                            "target_favoring_accuracy_change": (
                                witness.target_minus_reference_accuracy
                            ),
                            "reference_favoring_labels_hash": array_hash(
                                witness.reference_favoring_labels
                            ),
                            "target_favoring_labels_hash": array_hash(
                                witness.target_favoring_labels
                            ),
                            "reference_witness_attains_bound": bool(
                                np.isclose(
                                    witness.reference_minus_target_accuracy,
                                    witness.disagreement_rate,
                                )
                            ),
                            "target_witness_attains_bound": bool(
                                np.isclose(
                                    witness.target_minus_reference_accuracy,
                                    witness.disagreement_rate,
                                )
                            ),
                        }
                    )

    fields = tuple(rows[0].keys())
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    per_benchmark = []
    for benchmark in benchmarks:
        selected = [row for row in rows if row["benchmark"] == benchmark]
        per_benchmark.append(
            {
                "benchmark": benchmark,
                "cells": len(selected),
                "all_reference_witnesses_attain_bound": all(
                    bool(row["reference_witness_attains_bound"])
                    for row in selected
                ),
                "all_target_witnesses_attain_bound": all(
                    bool(row["target_witness_attains_bound"])
                    for row in selected
                ),
                "minimum_disagreement_rate": float(
                    min(float(row["disagreement_rate"]) for row in selected)
                ),
                "maximum_disagreement_rate": float(
                    max(float(row["disagreement_rate"]) for row in selected)
                ),
            }
        )
    summary = {
        "schema_version": "DisagreementTightnessWitnessAggregate/v1",
        "status": (
            "label-construction audit over stored unlabeled prediction pairs; the "
            "constructed labels are witnesses, not observed dataset labels"
        ),
        "cells": len(rows),
        "per_benchmark": per_benchmark,
        "gate_assessment": {
            "every_prediction_pair_attains_both_signs": all(
                bool(row["reference_witness_attains_bound"])
                and bool(row["target_witness_attains_bound"])
                for row in rows
            )
        },
        "theoretical_implication": (
            "For predictions alone, disagreement is a sharp distribution-free "
            "upper bound on absolute accuracy change. A uniformly tighter label-free "
            "guarantee requires additional assumptions or information."
        ),
        "prediction_artifact_hashes": prediction_artifact_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
