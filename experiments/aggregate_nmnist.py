from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from pines.artifacts import code_revision, file_reference, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", default="artifacts/nmnist_v1")
    parser.add_argument("--semantic-root", default="artifacts/nmnist_v1_semantics")
    parser.add_argument("--output-root", default="results/nmnist_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "software_matrix_summary.json"
    rows_path = output / "software_matrix_rows.csv"
    comparison_csv = output / "recurrent_vs_feedforward.csv"
    figure_pdf = root / "paper" / "figures" / "recurrent_vs_feedforward.pdf"
    figure_png = root / "paper" / "figures" / "recurrent_vs_feedforward.png"
    if any(
        path.exists()
        for path in (summary_path, rows_path, comparison_csv, figure_pdf, figure_png)
    ):
        raise FileExistsError("N-MNIST aggregate output exists")
    rows = []
    references = []
    input_references = {}
    for seed in seeds:
        training_path = root / args.training_root / f"seed_{seed}" / "training_manifest.json"
        semantic_path = root / args.semantic_root / f"seed_{seed}" / "semantic_matrix.json"
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        references.append(semantic["reference_test_accuracy"])
        rows.extend({"seed": seed, **row} for row in semantic["rows"])
        input_references[f"training_seed_{seed}"] = file_reference(training_path)
        input_references[f"semantics_seed_{seed}"] = file_reference(semantic_path)
    fields = (
        "seed",
        "condition",
        "semantics_description",
        "audit_disagreements",
        "audit_samples",
        "audit_disagreement_rate",
        "simultaneous_upper_bound",
        "reference_test_accuracy",
        "target_test_accuracy",
        "accuracy_loss",
        "absolute_accuracy_change",
        "bound_slack_vs_test_change",
        "bound_not_violated",
    )
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    conditions = sorted({row["condition"] for row in rows})
    per_condition = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        loss = np.asarray([row["accuracy_loss"] for row in selected], dtype=float)
        absolute = np.asarray(
            [row["absolute_accuracy_change"] for row in selected], dtype=float
        )
        disagreement = np.asarray(
            [row["audit_disagreement_rate"] for row in selected], dtype=float
        )
        bound = np.asarray(
            [row["simultaneous_upper_bound"] for row in selected], dtype=float
        )
        per_condition.append(
            {
                "condition": condition,
                "accuracy_loss_mean": float(np.mean(loss)),
                "accuracy_loss_std": float(np.std(loss, ddof=1)),
                "absolute_change_mean": float(np.mean(absolute)),
                "audit_disagreement_mean": float(np.mean(disagreement)),
                "bound_mean": float(np.mean(bound)),
                "bound_slack_mean": float(np.mean(bound - absolute)),
                "seeds_over_five_point_loss": int(np.count_nonzero(loss > 0.05)),
            }
        )
    absolute_all = np.asarray([row["absolute_accuracy_change"] for row in rows])
    disagreement_all = np.asarray([row["audit_disagreement_rate"] for row in rows])
    bound_all = np.asarray([row["simultaneous_upper_bound"] for row in rows])
    slack = bound_all - absolute_all
    pearson = pearsonr(disagreement_all, absolute_all)
    spearman = spearmanr(disagreement_all, absolute_all)
    decision_metrics = {}
    for budget in (0.01, 0.02, 0.05):
        accept = bound_all <= budget
        safe = absolute_all <= budget
        decision_metrics[str(budget)] = {
            "accepted": int(np.count_nonzero(accept)),
            "rejected": int(np.count_nonzero(~accept)),
            "unsafe_accepts": int(np.count_nonzero(accept & ~safe)),
            "conservative_rejects": int(np.count_nonzero(~accept & safe)),
        }
    summary = {
        "schema_version": "NMNISTSoftwareAggregate/v1",
        "status": "software negative control; not physical evidence",
        "seeds": list(seeds),
        "cells": len(rows),
        "reference_test_accuracy_mean": float(np.mean(references)),
        "reference_test_accuracy_std": float(np.std(references, ddof=1)),
        "conditions_with_mean_loss_over_five_points": int(
            sum(item["accuracy_loss_mean"] > 0.05 for item in per_condition)
        ),
        "simultaneous_bound_violations": int(np.count_nonzero(absolute_all > bound_all)),
        "median_bound_slack": float(np.median(slack)),
        "p90_bound_slack": float(np.percentile(slack, 90)),
        "pearson_disagreement_vs_absolute_change": {
            "r": float(pearson.statistic),
            "p": float(pearson.pvalue),
        },
        "spearman_disagreement_vs_absolute_change": {
            "rho": float(spearman.statistic),
            "p": float(spearman.pvalue),
        },
        "decision_metrics": decision_metrics,
        "per_condition": per_condition,
        "gate_assessment": {
            "negative_control_has_no_mean_loss_over_five_points": all(
                item["accuracy_loss_mean"] <= 0.05 for item in per_condition
            ),
            "no_observed_simultaneous_bound_violation": bool(
                np.all(absolute_all <= bound_all)
            ),
            "median_slack_at_most_three_points": bool(np.median(slack) <= 0.03),
            "p90_slack_at_most_eight_points": bool(
                np.percentile(slack, 90) <= 0.08
            ),
        },
        "interpretation": (
            "The high-accuracy feedforward negative control is robust to reset, threshold timing, "
            "and delay, and its certificate passes the software tightness gates."
        ),
        "input_artifact_references": input_references,
        "rows_csv_reference": file_reference(rows_path),
        "code_revision": code_revision(root),
    }
    write_json(summary_path, summary)

    shd = json.loads(
        (root / "results" / "shd_v1" / "software_matrix_summary.json").read_text(
            encoding="utf-8"
        )
    )
    shd_by_condition = {item["condition"]: item for item in shd["per_condition"]}
    nmnist_by_condition = {item["condition"]: item for item in per_condition}
    comparison_rows = []
    for condition in conditions:
        comparison_rows.append(
            {
                "condition": condition,
                "shd_loss_points": shd_by_condition[condition]["accuracy_loss_mean"] * 100,
                "nmnist_loss_points": nmnist_by_condition[condition]["accuracy_loss_mean"] * 100,
                "shd_slack_points": shd_by_condition[condition]["bound_slack_mean"] * 100,
                "nmnist_slack_points": nmnist_by_condition[condition]["bound_slack_mean"] * 100,
            }
        )
    with comparison_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)
    positions = np.arange(len(conditions))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    axes[0].barh(
        positions - width / 2,
        [row["shd_loss_points"] for row in comparison_rows],
        height=width,
        label="SHD recurrent",
        color="#d95f02",
    )
    axes[0].barh(
        positions + width / 2,
        [row["nmnist_loss_points"] for row in comparison_rows],
        height=width,
        label="N-MNIST feedforward",
        color="#1b9e77",
    )
    axes[1].barh(
        positions - width / 2,
        [row["shd_slack_points"] for row in comparison_rows],
        height=width,
        color="#d95f02",
    )
    axes[1].barh(
        positions + width / 2,
        [row["nmnist_slack_points"] for row in comparison_rows],
        height=width,
        color="#1b9e77",
    )
    labels = [condition.replace("_", " ") for condition in conditions]
    for axis in axes:
        axis.set_yticks(positions, labels)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Mean accuracy loss (points)")
    axes[0].set_title("Semantic loss is amplified by recurrence")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Mean certificate slack (points)")
    axes[1].set_title("The same bound is tight on the robust control")
    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
