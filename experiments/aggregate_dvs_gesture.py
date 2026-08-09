from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", default="artifacts/dvs_gesture_v3")
    parser.add_argument(
        "--semantic-root", default="artifacts/dvs_gesture_v3_semantics"
    )
    parser.add_argument("--output-root", default="results/dvs_gesture_v3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "software_matrix_summary.json"
    rows_path = output / "software_matrix_rows.csv"
    figure_pdf = root / "paper" / "figures" / "dvs_software_matrix.pdf"
    figure_png = root / "paper" / "figures" / "dvs_software_matrix.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("DVS aggregate destination exists; evidence is immutable")

    rows: list[dict[str, object]] = []
    references: list[float] = []
    input_hashes: dict[str, str] = {}
    for seed in seeds:
        training_path = (
            root / args.training_root / f"seed_{seed}" / "training_manifest.json"
        )
        semantic_path = (
            root / args.semantic_root / f"seed_{seed}" / "semantic_matrix.json"
        )
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        references.append(float(semantic["reference_test_accuracy"]))
        rows.extend({"seed": seed, **row} for row in semantic["rows"])
        input_hashes[f"training_seed_{seed}"] = sha256_file(training_path)
        input_hashes[f"semantics_seed_{seed}"] = sha256_file(semantic_path)

    fields = (
        "seed",
        "condition",
        "semantics_hash",
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

    conditions = sorted({str(row["condition"]) for row in rows})
    per_condition: list[dict[str, object]] = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        loss = np.asarray([float(row["accuracy_loss"]) for row in selected])
        absolute = np.asarray(
            [float(row["absolute_accuracy_change"]) for row in selected]
        )
        disagreement = np.asarray(
            [float(row["audit_disagreement_rate"]) for row in selected]
        )
        bound = np.asarray(
            [float(row["simultaneous_upper_bound"]) for row in selected]
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

    absolute_all = np.asarray(
        [float(row["absolute_accuracy_change"]) for row in rows]
    )
    disagreement_all = np.asarray(
        [float(row["audit_disagreement_rate"]) for row in rows]
    )
    bound_all = np.asarray([float(row["simultaneous_upper_bound"]) for row in rows])
    slack = bound_all - absolute_all
    pearson = pearsonr(disagreement_all, absolute_all)
    spearman = spearmanr(disagreement_all, absolute_all)
    decision_metrics: dict[str, dict[str, int]] = {}
    for budget in (0.01, 0.02, 0.05):
        accept = bound_all <= budget
        safe = absolute_all <= budget
        decision_metrics[str(budget)] = {
            "accepted": int(np.count_nonzero(accept)),
            "rejected": int(np.count_nonzero(~accept)),
            "unsafe_accepts": int(np.count_nonzero(accept & ~safe)),
            "conservative_rejects": int(np.count_nonzero(~accept & safe)),
        }
    problem_conditions = int(
        sum(float(item["accuracy_loss_mean"]) > 0.05 for item in per_condition)
    )
    violations = int(np.count_nonzero(absolute_all > bound_all))
    summary = {
        "schema_version": "DVSGestureSoftwareAggregate/v1",
        "status": (
            "development software evidence; semantic targets were untouched before freeze, "
            "but official test accuracy was used for reference-model selection; not physical evidence"
        ),
        "seeds": list(seeds),
        "cells": len(rows),
        "audit_samples_per_seed": int(rows[0]["audit_samples"]),
        "reference_test_accuracy_mean": float(np.mean(references)),
        "reference_test_accuracy_std": float(np.std(references, ddof=1)),
        "conditions_with_mean_loss_over_five_points": problem_conditions,
        "simultaneous_bound_violations": violations,
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
            "reference_mean_at_least_80_percent": bool(np.mean(references) >= 0.80),
            "no_observed_simultaneous_bound_violation": violations == 0,
            "median_slack_at_most_three_points": bool(np.median(slack) <= 0.03),
            "p90_slack_at_most_eight_points": bool(
                np.percentile(slack, 90) <= 0.08
            ),
            "four_conditions_over_five_point_loss": problem_conditions >= 4,
        },
        "formulation_assessment": (
            "A credible convolutional/recurrent reference confirms severe semantic transport "
            "loss and predictive disagreement ranking, but the distribution-free certificate "
            "again fails the frozen tightness gates. The small audit set also creates a roughly "
            "five-point simultaneous-confidence floor near zero observed disagreement. This is "
            "development evidence because official test accuracy selected the reference pipeline; "
            "a fresh sequestered replication is required for submission."
        ),
        "input_artifact_hashes": input_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.3), constrained_layout=True)
    axes[0].scatter(
        absolute_all * 100,
        bound_all * 100,
        c=disagreement_all * 100,
        cmap="viridis",
        s=38,
        alpha=0.85,
    )
    maximum = max(float(np.max(absolute_all)), float(np.max(bound_all))) * 100 + 2
    axes[0].plot([0, maximum], [0, maximum], "k--", linewidth=1)
    axes[0].set(
        xlabel="Actual absolute accuracy change (points)",
        ylabel="95% simultaneous upper bound (points)",
        title="Coverage holds, but bounds remain loose",
    )
    ordered = sorted(
        per_condition, key=lambda item: float(item["accuracy_loss_mean"])
    )
    positions = np.arange(len(ordered))
    axes[1].barh(
        positions,
        [float(item["accuracy_loss_mean"]) * 100 for item in ordered],
        color="#7570b3",
    )
    axes[1].set_yticks(
        positions, [str(item["condition"]).replace("_", " ") for item in ordered]
    )
    axes[1].set(
        xlabel="Mean accuracy loss (points)",
        title="Floor-rounded fixed point dominates loss",
    )
    axes[1].grid(axis="x", alpha=0.2)
    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
