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
    parser.add_argument("--training-root", default="artifacts/shd_v1_final")
    parser.add_argument("--semantic-root", default="artifacts/shd_v1_semantics_final")
    parser.add_argument("--output-root", default="results/shd_v1")
    parser.add_argument("--figure-root", default="paper/figures")
    parser.add_argument("--table-root", default="paper/generated")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "software_matrix_summary.json"
    rows_path = output_root / "software_matrix_rows.csv"
    figure_pdf = root / args.figure_root / "shd_software_pilot.pdf"
    figure_png = root / args.figure_root / "shd_software_pilot.png"
    table_path = root / args.table_root / "shd_software_pilot_table.tex"
    destinations = (summary_path, rows_path, figure_pdf, figure_png, table_path)
    if any(path.exists() for path in destinations):
        raise FileExistsError("aggregate destination exists; evidence is saved")
    all_rows: list[dict[str, object]] = []
    reference_accuracies = []
    input_references: dict[str, str] = {}
    for seed in seeds:
        training_path = root / args.training_root / f"seed_{seed}" / "training_manifest.json"
        semantic_path = root / args.semantic_root / f"seed_{seed}" / "semantic_matrix.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        input_references[f"training_seed_{seed}"] = file_reference(training_path)
        input_references[f"semantics_seed_{seed}"] = file_reference(semantic_path)
        reference_accuracies.append(semantic["reference_test_accuracy"])
        for row in semantic["rows"]:
            all_rows.append({"seed": seed, **row})
    fieldnames = (
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    conditions = sorted({str(row["condition"]) for row in all_rows})
    per_condition = []
    for condition in conditions:
        selected = [row for row in all_rows if row["condition"] == condition]
        losses = np.asarray([float(row["accuracy_loss"]) for row in selected])
        absolute = np.asarray(
            [float(row["absolute_accuracy_change"]) for row in selected]
        )
        disagreements = np.asarray(
            [float(row["audit_disagreement_rate"]) for row in selected]
        )
        bounds = np.asarray(
            [float(row["simultaneous_upper_bound"]) for row in selected]
        )
        per_condition.append(
            {
                "condition": condition,
                "accuracy_loss_mean": float(np.mean(losses)),
                "accuracy_loss_std": float(np.std(losses, ddof=1)),
                "absolute_change_mean": float(np.mean(absolute)),
                "audit_disagreement_mean": float(np.mean(disagreements)),
                "bound_mean": float(np.mean(bounds)),
                "bound_slack_mean": float(np.mean(bounds - absolute)),
                "seeds_over_five_point_loss": int(np.count_nonzero(losses > 0.05)),
            }
        )
    absolute_all = np.asarray(
        [float(row["absolute_accuracy_change"]) for row in all_rows]
    )
    disagreement_all = np.asarray(
        [float(row["audit_disagreement_rate"]) for row in all_rows]
    )
    bound_all = np.asarray(
        [float(row["simultaneous_upper_bound"]) for row in all_rows]
    )
    slack_all = bound_all - absolute_all
    pearson = pearsonr(disagreement_all, absolute_all)
    spearman = spearmanr(disagreement_all, absolute_all)
    problem_conditions = sum(
        item["accuracy_loss_mean"] > 0.05 for item in per_condition
    )
    coverage_violations = sum(
        not bool(row["bound_not_violated"]) for row in all_rows
    )
    median_slack = float(np.median(slack_all))
    p90_slack = float(np.percentile(slack_all, 90))
    summary = {
        "schema_version": "SHDSoftwareAggregate/v1",
        "status": "software pilot; not physical evidence",
        "seeds": list(seeds),
        "cells": len(all_rows),
        "reference_test_accuracy_mean": float(np.mean(reference_accuracies)),
        "reference_test_accuracy_std": float(np.std(reference_accuracies, ddof=1)),
        "conditions_with_mean_loss_over_five_points": problem_conditions,
        "simultaneous_bound_violations": coverage_violations,
        "median_bound_slack": median_slack,
        "p90_bound_slack": p90_slack,
        "pearson_disagreement_vs_absolute_change": {
            "r": float(pearson.statistic),
            "p": float(pearson.pvalue),
        },
        "spearman_disagreement_vs_absolute_change": {
            "rho": float(spearman.statistic),
            "p": float(spearman.pvalue),
        },
        "per_condition": per_condition,
        "gate_assessment": {
            "four_conditions_over_five_point_loss": problem_conditions >= 4,
            "no_observed_simultaneous_bound_violation": coverage_violations == 0,
            "median_slack_at_most_three_points": median_slack <= 0.03,
            "p90_slack_at_most_eight_points": p90_slack <= 0.08,
            "two_physical_backends": False,
            "both_primary_tasks": False,
        },
        "formulation_assessment": (
            "The recurrent semantic transport problem is supported and disagreement is predictive, "
            "but the current distribution-free disagreement certificate fails the preregistered "
            "tightness gates and does not yet support the intended top-venue claim."
        ),
        "input_artifact_references": input_references,
        "rows_csv_reference": file_reference(rows_path),
        "code_revision": code_revision(root),
    }
    write_json(summary_path, summary)

    colors = {condition: plt.cm.tab10(index % 10) for index, condition in enumerate(conditions)}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for condition in conditions:
        selected = [row for row in all_rows if row["condition"] == condition]
        actual = np.asarray(
            [float(row["absolute_accuracy_change"]) for row in selected]
        ) * 100
        bounds = np.asarray(
            [float(row["simultaneous_upper_bound"]) for row in selected]
        ) * 100
        disagreements = np.asarray(
            [float(row["audit_disagreement_rate"]) for row in selected]
        ) * 100
        axes[0].scatter(actual, bounds, s=30, alpha=0.8, color=colors[condition])
        axes[1].scatter(
            disagreements,
            actual,
            s=30,
            alpha=0.8,
            color=colors[condition],
            label=condition.replace("_", " "),
        )
    maximum = max(float(np.max(absolute_all)), float(np.max(bound_all))) * 100 + 2
    axes[0].plot([0, maximum], [0, maximum], "k--", linewidth=1, label="tight bound")
    axes[0].set(xlabel="Actual absolute accuracy change (points)", ylabel="95% simultaneous upper bound (points)", title="Coverage is valid but loose")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(xlabel="Unlabeled audit disagreement (points)", ylabel="Actual absolute accuracy change (points)", title=f"Predictive ranking, not a tight bound ($r$={pearson.statistic:.3f})")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Condition & Loss (pp) & Disagreement (pp) & Bound (pp) \\\\",
        "\\midrule",
    ]
    for item in sorted(per_condition, key=lambda value: value["accuracy_loss_mean"], reverse=True):
        label = str(item["condition"]).replace("_", "\\_")
        lines.append(
            f"{label} & {item['accuracy_loss_mean'] * 100:.1f} & "
            f"{item['audit_disagreement_mean'] * 100:.1f} & "
            f"{item['bound_mean'] * 100:.1f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
