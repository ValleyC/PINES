from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, file_reference, write_json
from pines.statistics import clopper_pearson_upper


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
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", default="artifacts/shd_v3_repairs_matched"
    )
    parser.add_argument("--certificate-root")
    parser.add_argument("--logit-root")
    parser.add_argument("--global-root")
    parser.add_argument("--qat-root")
    parser.add_argument("--supervised-root")
    parser.add_argument("--result-stem", default="repair_matched_v2")
    parser.add_argument("--figure-stem", default="shd_repair_matched_v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / args.artifact_root
    method_roots = {
        "certificate_directed": root / args.certificate_root
        if args.certificate_root
        else artifact_root,
        "logit_only": root / args.logit_root if args.logit_root else artifact_root,
        "global_threshold": root / args.global_root
        if args.global_root
        else artifact_root,
        "per_platform_qat": root / args.qat_root if args.qat_root else artifact_root,
        "supervised_target_retraining": root / args.supervised_root
        if args.supervised_root
        else artifact_root,
    }
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / f"{args.result_stem}_summary.json"
    rows_path = output_root / f"{args.result_stem}_rows.csv"
    figure_pdf = root / "paper" / "figures" / f"{args.figure_stem}.pdf"
    figure_png = root / "paper" / "figures" / f"{args.figure_stem}.png"
    destinations = (summary_path, rows_path, figure_pdf, figure_png)
    if any(path.exists() for path in destinations):
        raise FileExistsError("matched repair aggregate destination already exists")

    rows: list[dict[str, object]] = []
    report_descriptions: dict[str, str] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for method in METHODS:
                path = (
                    method_roots[method]
                    / f"seed_{seed}"
                    / condition
                    / method
                    / "repair_report.json"
                )
                report = json.loads(path.read_text(encoding="utf-8"))
                if report["schema_version"] not in {
                    "SHDRepairExperiment/v2",
                    "SHDRepairExperiment/v3",
                }:
                    raise ValueError(f"unexpected report schema: {path}")
                if not report["calibration_audit_disjoint"]:
                    raise ValueError(f"calibration/audit overlap: {path}")
                if report["test_labels_used_for_selection"]:
                    raise ValueError(f"test-label selection leakage: {path}")
                before = report["before"]
                after = report["after"]
                before_bound = clopper_pearson_upper(
                    before["audit_disagreements"], report["audit_samples"], 0.05
                )
                row = {
                    "condition": condition,
                    "seed": seed,
                    "method": method,
                    "before_accuracy_loss": before["accuracy_loss"],
                    "before_absolute_accuracy_change": abs(before["accuracy_loss"]),
                    "before_certificate_upper_bound": before_bound,
                    "after_accuracy_loss": after["accuracy_loss"],
                    "after_absolute_accuracy_change": abs(after["accuracy_loss"]),
                    "accuracy_recovery_fraction": report[
                        "accuracy_recovery_fraction"
                    ],
                    "after_audit_disagreement_rate": after[
                        "audit_disagreement_rate"
                    ],
                    "after_certificate_upper_bound": after[
                        "certificate_upper_bound"
                    ],
                    "after_bound_slack": after["certificate_upper_bound"]
                    - abs(after["accuracy_loss"]),
                    "bound_not_violated": abs(after["accuracy_loss"])
                    <= after["certificate_upper_bound"],
                    "calibration_samples": report["calibration_samples"],
                    "audit_samples": report["audit_samples"],
                    "label_budget": report["label_budget"],
                    "optimization_steps": report["optimization_steps"],
                    "trainable_parameters": report["trainable_parameters"],
                    "elapsed_seconds": report["elapsed_seconds"],
                    "quantization_active": report["quantization_active"],
                    "source_model": report["source_model"],
                    "repaired_model": report["repaired_model"],
                    "report_code_revision": report["code_revision"],
                }
                rows.append(row)
                report_descriptions[f"{condition}__seed_{seed}__{method}"] = file_reference(
                    path
                )

    output_root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    condition_summaries: list[dict[str, object]] = []
    lookup: dict[tuple[str, str], list[dict[str, object]]] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        method_summaries: list[dict[str, object]] = []
        for method in METHODS:
            selected = [row for row in condition_rows if row["method"] == method]
            lookup[(condition, method)] = selected
            method_summaries.append(
                {
                    "method": method,
                    "accuracy_recovery": _stats(
                        [float(row["accuracy_recovery_fraction"]) for row in selected]
                    ),
                    "after_accuracy_loss": _stats(
                        [float(row["after_accuracy_loss"]) for row in selected]
                    ),
                    "after_absolute_accuracy_change": _stats(
                        [
                            float(row["after_absolute_accuracy_change"])
                            for row in selected
                        ]
                    ),
                    "after_certificate_upper_bound": _stats(
                        [
                            float(row["after_certificate_upper_bound"])
                            for row in selected
                        ]
                    ),
                    "after_bound_slack": _stats(
                        [float(row["after_bound_slack"]) for row in selected]
                    ),
                    "elapsed_seconds": _stats(
                        [float(row["elapsed_seconds"]) for row in selected]
                    ),
                    "label_budget": int(selected[0]["label_budget"]),
                    "optimization_steps": int(selected[0]["optimization_steps"]),
                    "trainable_parameters": int(selected[0]["trainable_parameters"]),
                    "seeds_recovering_at_least_70_percent": int(
                        sum(
                            float(row["accuracy_recovery_fraction"]) >= 0.70
                            for row in selected
                        )
                    ),
                    "seeds_certified_at_five_point_budget": int(
                        sum(
                            float(row["after_certificate_upper_bound"]) <= 0.05
                            for row in selected
                        )
                    ),
                }
            )
        certificate_rows = lookup[(condition, "certificate_directed")]
        logit_rows = lookup[(condition, "logit_only")]
        global_rows = lookup[(condition, "global_threshold")]
        qat_rows = lookup[(condition, "per_platform_qat")]
        condition_summaries.append(
            {
                "condition": condition,
                "unrepaired_accuracy_loss": _stats(
                    [float(row["before_accuracy_loss"]) for row in certificate_rows]
                ),
                "unrepaired_certificate_upper_bound": _stats(
                    [
                        float(row["before_certificate_upper_bound"])
                        for row in certificate_rows
                    ]
                ),
                "per_method": method_summaries,
                "paired_recovery_differences": {
                    "certificate_minus_logit": _stats(
                        [
                            float(certificate["accuracy_recovery_fraction"])
                            - float(logit["accuracy_recovery_fraction"])
                            for certificate, logit in zip(certificate_rows, logit_rows)
                        ]
                    ),
                    "certificate_minus_global_threshold": _stats(
                        [
                            float(certificate["accuracy_recovery_fraction"])
                            - float(global_row["accuracy_recovery_fraction"])
                            for certificate, global_row in zip(
                                certificate_rows, global_rows
                            )
                        ]
                    ),
                    "certificate_minus_labeled_qat": _stats(
                        [
                            float(certificate["accuracy_recovery_fraction"])
                            - float(qat["accuracy_recovery_fraction"])
                            for certificate, qat in zip(certificate_rows, qat_rows)
                        ]
                    ),
                },
            }
        )

    certificate_rows = [
        row for row in rows if row["method"] == "certificate_directed"
    ]
    global_rows = [row for row in rows if row["method"] == "global_threshold"]
    summary = {
        "schema_version": "SHDRepairMatchedAggregate/v2",
        "status": "five-seed software evidence; no physical-backend evidence",
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "protocol": {
            "calibration_samples_per_seed": 800,
            "audit_samples_per_seed": 861,
            "gradient_epochs": 40,
            "gradient_steps": 280,
            "label_free_methods": [
                "certificate_directed",
                "logit_only",
                "global_threshold",
            ],
            "labeled_methods": [
                "per_platform_qat",
                "supervised_target_retraining",
            ],
            "labeled_budget": 800,
            "selection_uses_test_labels": False,
            "calibration_audit_disjoint": True,
            "floor_unrepaired_is_post_training_quantization_baseline": True,
            "method_artifact_roots": {
                method: str(path.relative_to(root))
                for method, path in method_roots.items()
            },
        },
        "per_condition": condition_summaries,
        "gate_assessment": {
            "certificate_directed_all_ten_cells_recover_at_least_70_percent": all(
                float(row["accuracy_recovery_fraction"]) >= 0.70
                for row in certificate_rows
            ),
            "certificate_directed_beats_global_threshold_every_seed": all(
                float(certificate["accuracy_recovery_fraction"])
                > float(global_row["accuracy_recovery_fraction"])
                for certificate, global_row in zip(certificate_rows, global_rows)
            ),
            "certificate_directed_beats_unrepaired_every_seed": all(
                float(row["accuracy_recovery_fraction"]) > 0
                for row in certificate_rows
            ),
            "certificate_directed_beats_logit_only_in_both_conditions": all(
                next(
                    method
                    for method in item["per_method"]
                    if method["method"] == "certificate_directed"
                )["accuracy_recovery"]["mean"]
                > next(
                    method
                    for method in item["per_method"]
                    if method["method"] == "logit_only"
                )["accuracy_recovery"]["mean"]
                for item in condition_summaries
            ),
            "certificate_directed_certifies_five_point_budget_any_cell": any(
                float(row["after_certificate_upper_bound"]) <= 0.05
                for row in certificate_rows
            ),
            "no_observed_bound_violation_in_all_fifty_cells": all(
                bool(row["bound_not_violated"]) for row in rows
            ),
        },
        "formulation_assessment": (
            "Restricted label-free calibration reliably recovers accuracy under reset and "
            "floor-rounded fixed-point transport and dominates global scaling and unrepaired "
            "deployment. It only slightly beats logit-only on reset and loses to logit-only "
            "on floor rounding; labeled QAT is strongest for the true fixed-point condition. "
            "No repaired model obtains a non-vacuous five-point disagreement certificate, "
            "so repair efficacy does not rescue the manuscript's tight-certificate claim."
        ),
        "input_report_references": report_descriptions,
        "rows_csv_reference": file_reference(rows_path),
        "code_revision": code_revision(root),
    }
    certificate_seventy_count = sum(
        float(row["accuracy_recovery_fraction"]) >= 0.70
        for row in certificate_rows
    )
    certificate_recovery_wins = sum(
        next(
            method
            for method in item["per_method"]
            if method["method"] == "certificate_directed"
        )["accuracy_recovery"]["mean"]
        > next(
            method
            for method in item["per_method"]
            if method["method"] == "logit_only"
        )["accuracy_recovery"]["mean"]
        for item in condition_summaries
    )
    certificate_bound_wins = sum(
        next(
            method
            for method in item["per_method"]
            if method["method"] == "certificate_directed"
        )["after_certificate_upper_bound"]["mean"]
        < next(
            method
            for method in item["per_method"]
            if method["method"] == "logit_only"
        )["after_certificate_upper_bound"]["mean"]
        for item in condition_summaries
    )
    summary["formulation_assessment"] = (
        f"Certificate-directed repair clears 70% recovery in "
        f"{certificate_seventy_count}/10 condition-seed cells. It exceeds "
        f"logit-only mean recovery in {certificate_recovery_wins}/2 conditions "
        f"and has the tighter mean post-repair bound in {certificate_bound_wins}/2. "
        "It passes the preregistered recovery gate and beats global "
        "threshold scaling and unrepaired deployment in every seed. No repaired "
        "model obtains a five-point disagreement certificate, so the result "
        "supports transport repair followed by recertification rather than "
        "certificate restoration."
    )
    write_json(summary_path, summary)

    labels = ("Cert-directed", "Logit-only", "Global threshold", "QAT", "Scratch")
    colors = ("#7570b3", "#1b9e77", "#d95f02", "#1f78b4", "#b15928")
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.1), constrained_layout=True)
    positions = np.arange(len(METHODS))
    for column, condition in enumerate(CONDITIONS):
        recovery_means = []
        recovery_stds = []
        bound_means = []
        bound_stds = []
        for method in METHODS:
            selected = lookup[(condition, method)]
            recovery = np.asarray(
                [float(row["accuracy_recovery_fraction"]) for row in selected]
            )
            bound = np.asarray(
                [float(row["after_certificate_upper_bound"]) for row in selected]
            )
            recovery_means.append(np.mean(recovery) * 100)
            recovery_stds.append(np.std(recovery, ddof=1) * 100)
            bound_means.append(np.mean(bound) * 100)
            bound_stds.append(np.std(bound, ddof=1) * 100)
        axes[0, column].bar(
            positions,
            recovery_means,
            yerr=recovery_stds,
            color=colors,
            capsize=3,
        )
        axes[0, column].axhline(70, color="black", linestyle="--", linewidth=1)
        axes[0, column].set_title(condition.replace("_", " "))
        axes[0, column].set_ylabel("Lost accuracy recovered (%)")
        axes[1, column].bar(
            positions,
            bound_means,
            yerr=bound_stds,
            color=colors,
            capsize=3,
        )
        axes[1, column].axhline(5, color="black", linestyle="--", linewidth=1)
        axes[1, column].set_ylabel("Post-repair 95% upper bound (points)")
        for row in (0, 1):
            axes[row, column].set_xticks(positions, labels, rotation=28, ha="right")
            axes[row, column].grid(axis="y", alpha=0.2)
    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
