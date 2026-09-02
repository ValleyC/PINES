from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, file_reference, write_json


SEEDS = (1701, 2718, 3141, 5772, 8119)
METHODS = ("certificate_directed", "logit_only", "global_threshold")
CONDITION = "floor_rounding_saturation"


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
    parser.add_argument("--certificate-root")
    parser.add_argument("--logit-root")
    parser.add_argument("--global-root")
    parser.add_argument("--include-qat", action="store_true")
    parser.add_argument("--qat-root")
    parser.add_argument("--include-supervised", action="store_true")
    parser.add_argument("--supervised-root")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / "artifacts" / "dvs_gesture_v4_repairs"
    output_root = root / "results" / "dvs_gesture_v3"
    methods = METHODS + (("per_platform_qat",) if args.include_qat else ())
    methods += (
        ("supervised_target_retraining",)
        if args.include_supervised
        else ()
    )
    suffix = f"_{args.suffix}" if args.suffix else ""
    method_roots = {
        "certificate_directed": root / args.certificate_root
        if args.certificate_root
        else artifact_root,
        "logit_only": root / args.logit_root if args.logit_root else artifact_root,
        "global_threshold": root / args.global_root
        if args.global_root
        else artifact_root,
    }
    summary_path = output_root / f"repair_floor{suffix}_summary.json"
    rows_path = output_root / f"repair_floor{suffix}_rows.csv"
    figure_pdf = root / "paper" / "figures" / f"dvs_repair_floor{suffix}.pdf"
    figure_png = root / "paper" / "figures" / f"dvs_repair_floor{suffix}.png"
    if any(
        path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)
    ):
        raise FileExistsError("DVS repair aggregate destination already exists")

    rows: list[dict[str, object]] = []
    report_descriptions: dict[str, str] = {}
    for seed in SEEDS:
        for method in methods:
            method_artifact_root = (
                root / args.qat_root
                if method == "per_platform_qat" and args.qat_root
                else root / args.supervised_root
                if method == "supervised_target_retraining"
                and args.supervised_root
                else method_roots[method]
            )
            path = (
                method_artifact_root
                / f"seed_{seed}"
                / CONDITION
                / method
                / "repair_report.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["schema_version"] != "DVSGestureRepairExperiment/v1":
                raise ValueError(f"unexpected report schema: {path}")
            if not report["calibration_audit_disjoint"]:
                raise ValueError(f"calibration/audit overlap: {path}")
            if report["test_labels_used_for_selection"]:
                raise ValueError(f"test-label selection leakage: {path}")
            after = report["after"]
            row = {
                "seed": seed,
                "method": method,
                "condition": CONDITION,
                "before_accuracy_loss": report["before"]["accuracy_loss"],
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
                "confidence_alpha": report["confidence_alpha"],
                "source_model": report["source_model"],
                "repaired_model": report["repaired_model"],
                "report_code_revision": report["code_revision"],
            }
            rows.append(row)
            report_descriptions[f"seed_{seed}__{method}"] = file_reference(path)

    output_root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    method_rows: dict[str, list[dict[str, object]]] = {}
    per_method: list[dict[str, object]] = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        method_rows[method] = selected
        per_method.append(
            {
                "method": method,
                "accuracy_recovery": _stats(
                    [float(row["accuracy_recovery_fraction"]) for row in selected]
                ),
                "after_accuracy_loss": _stats(
                    [float(row["after_accuracy_loss"]) for row in selected]
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

    certificate_rows = method_rows["certificate_directed"]
    logit_rows = method_rows["logit_only"]
    global_rows = method_rows["global_threshold"]
    qat_rows = method_rows.get("per_platform_qat")
    supervised_rows = method_rows.get("supervised_target_retraining")
    certificate_logit_win_count = sum(
        float(certificate["accuracy_recovery_fraction"])
        > float(logit["accuracy_recovery_fraction"])
        for certificate, logit in zip(certificate_rows, logit_rows)
    )
    certificate_minus_logit_recovery = 100.0 * float(
        np.mean(
            [
                float(certificate["accuracy_recovery_fraction"])
                - float(logit["accuracy_recovery_fraction"])
                for certificate, logit in zip(certificate_rows, logit_rows)
            ]
        )
    )
    certificate_minus_logit_bound = 100.0 * float(
        np.mean(
            [
                float(certificate["after_certificate_upper_bound"])
                - float(logit["after_certificate_upper_bound"])
                for certificate, logit in zip(certificate_rows, logit_rows)
            ]
        )
    )
    summary = {
        "schema_version": (
            "DVSGestureRepairAggregate/v3"
            if args.include_supervised
            else
            "DVSGestureRepairAggregate/v2"
            if args.include_qat
            else "DVSGestureRepairAggregate/v1"
        ),
        "status": (
            "five-seed software development evidence; official test data helped select "
            "the reference pipeline before target semantics were frozen"
        ),
        "condition": CONDITION,
        "seeds": list(SEEDS),
        "methods": list(methods),
        "protocol": {
            "calibration_samples_per_seed": 97,
            "audit_samples_per_seed": 104,
            "label_budgets": {
                method: int(method_rows[method][0]["label_budget"])
                for method in methods
            },
            "gradient_epochs": 40,
            "gradient_steps": 280,
            "simultaneous_alpha_per_report": 0.005,
            "qat_artifact_root": args.qat_root,
            "supervised_artifact_root": args.supervised_root,
            "selection_uses_test_labels": False,
            "calibration_audit_disjoint": True,
            "method_artifact_roots": {
                method: str(
                    (
                        root / args.qat_root
                        if method == "per_platform_qat" and args.qat_root
                        else root / args.supervised_root
                        if method == "supervised_target_retraining"
                        and args.supervised_root
                        else method_roots[method]
                    ).relative_to(root)
                )
                for method in methods
            },
        },
        "per_method": per_method,
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
            "certificate_minus_logit_bound": _stats(
                [
                    float(certificate["after_certificate_upper_bound"])
                    - float(logit["after_certificate_upper_bound"])
                    for certificate, logit in zip(certificate_rows, logit_rows)
                ]
            ),
            **(
                {
                    "certificate_minus_labeled_qat": _stats(
                        [
                            float(certificate["accuracy_recovery_fraction"])
                            - float(qat["accuracy_recovery_fraction"])
                            for certificate, qat in zip(certificate_rows, qat_rows)
                        ]
                    ),
                    "certificate_minus_labeled_qat_bound": _stats(
                        [
                            float(certificate["after_certificate_upper_bound"])
                            - float(qat["after_certificate_upper_bound"])
                            for certificate, qat in zip(certificate_rows, qat_rows)
                        ]
                    ),
                }
                if qat_rows is not None
                else {}
            ),
            **(
                {
                    "certificate_minus_supervised_retraining": _stats(
                        [
                            float(certificate["accuracy_recovery_fraction"])
                            - float(supervised["accuracy_recovery_fraction"])
                            for certificate, supervised in zip(
                                certificate_rows, supervised_rows
                            )
                        ]
                    ),
                    "certificate_minus_supervised_retraining_bound": _stats(
                        [
                            float(certificate["after_certificate_upper_bound"])
                            - float(supervised["after_certificate_upper_bound"])
                            for certificate, supervised in zip(
                                certificate_rows, supervised_rows
                            )
                        ]
                    ),
                }
                if supervised_rows is not None
                else {}
            ),
        },
        "gate_assessment": {
            "certificate_directed_all_seeds_recover_at_least_70_percent": all(
                float(row["accuracy_recovery_fraction"]) >= 0.70
                for row in certificate_rows
            ),
            "certificate_directed_beats_global_threshold_every_seed": all(
                float(certificate["accuracy_recovery_fraction"])
                > float(global_row["accuracy_recovery_fraction"])
                for certificate, global_row in zip(certificate_rows, global_rows)
            ),
            "certificate_directed_beats_logit_only_every_seed": all(
                float(certificate["accuracy_recovery_fraction"])
                > float(logit["accuracy_recovery_fraction"])
                for certificate, logit in zip(certificate_rows, logit_rows)
            ),
            "certificate_directed_beats_logit_only_mean_recovery": bool(
                np.mean(
                    [
                        float(row["accuracy_recovery_fraction"])
                        for row in certificate_rows
                    ]
                )
                > np.mean(
                    [float(row["accuracy_recovery_fraction"]) for row in logit_rows]
                )
            ),
            "certificate_directed_certifies_five_point_budget_any_seed": any(
                float(row["after_certificate_upper_bound"]) <= 0.05
                for row in certificate_rows
            ),
            "no_observed_bound_violation": all(
                bool(row["bound_not_violated"]) for row in rows
            ),
        },
        "formulation_assessment": (
            "Restricted label-free calibration clears 70 percent recovery for every seed "
            f"and beats logit-only in {certificate_logit_win_count}/5 paired seeds. Its "
            f"mean recovery is {certificate_minus_logit_recovery:.1f} points higher, while "
            f"its audit bound is {certificate_minus_logit_bound:.1f} points looser. Labeled "
            "QAT has the strongest mean recovery when included. The task-tuned objective "
            "supports a distinct recovery advantage, but not universal bound dominance or "
            "a five-point certificate. This remains development rather than sequestered evidence."
            if args.include_qat
            else "Restricted label-free calibration clears 70 percent recovery for every "
            f"seed and beats logit-only in {certificate_logit_win_count}/5 paired seeds. "
            f"Its mean recovery is {certificate_minus_logit_recovery:.1f} points higher, "
            f"while its audit bound is {certificate_minus_logit_bound:.1f} points looser. "
            "This supports a distinct recovery advantage, but not universal bound dominance "
            "or a five-point certificate. The result remains development evidence."
        ),
        "input_report_references": report_descriptions,
        "rows_csv_reference": file_reference(rows_path),
        "code_revision": code_revision(root),
    }
    write_json(summary_path, summary)

    labels = ("Cert-directed", "Logit-only", "Global threshold") + (
        ("QAT (97 labels)",) if args.include_qat else ()
    ) + (
        ("From scratch (97 labels)",) if args.include_supervised else ()
    )
    colors = ("#7570b3", "#1b9e77", "#d95f02") + (
        ("#1f78b4",) if args.include_qat else ()
    ) + (
        ("#e7298a",) if args.include_supervised else ()
    )
    recovery = []
    recovery_error = []
    bounds = []
    bound_error = []
    for method in methods:
        selected = method_rows[method]
        recovery_values = np.asarray(
            [float(row["accuracy_recovery_fraction"]) for row in selected]
        )
        bound_values = np.asarray(
            [float(row["after_certificate_upper_bound"]) for row in selected]
        )
        recovery.append(np.mean(recovery_values) * 100)
        recovery_error.append(np.std(recovery_values, ddof=1) * 100)
        bounds.append(np.mean(bound_values) * 100)
        bound_error.append(np.std(bound_values, ddof=1) * 100)
    positions = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].bar(
        positions, recovery, yerr=recovery_error, color=colors, capsize=3
    )
    axes[0].axhline(70, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Lost accuracy recovered (%)")
    axes[0].set_title("DVS fixed-point repair")
    axes[1].bar(positions, bounds, yerr=bound_error, color=colors, capsize=3)
    axes[1].axhline(5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Simultaneous 95% upper bound (points)")
    axes[1].set_title("Decision identity remains uncertified")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
