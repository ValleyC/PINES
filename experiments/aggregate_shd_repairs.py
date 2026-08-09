from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-root", default="artifacts/shd_v1_repairs_frozen")
    parser.add_argument("--output-root", default="results/shd_v1")
    parser.add_argument("--condition", default="reset_to_value")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    methods = ("certificate_directed", "logit_only", "global_threshold")
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "repair_reset_summary.json"
    rows_path = output / "repair_reset_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_repair_comparison.pdf"
    figure_png = root / "paper" / "figures" / "shd_repair_comparison.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("repair aggregate destination exists; evidence is immutable")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    for seed in seeds:
        for method in methods:
            path = (
                root
                / args.repair_root
                / f"seed_{seed}"
                / args.condition
                / method
                / "repair_report.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report["calibration_audit_disjoint"]:
                raise ValueError(f"calibration/audit overlap in {path}")
            row = {
                "seed": seed,
                "method": method,
                "condition": report["condition"],
                "before_accuracy_loss": report["before"]["accuracy_loss"],
                "after_accuracy_loss": report["after"]["accuracy_loss"],
                "accuracy_recovery_fraction": report["accuracy_recovery_fraction"],
                "before_audit_disagreement_rate": report["before"][
                    "audit_disagreement_rate"
                ],
                "after_audit_disagreement_rate": report["after"][
                    "audit_disagreement_rate"
                ],
                "post_repair_upper_bound": report["after"][
                    "certificate_upper_bound"
                ],
                "calibration_samples": report["calibration_samples"],
                "audit_samples": report["audit_samples"],
                "label_budget": report["label_budget"],
                "elapsed_seconds": report["elapsed_seconds"],
                "source_model_hash": report["source_model_hash"],
                "repaired_model_hash": report["repaired_model_hash"],
                "report_code_revision": report["code_revision"],
            }
            rows.append(row)
            report_hashes[f"seed_{seed}_{method}"] = sha256_file(path)

    fields = tuple(rows[0].keys())
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    per_method: list[dict[str, object]] = []
    method_rows: dict[str, list[dict[str, object]]] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        method_rows[method] = selected
        recovery = np.asarray(
            [float(row["accuracy_recovery_fraction"]) for row in selected]
        )
        after_loss = np.asarray(
            [float(row["after_accuracy_loss"]) for row in selected]
        )
        disagreement = np.asarray(
            [float(row["after_audit_disagreement_rate"]) for row in selected]
        )
        bound = np.asarray(
            [float(row["post_repair_upper_bound"]) for row in selected]
        )
        elapsed = np.asarray([float(row["elapsed_seconds"]) for row in selected])
        per_method.append(
            {
                "method": method,
                "accuracy_recovery_mean": float(np.mean(recovery)),
                "accuracy_recovery_std": float(np.std(recovery, ddof=1)),
                "after_accuracy_loss_mean": float(np.mean(after_loss)),
                "after_audit_disagreement_mean": float(np.mean(disagreement)),
                "post_repair_upper_bound_mean": float(np.mean(bound)),
                "post_repair_upper_bound_min": float(np.min(bound)),
                "post_repair_upper_bound_max": float(np.max(bound)),
                "mean_elapsed_seconds": float(np.mean(elapsed)),
                "seeds_recovering_at_least_70_percent": int(
                    np.count_nonzero(recovery >= 0.70)
                ),
                "seeds_certified_at_five_point_budget": int(
                    np.count_nonzero(bound <= 0.05)
                ),
            }
        )

    recovery_by_method = {
        method: np.asarray(
            [float(row["accuracy_recovery_fraction"]) for row in method_rows[method]]
        )
        for method in methods
    }
    bound_by_method = {
        method: np.asarray(
            [float(row["post_repair_upper_bound"]) for row in method_rows[method]]
        )
        for method in methods
    }
    certificate_recovery = recovery_by_method["certificate_directed"]
    summary = {
        "schema_version": "SHDRepairAggregate/v1",
        "status": "five-seed software repair evidence; not physical evidence",
        "condition": args.condition,
        "seeds": list(seeds),
        "calibration_samples_per_seed": int(rows[0]["calibration_samples"]),
        "audit_samples_per_seed": int(rows[0]["audit_samples"]),
        "label_budget": 0,
        "per_method": per_method,
        "paired_comparisons": {
            "certificate_minus_logit_recovery_mean": float(
                np.mean(
                    certificate_recovery - recovery_by_method["logit_only"]
                )
            ),
            "certificate_minus_global_recovery_mean": float(
                np.mean(
                    certificate_recovery - recovery_by_method["global_threshold"]
                )
            ),
            "certificate_minus_logit_bound_mean": float(
                np.mean(
                    bound_by_method["certificate_directed"]
                    - bound_by_method["logit_only"]
                )
            ),
        },
        "gate_assessment": {
            "certificate_directed_mean_recovery_at_least_70_percent": bool(
                np.mean(certificate_recovery) >= 0.70
            ),
            "certificate_directed_beats_global_threshold_mean_recovery": bool(
                np.mean(certificate_recovery)
                > np.mean(recovery_by_method["global_threshold"])
            ),
            "certificate_directed_beats_logit_only_mean_recovery": bool(
                np.mean(certificate_recovery)
                > np.mean(recovery_by_method["logit_only"])
            ),
            "certificate_directed_certifies_five_point_budget": bool(
                np.all(bound_by_method["certificate_directed"] <= 0.05)
            ),
            "calibration_audit_disjoint": True,
        },
        "formulation_assessment": (
            "Certificate-directed repair restores more held-out accuracy than global threshold "
            "scaling and slightly more than logit-only distillation, but its untouched-audit "
            "disagreement bound remains vacuous and is looser than logit-only. Accuracy recovery "
            "does not imply restored decision identity."
        ),
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    labels = ("Certificate-directed", "Logit-only", "Global threshold")
    colors = ("#7570b3", "#1b9e77", "#d95f02")
    positions = np.arange(len(seeds))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for index, method in enumerate(methods):
        offset = (index - 1) * width
        axes[0].bar(
            positions + offset,
            recovery_by_method[method] * 100,
            width=width,
            label=labels[index],
            color=colors[index],
        )
        axes[1].bar(
            positions + offset,
            bound_by_method[method] * 100,
            width=width,
            color=colors[index],
        )
    axes[0].axhline(70, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Lost accuracy recovered (%)")
    axes[0].set_title("Accuracy recovery clears the target")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Post-repair 95% upper bound (points)")
    axes[1].set_title("But no repair certifies a 5-point budget")
    for axis in axes:
        axis.set_xticks(positions, [str(seed) for seed in seeds])
        axis.set_xlabel("Training seed")
        axis.grid(axis="y", alpha=0.2)
    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
