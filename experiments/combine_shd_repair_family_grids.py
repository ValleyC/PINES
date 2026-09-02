from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, file_reference, write_json


METHODS = (
    "no_repair",
    "certificate_directed",
    "logit_only",
    "global_threshold",
    "per_platform_qat",
)


def _load_summary(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "SHDRepairFamilyGridAggregate/v3":
        raise ValueError(f"unsupported repair-family aggregate: {path}")
    rows = {row["method"]: row for row in report["aggregate_rows"]}
    if tuple(rows) != METHODS:
        raise ValueError(f"method order mismatch: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset-summary",
        default=(
            "results/shd_v1/"
            "shd_repair_family_grid_full_audit_reset_clustered_v3_summary.json"
        ),
    )
    parser.add_argument(
        "--floor-summary",
        default=(
            "results/shd_v1/"
            "shd_repair_family_grid_full_audit_floor_clustered_v3_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/shd_v1/shd_repair_family_grid_combined_v1.json",
    )
    parser.add_argument(
        "--figure", default="paper/figures/shd_repair_family_grid_combined_v1.pdf"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = {
        "reset_to_value": root / args.reset_summary,
        "floor_rounding_saturation": root / args.floor_summary,
    }
    reports = {condition: _load_summary(path) for condition, path in paths.items()}
    designs = {
        (
            tuple(report["seeds"]),
            int(report["sample_count_per_seed"]),
            float(report["relative_radius"]),
            int(report["grid_resolution"]),
        )
        for report in reports.values()
    }
    if len(designs) != 1:
        raise ValueError("repair-family conditions do not share one audit design")

    route = {
        condition: report["route_assessment"] for condition, report in reports.items()
    }
    combined = {
        "schema_version": "SHDRepairFamilyGridCombined/v1",
        "status": (
            "label-free finite-grid full-audit diagnostic with input-cluster and "
            "training-seed uncertainty; not a continuous-family certificate"
        ),
        "conditions": list(reports),
        "design": {
            "seeds": reports["reset_to_value"]["seeds"],
            "sample_count_per_seed": reports["reset_to_value"][
                "sample_count_per_seed"
            ],
            "relative_radius": reports["reset_to_value"]["relative_radius"],
            "grid_resolution": reports["reset_to_value"]["grid_resolution"],
            "uses_labels": False,
        },
        "condition_summaries": {
            condition: {
                "aggregate_rows": report["aggregate_rows"],
                "route_assessment": report["route_assessment"],
            }
            for condition, report in reports.items()
        },
        "gates": {
            "certificate_directed_gain_positive_every_seed_both_conditions": all(
                row["certificate_directed_gain_positive_every_seed"]
                for row in route.values()
            ),
            "certificate_directed_gain_cluster_interval_positive_both_conditions": all(
                next(
                    item
                    for item in reports[condition]["aggregate_rows"]
                    if item["method"] == "certificate_directed"
                )["paired_gain_input_cluster_bootstrap_95_percent_interval"][0]
                > 0
                for condition in reports
            ),
            "certificate_directed_beats_logit_only_mean_both_conditions": all(
                row["certificate_directed_beats_logit_only_mean"]
                for row in route.values()
            ),
            "certificate_directed_vs_logit_interval_excludes_zero_any_condition": any(
                interval[0] > 0 or interval[1] < 0
                for interval in (
                    row[
                        "certificate_directed_minus_logit_only_input_cluster_bootstrap_95_percent_interval"
                    ]
                    for row in route.values()
                )
            ),
        },
        "source_references": {
            condition: file_reference(path) for condition, path in paths.items()
        },
        "code_revision": code_revision(root),
        "interpretation": (
            "Certificate-directed repair produces a positive family-identity gain in "
            "every seed and both conditions, with positive input-cluster intervals. "
            "Its paired difference from logit-only includes zero in both conditions, "
            "so these data support restoration but not objective superiority."
        ),
    }
    output_path = root / args.output
    write_json(output_path, combined)

    labels = {
        "no_repair": "no repair",
        "certificate_directed": "certificate-\ndirected",
        "logit_only": "logit-only",
        "global_threshold": "global\nthreshold",
        "per_platform_qat": "labeled\nfine-tune",
    }
    titles = {
        "reset_to_value": "Reset-to-value family",
        "floor_rounding_saturation": "Floor fixed-point family",
    }
    colors = ["0.65", "#d95f02", "#1b9e77", "#7570b3", "#e6ab02"]
    x = np.arange(len(METHODS), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True, constrained_layout=True)
    for axis, condition in zip(axes, reports, strict=True):
        by_method = {
            row["method"]: row for row in reports[condition]["aggregate_rows"]
        }
        means = np.asarray(
            [by_method[method]["grid_family_identity_fraction"]["mean"] for method in METHODS]
        )
        cluster_intervals = np.asarray(
            [
                by_method[method]["input_cluster_bootstrap_95_percent_interval"]
                for method in METHODS
            ]
        )
        seed_intervals = np.asarray(
            [by_method[method]["training_seed_t_95_percent_interval"] for method in METHODS]
        )
        axis.bar(x, 100 * means, color=colors, width=0.72)
        axis.errorbar(
            x - 0.055,
            100 * means,
            yerr=100
            * np.stack(
                [means - cluster_intervals[:, 0], cluster_intervals[:, 1] - means]
            ),
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1.1,
            label="input-cluster 95% interval",
        )
        axis.errorbar(
            x + 0.055,
            100 * means,
            yerr=100
            * np.stack([means - seed_intervals[:, 0], seed_intervals[:, 1] - means]),
            fmt="none",
            ecolor="#377eb8",
            capsize=3,
            linewidth=1.1,
            label="training-seed 95% interval",
        )
        axis.set_title(titles[condition])
        axis.set_xticks(x, [labels[method] for method in METHODS], fontsize=8)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Identity to source/reference over 9x9 family (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    figure_path = root / args.figure
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(combined["gates"], indent=2))


if __name__ == "__main__":
    main()
