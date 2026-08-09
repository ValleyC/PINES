from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import (
    code_revision,
    sha256_file,
    write_json_immutable,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/shd_cartesian_margin_profile_v1.json",
    )
    parser.add_argument(
        "--output",
        default="results/shd_v1/cartesian_margin_profile_v1_summary.json",
    )
    parser.add_argument(
        "--rows", default="results/shd_v1/cartesian_margin_profile_v1_rows.csv"
    )
    parser.add_argument(
        "--figure", default="paper/figures/shd_cartesian_margin_profile_v1.pdf"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    profile = _load(config_path)
    if profile.get("schema_version") != "SHDCartesianMarginProfile/v1":
        raise ValueError("unsupported Cartesian margin-profile configuration")

    source_reports: dict[str, str] = {}
    source_verifications: dict[str, str] = {}
    source_configs: dict[str, str] = {}
    input_summaries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    reference_budgets: tuple[int, ...] | None = None
    reference_members: tuple[str, ...] | None = None

    for experiment in profile["experiments"]:
        report_path = root / experiment["report"]
        verification_path = root / experiment["verification"]
        experiment_config_path = root / experiment["config"]
        report = _load(report_path)
        verification = _load(verification_path)
        experiment_config = _load(experiment_config_path)
        if report.get("schema_version") != "SHDHybridCartesianMemberScalingResult/v1":
            raise ValueError(f"unsupported member report {report_path}")
        if verification.get("schema_version") != (
            "SHDHybridCartesianMemberScalingArtifactVerification/v1"
        ):
            raise ValueError(f"unsupported verification {verification_path}")
        if verification.get("report_hash") != sha256_file(report_path):
            raise ValueError(f"report verification hash mismatch for {report_path}")
        required_checks = (
            "all_member_rows_match_shards",
            "all_members_verified",
            "all_shard_hashes_match",
            "scientific_sources_match_execution_manifest",
        )
        if not all(verification.get(check) is True for check in required_checks):
            raise ValueError(f"verification did not pass for {report_path}")
        if verification.get("config_hash") != sha256_file(experiment_config_path):
            raise ValueError(f"config verification hash mismatch for {report_path}")
        if report.get("config_hash") != sha256_file(experiment_config_path):
            raise ValueError(f"report config hash mismatch for {report_path}")

        selection_path = root / report["selection_report"]
        if report["selection_report_hash"] != sha256_file(selection_path):
            raise ValueError(f"selection hash mismatch for {report_path}")
        selection = _load(selection_path)
        selection_field = experiment_config["selection"]["field"]
        selected = selection[selection_field]
        if int(selected["dataset_index"]) != int(report["dataset_index"]):
            raise ValueError(f"selected dataset index mismatch for {report_path}")

        members = sorted(report["member_rows"], key=lambda row: int(row["member_index"]))
        if len(members) != 16:
            raise ValueError(f"expected 16 members in {report_path}")
        member_hashes = tuple(row["semantics_hash"] for row in members)
        if reference_members is None:
            reference_members = member_hashes
        elif member_hashes != reference_members:
            raise ValueError("Cartesian member identities differ across inputs")
        budgets = tuple(
            int(row["maximum_polygon_leaves"])
            for row in members[0]["budget_rows"]
        )
        if reference_budgets is None:
            reference_budgets = budgets
        elif budgets != reference_budgets:
            raise ValueError("certificate budgets differ across inputs")

        input_rows: list[dict[str, object]] = []
        for budget_index, budget in enumerate(budgets):
            budget_rows = [member["budget_rows"][budget_index] for member in members]
            fractions = np.asarray(
                [float(row["certified_parameter_fraction"]) for row in budget_rows],
                dtype=np.float64,
            )
            complete_count = int(sum(bool(row["certified"]) for row in budget_rows))
            row = {
                "label": experiment["label"],
                "quantile": float(experiment["quantile"]),
                "dataset_index": int(report["dataset_index"]),
                "minimum_reference_margin": float(selected["minimum_reference_margin"]),
                "maximum_polygon_leaves": budget,
                "mean_certified_parameter_fraction": float(fractions.mean()),
                "median_certified_parameter_fraction": float(np.median(fractions)),
                "minimum_certified_parameter_fraction": float(fractions.min()),
                "maximum_certified_parameter_fraction": float(fractions.max()),
                "fully_certified_member_count": complete_count,
                "full_family_certified": complete_count == len(members),
                "total_branch_cap_hits": int(
                    sum(int(member_row["branch_cap_hits"]) for member_row in budget_rows)
                ),
                "total_unresolved_leaves": int(
                    sum(int(member_row["unresolved_leaves"]) for member_row in budget_rows)
                ),
                "total_affine_certified_leaves": int(
                    sum(
                        int(member_row["affine_certified_leaves"])
                        for member_row in budget_rows
                    )
                ),
                "total_branch_certified_leaves": int(
                    sum(
                        int(member_row["branch_certified_leaves"])
                        for member_row in budget_rows
                    )
                ),
                "any_branch_prediction_rejection": any(
                    int(member_row["branch_prediction_rejections"]) > 0
                    for member_row in budget_rows
                ),
                "total_member_seconds": float(
                    sum(float(member_row["seconds"]) for member_row in budget_rows)
                ),
                "maximum_member_seconds": float(
                    max(float(member_row["seconds"]) for member_row in budget_rows)
                ),
            }
            rows.append(row)
            input_rows.append(row)
        earliest_complete = next(
            (
                int(row["maximum_polygon_leaves"])
                for row in input_rows
                if row["full_family_certified"]
            ),
            None,
        )
        input_summaries.append(
            {
                "label": experiment["label"],
                "quantile": float(experiment["quantile"]),
                "dataset_index": int(report["dataset_index"]),
                "minimum_reference_margin": float(selected["minimum_reference_margin"]),
                "earliest_complete_budget": earliest_complete,
                "report": experiment["report"],
                "report_hash": sha256_file(report_path),
                "verification": experiment["verification"],
                "verification_hash": sha256_file(verification_path),
            }
        )
        source_reports[experiment["report"]] = sha256_file(report_path)
        source_verifications[experiment["verification"]] = sha256_file(
            verification_path
        )
        source_configs[experiment["config"]] = sha256_file(experiment_config_path)

    if reference_budgets is None:
        raise ValueError("profile contains no experiments")
    input_summaries.sort(key=lambda row: float(row["quantile"]))
    rows.sort(key=lambda row: (float(row["quantile"]), int(row["maximum_polygon_leaves"])))

    rows_path = root / args.rows
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    if rows_path.exists():
        raise FileExistsError(f"immutable rows artifact exists: {rows_path}")
    with rows_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = root / args.figure
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    if figure_path.exists():
        raise FileExistsError(f"immutable figure artifact exists: {figure_path}")
    figure_png = figure_path.with_suffix(".png")
    if figure_png.exists():
        raise FileExistsError(f"immutable figure artifact exists: {figure_png}")
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharex=True, sharey=True)
    colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(input_summaries)))
    for summary, color in zip(input_summaries, colors, strict=True):
        input_rows = [
            row for row in rows if int(row["dataset_index"]) == int(summary["dataset_index"])
        ]
        budgets = [int(row["maximum_polygon_leaves"]) for row in input_rows]
        label = (
            f"q={float(summary['quantile']):.1f}, "
            f"margin={float(summary['minimum_reference_margin']):.2f}"
        )
        axes[0].plot(
            budgets,
            [100.0 * float(row["mean_certified_parameter_fraction"]) for row in input_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        axes[1].plot(
            budgets,
            [100.0 * float(row["minimum_certified_parameter_fraction"]) for row in input_rows],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axes[0].set_title("Mean area across 16 members")
    axes[1].set_title("Worst-member proved area")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(reference_budgets, [str(value) for value in reference_budgets])
        axis.set_xlabel("Maximum leaves per member")
        axis.set_ylim(-2.0, 102.0)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Certified parameter area (%)")
    axes[1].axhline(100.0, color="black", linestyle="--", linewidth=1)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    figure.savefig(figure_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    budget_summaries = []
    for budget in reference_budgets:
        budget_rows = [row for row in rows if row["maximum_polygon_leaves"] == budget]
        budget_summaries.append(
            {
                "maximum_polygon_leaves": budget,
                "fully_certified_input_count": int(
                    sum(bool(row["full_family_certified"]) for row in budget_rows)
                ),
                "input_count": len(budget_rows),
                "mean_of_member_mean_certified_parameter_fraction": float(
                    np.mean(
                        [row["mean_certified_parameter_fraction"] for row in budget_rows]
                    )
                ),
                "minimum_worst_member_certified_parameter_fraction": float(
                    min(row["minimum_certified_parameter_fraction"] for row in budget_rows)
                ),
                "total_branch_cap_hits": int(
                    sum(int(row["total_branch_cap_hits"]) for row in budget_rows)
                ),
                "total_unresolved_leaves": int(
                    sum(int(row["total_unresolved_leaves"]) for row in budget_rows)
                ),
                "any_branch_prediction_rejection": any(
                    bool(row["any_branch_prediction_rejection"]) for row in budget_rows
                ),
            }
        )

    output = {
        "schema_version": "SHDCartesianMarginProfileAggregate/v1",
        "status": profile["status"],
        "interpretation": (
            "Margin-stratified label-free development inputs test analyzer "
            "tractability beyond the maximum-margin input. The selected set is "
            "not random and cannot estimate population certificate coverage."
        ),
        "code_revision": code_revision(root),
        "profile_config": args.config,
        "profile_config_hash": sha256_file(config_path),
        "aggregate_source_hash": sha256_file(Path(__file__).resolve()),
        "source_reports": source_reports,
        "source_verifications": source_verifications,
        "source_configs": source_configs,
        "input_summaries": input_summaries,
        "budget_summaries": budget_summaries,
        "rows_csv": args.rows,
        "rows_csv_hash": sha256_file(rows_path),
        "figure_pdf": args.figure,
        "figure_pdf_hash": sha256_file(figure_path),
        "figure_png": str(figure_png.relative_to(root)).replace("\\", "/"),
        "figure_png_hash": sha256_file(figure_png),
    }
    output_path = root / args.output
    write_json_immutable(output_path, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
