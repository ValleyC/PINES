from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/shd_cartesian_q10_branch_cap_profile_v1.json",
    )
    parser.add_argument(
        "--output",
        default="results/shd_v1/cartesian_q10_branch_cap_profile_v1_summary.json",
    )
    parser.add_argument(
        "--rows",
        default="results/shd_v1/cartesian_q10_branch_cap_profile_v1_rows.csv",
    )
    parser.add_argument(
        "--figure",
        default="paper/figures/shd_cartesian_q10_branch_cap_profile_v1.pdf",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    profile_path = root / args.config
    profile = _load(profile_path)
    if profile.get("schema_version") != "SHDCartesianBranchCapProfile/v1":
        raise ValueError("unsupported branch-cap profile")

    rows: list[dict[str, object]] = []
    source_reports: dict[str, str] = {}
    source_verifications: dict[str, str] = {}
    source_configs: dict[str, str] = {}
    identity: tuple[object, ...] | None = None
    for experiment in profile["experiments"]:
        report_path = root / experiment["report"]
        verification_path = root / experiment["verification"]
        experiment_config_path = root / experiment["config"]
        report = _load(report_path)
        verification = _load(verification_path)
        experiment_config = _load(experiment_config_path)
        if report.get("schema_version") != "SHDHybridCartesianMemberScalingResult/v1":
            raise ValueError(f"unsupported report {report_path}")
        if verification.get("schema_version") != (
            "SHDHybridCartesianMemberScalingArtifactVerification/v1"
        ):
            raise ValueError(f"unsupported verification {verification_path}")
        if verification.get("report_hash") != sha256_file(report_path):
            raise ValueError(f"report hash mismatch for {report_path}")
        if verification.get("config_hash") != sha256_file(experiment_config_path):
            raise ValueError(f"config hash mismatch for {report_path}")
        required_checks = (
            "all_member_rows_match_shards",
            "all_members_verified",
            "all_shard_hashes_match",
            "scientific_sources_match_execution_manifest",
        )
        if not all(verification.get(check) is True for check in required_checks):
            raise ValueError(f"verification did not pass for {report_path}")
        declared_cap = int(experiment_config["certificate_budgets"]["maximum_local_branches"])
        if declared_cap != int(experiment["branch_cap"]):
            raise ValueError(f"declared branch cap mismatch for {report_path}")

        member_index = int(experiment["member_index"])
        members = [
            row for row in report["member_rows"] if int(row["member_index"]) == member_index
        ]
        if len(members) != 1:
            raise ValueError(f"member {member_index} is not unique in {report_path}")
        member = members[0]
        budget_rows = [
            row for row in member["budget_rows"] if int(row["maximum_polygon_leaves"]) == 256
        ]
        if len(budget_rows) != 1:
            raise ValueError(f"256-leaf row is not unique in {report_path}")
        budget = budget_rows[0]
        current_identity = (
            int(report["seed"]),
            int(report["dataset_index"]),
            member["model_hash"],
            member["semantics_hash"],
            member["box_hash"],
            report["selection_report_hash"],
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("branch-cap rows do not analyze one identical model/input/member/box")
        rows.append(
            {
                "maximum_local_branches": declared_cap,
                "maximum_polygon_leaves": 256,
                "certified": bool(budget["certified"]),
                "certified_parameter_fraction": float(
                    budget["certified_parameter_fraction"]
                ),
                "unresolved_parameter_fraction": float(
                    budget["unresolved_parameter_fraction"]
                ),
                "branch_cap_hits": int(budget["branch_cap_hits"]),
                "unresolved_leaves": int(budget["unresolved_leaves"]),
                "branch_certified_leaves": int(budget["branch_certified_leaves"]),
                "affine_certified_leaves": int(budget["affine_certified_leaves"]),
                "branch_prediction_rejections": int(
                    budget["branch_prediction_rejections"]
                ),
                "seconds": float(budget["seconds"]),
            }
        )
        source_reports[experiment["report"]] = sha256_file(report_path)
        source_verifications[experiment["verification"]] = sha256_file(
            verification_path
        )
        source_configs[experiment["config"]] = sha256_file(experiment_config_path)
    rows.sort(key=lambda row: int(row["maximum_local_branches"]))

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
    figure_png = figure_path.with_suffix(".png")
    if figure_path.exists() or figure_png.exists():
        raise FileExistsError("immutable branch-cap figure already exists")
    caps = [int(row["maximum_local_branches"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.35))
    axes[0].plot(
        caps,
        [100.0 * float(row["certified_parameter_fraction"]) for row in rows],
        color="#2a9d8f",
        marker="o",
        linewidth=2,
    )
    axes[0].axhline(100.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Certified parameter area (%)")
    axes[0].set_title("Sound area at 256 leaves")
    axes[1].plot(
        caps,
        [int(row["unresolved_leaves"]) for row in rows],
        color="#d95f02",
        marker="s",
        linewidth=2,
        label="unresolved leaves",
    )
    axes[1].set_ylabel("Unresolved leaves")
    axes[1].set_title("Residual proof complexity")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(caps, [str(value) for value in caps])
        axis.set_xlabel("Maximum local branches")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    figure.savefig(figure_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    output = {
        "schema_version": "SHDCartesianBranchCapProfileAggregate/v1",
        "status": profile["status"],
        "interpretation": (
            "The ablation holds the label-free selected input, model, semantics "
            "member, continuous box, polygon-leaf budget, and guard-cut budget "
            "fixed while varying only the local branch cap."
        ),
        "code_revision": code_revision(root),
        "profile_config": args.config,
        "profile_config_hash": sha256_file(profile_path),
        "aggregate_source_hash": sha256_file(Path(__file__).resolve()),
        "source_reports": source_reports,
        "source_verifications": source_verifications,
        "source_configs": source_configs,
        "rows": rows,
        "rows_csv": args.rows,
        "rows_csv_hash": sha256_file(rows_path),
        "figure_pdf": args.figure,
        "figure_pdf_hash": sha256_file(figure_path),
        "figure_png": str(figure_png.relative_to(root)).replace("\\", "/"),
        "figure_png_hash": sha256_file(figure_png),
    }
    write_json_immutable(root / args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
