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
    parser.add_argument(
        "--input-root", default="artifacts/shd_v1_continuous_radius_sweep"
    )
    parser.add_argument("--output-root", default="results/shd_v1")
    parser.add_argument(
        "--joint-partition-report",
        default=(
            "artifacts/shd_v1_partitioned_continuous/seed_1701/"
            "partitioned_continuous_report.json"
        ),
    )
    parser.add_argument(
        "--joint-p16-report",
        default=(
            "artifacts/shd_v1_partitioned_continuous_p16_64/seed_1701/"
            "partitioned_continuous_report.json"
        ),
    )
    parser.add_argument(
        "--timestep-partition-report",
        default=(
            "artifacts/shd_v1_partitioned_timestep/seed_1701/"
            "partitioned_continuous_report.json"
        ),
    )
    parser.add_argument(
        "--threshold-partition-report",
        default=(
            "artifacts/shd_v1_partitioned_threshold/seed_1701/"
            "partitioned_continuous_report.json"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    axes = ("timestep", "threshold", "joint")
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "continuous_radius_sweep_summary.json"
    rows_path = output / "continuous_radius_sweep_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_continuous_radius_sweep.pdf"
    figure_png = root / "paper" / "figures" / "shd_continuous_radius_sweep.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("continuous radius-sweep aggregate exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    for seed in seeds:
        path = root / args.input_root / f"seed_{seed}" / "radius_sweep_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["seed"]) != seed:
            raise ValueError("seed mismatch in radius-sweep report")
        report_hashes[f"seed_{seed}"] = sha256_file(path)
        rows.extend({"seed": seed, **row} for row in report["rows"])

    fields = tuple(rows[0].keys())
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    radii = tuple(sorted({float(row["relative_radius"]) for row in rows}))
    trajectories: dict[str, list[dict[str, object]]] = {}
    for varied_axis in axes:
        entries = []
        for radius in radii:
            selected = [
                row
                for row in rows
                if row["varied_axis"] == varied_axis
                and float(row["relative_radius"]) == radius
            ]
            coverage = np.asarray(
                [float(row["certified_fraction"]) for row in selected]
            )
            entries.append(
                {
                    "relative_radius": radius,
                    "certified_mean": float(np.mean(coverage)),
                    "certified_std": float(np.std(coverage, ddof=1)),
                    "minimum_seed_coverage": float(np.min(coverage)),
                    "maximum_seed_coverage": float(np.max(coverage)),
                }
            )
        trajectories[varied_axis] = entries

    partition_specs = {
        "joint_128_up_to_8": args.joint_partition_report,
        "joint_64_at_16": args.joint_p16_report,
        "timestep_128_up_to_8": args.timestep_partition_report,
        "threshold_128_up_to_8": args.threshold_partition_report,
    }
    partition_kill_checks: dict[str, dict[str, object]] = {}
    partition_hashes: dict[str, str] = {}
    for name, relative_path in partition_specs.items():
        path = root / relative_path
        report = json.loads(path.read_text(encoding="utf-8"))
        finest = report["rows"][-1]
        partition_hashes[name] = sha256_file(path)
        partition_kill_checks[name] = {
            "samples": int(report["audit_samples"]),
            "relative_radius": float(report["relative_radius"]),
            "varied_axes": report.get("varied_axes", "joint"),
            "finest_partitions_per_varied_axis": int(
                finest["partitions_per_axis"]
            ),
            "finest_subbox_count": int(finest["subbox_count"]),
            "finest_certified_fraction": float(finest["certified_fraction"]),
            "finest_seconds": float(finest["seconds"]),
        }

    target_subbox_radius = 1e-5
    partitions_per_axis = int(np.ceil(0.01 / target_subbox_radius))
    summary = {
        "schema_version": "SHDContinuousRadiusSweepAggregate/v1",
        "status": (
            "five-seed full-audit sound reference-member diagnostic plus a "
            "single-seed uniform-partition scalability kill check"
        ),
        "seeds": list(seeds),
        "audit_samples_per_seed": int(rows[0]["samples"]),
        "trajectories": trajectories,
        "partition_kill_checks": partition_kill_checks,
        "uniform_partition_scaling": {
            "target_subbox_relative_radius": target_subbox_radius,
            "partitions_per_axis_to_cover_plus_minus_one_percent": partitions_per_axis,
            "continuous_subboxes_for_joint_two_axis_grid": partitions_per_axis**2,
            "abstract_members_with_16_discrete_semantics": partitions_per_axis**2 * 16,
            "interpretation": (
                "The radius sweep first gives useful joint reference-member coverage "
                "near 1e-5. Uniformly covering plus/minus one percent at that radius "
                "requires one million continuous boxes before the 16 discrete members."
            ),
        },
        "gate_assessment": {
            "joint_reference_coverage_at_1e4_below_20_percent": bool(
                next(
                    item["certified_mean"]
                    for item in trajectories["joint"]
                    if item["relative_radius"] == 1e-4
                )
                < 0.20
            ),
            "uniform_partition_kill_checks_recover_any_input": any(
                item["finest_certified_fraction"] > 0
                for item in partition_kill_checks.values()
            ),
        },
        "formulation_assessment": (
            "The current interval domain has a five-seed coverage cliff between "
            "relative radii 1e-5 and 1e-4 even for the reference discrete semantics. "
            "Uniform partitioning cannot scale from this tolerance to a meaningful "
            "plus/minus one-percent family; a dependency-preserving abstraction is "
            "required."
        ),
        "input_report_hashes": report_hashes,
        "partition_report_hashes": partition_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    colors = {"timestep": "#1b9e77", "threshold": "#d95f02", "joint": "#7570b3"}
    labels = {"timestep": "Timestep only", "threshold": "Threshold only", "joint": "Joint"}
    fig, axis = plt.subplots(figsize=(6.7, 4.0), constrained_layout=True)
    for varied_axis in axes:
        entries = trajectories[varied_axis]
        x = np.asarray([float(item["relative_radius"]) for item in entries])
        mean = np.asarray([float(item["certified_mean"]) for item in entries]) * 100
        std = np.asarray([float(item["certified_std"]) for item in entries]) * 100
        axis.plot(x, mean, marker="o", color=colors[varied_axis], label=labels[varied_axis])
        axis.fill_between(x, mean - std, mean + std, color=colors[varied_axis], alpha=0.15)
    axis.axhline(20, color="black", linestyle="--", linewidth=1, label="20% gate")
    axis.set_xscale("log")
    axis.set_ylim(0, 102)
    axis.set_xlabel("Relative uncertainty radius")
    axis.set_ylabel("Soundly certified SHD inputs (%)")
    axis.set_title("Reference-member interval coverage collapses before 0.01% uncertainty")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
