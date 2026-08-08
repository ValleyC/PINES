from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / "branch_set_feasibility_summary.json"
    figure_pdf = root / "paper" / "figures" / "shd_branch_set_feasibility.pdf"
    figure_png = root / "paper" / "figures" / "shd_branch_set_feasibility.png"
    if any(path.exists() for path in (summary_path, figure_pdf, figure_png)):
        raise FileExistsError("branch-set feasibility destination already exists")

    full_paths = (
        root
        / "artifacts"
        / "shd_v6_branch_set_partition"
        / "seed_1701"
        / "branch_set_partition.json",
        root
        / "artifacts"
        / "shd_v7_branch_set_partition"
        / "seed_1701"
        / "branch_set_partition.json",
        root
        / "artifacts"
        / "shd_v8_branch_set_partition"
        / "seed_1701"
        / "branch_set_partition.json",
        root
        / "artifacts"
        / "shd_v9_branch_set_partition"
        / "seed_1701"
        / "branch_set_partition.json",
    )
    cell_paths = {
        mode: root
        / "artifacts"
        / "shd_v10_branch_set_cells"
        / "seed_1701"
        / f"branch_set_cells_{mode}.json"
        for mode in ("merged", "unmerged")
    }
    full_rows = []
    input_hashes = {}
    for path in full_paths:
        report = _load(path)
        if report["schema_version"] != "SHDBranchSetPartitionDiagnostic/v1":
            raise ValueError(f"unexpected schema: {path}")
        for row in report["rows"]:
            full_rows.append(
                {
                    **row,
                    "max_branches": report["max_branches"],
                    "sample_count": report["sample_count"],
                }
            )
        input_hashes[str(path.relative_to(root))] = sha256_file(path)

    cell_rows = []
    for mode, path in cell_paths.items():
        report = _load(path)
        if report["schema_version"] != "SHDBranchSetCellSweep/v1":
            raise ValueError(f"unexpected schema: {path}")
        for row in report["rows"]:
            cell_rows.append({"mode": mode, **row})
        input_hashes[str(path.relative_to(root))] = sha256_file(path)

    cell_aggregate = []
    for mode in ("merged", "unmerged"):
        for partitions in (32, 64, 128):
            selected = [
                row
                for row in cell_rows
                if row["mode"] == mode
                and int(row["partitions_per_axis"]) == partitions
            ]
            cell_aggregate.append(
                {
                    "mode": mode,
                    "partitions_per_axis": partitions,
                    "representative_cells": len(selected),
                    "samples_per_cell": int(selected[0]["samples"]),
                    "complete_cell_input_fraction": float(
                        np.mean([float(row["complete_fraction"]) for row in selected])
                    ),
                    "median_of_cell_median_abort_steps": float(
                        np.median([float(row["median_abort_step"]) for row in selected])
                    ),
                    "maximum_state_branches_observed": int(
                        max(int(row["maximum_state_branches_observed"]) for row in selected)
                    ),
                }
            )

    summary = {
        "schema_version": "SHDBranchSetFeasibilityAggregate/v1",
        "status": (
            "single-seed staged kill diagnostic; every reported certificate coverage "
            "is sound, while cap-limited runs are inconclusive"
        ),
        "seed": 1701,
        "relative_radius": 0.01,
        "horizon": 50,
        "full_cover_attempt_rows": full_rows,
        "representative_cell_rows": cell_rows,
        "representative_cell_aggregate": cell_aggregate,
        "route_assessment": {
            "any_complete_full_cover_input": any(
                int(row["all_cells_complete_inputs"]) > 0 for row in full_rows
            ),
            "any_complete_representative_cell_input_at_cap_65536": any(
                float(row["complete_fraction"]) > 0 for row in cell_rows
            ),
            "largest_partition_tested": 128,
            "subboxes_at_largest_partition": 128 * 128,
            "largest_branch_cap_tested": 65536,
            "best_median_abort_step": max(
                float(row["median_abort_step"]) for row in cell_rows
            ),
        },
        "interpretation": (
            "Separating spike-vector branches is sound but not computationally viable in "
            "this implementation. Finer parameter cells delay the state explosion, yet "
            "neither conservative merging nor retaining every state completes representative "
            "50-step cells with a 65,536-state cap. A successful analyzer needs stronger "
            "correlation or symbolic guard compression, not a larger explicit branch list."
        ),
        "input_report_hashes": input_hashes,
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    for mode, color, marker in (
        ("merged", "#7570b3", "o"),
        ("unmerged", "#d95f02", "s"),
    ):
        selected = [row for row in cell_aggregate if row["mode"] == mode]
        axis.plot(
            [row["partitions_per_axis"] for row in selected],
            [row["median_of_cell_median_abort_steps"] for row in selected],
            color=color,
            marker=marker,
            label=f"{mode}, cap 65,536",
        )
    axis.axhline(50, linestyle="--", color="0.35", label="complete horizon")
    axis.set_xscale("log", base=2)
    axis.set_xticks((32, 64, 128), ("32", "64", "128"))
    axis.set_ylim(0, 53)
    axis.set_xlabel("Uniform partitions per parameter axis")
    axis.set_ylabel("Median abort timestep")
    axis.set_title("Explicit branch sets remain incomplete")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
