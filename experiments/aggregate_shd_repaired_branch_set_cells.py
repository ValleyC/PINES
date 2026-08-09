from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


METHODS = ("no_repair", "certificate_directed", "logit_only")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = (
        root
        / "artifacts"
        / "shd_v13_repaired_branch_cells"
        / "seed_1701"
        / "reset_to_value_branch_cells.json"
    )
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / "repaired_branch_set_cells_summary.json"
    figure_pdf = root / "paper" / "figures" / "shd_repaired_branch_set_cells.pdf"
    figure_png = root / "paper" / "figures" / "shd_repaired_branch_set_cells.png"
    if any(path.exists() for path in (summary_path, figure_pdf, figure_png)):
        raise FileExistsError("repaired branch-cell aggregate destination already exists")
    report = json.loads(source_path.read_text(encoding="utf-8"))
    if report["schema_version"] != "SHDRepairedBranchSetCellDiagnostic/v1":
        raise ValueError("unexpected repaired branch-cell report schema")

    aggregate_rows = []
    for method in METHODS:
        for partitions in report["partitions"]:
            selected = [
                row
                for row in report["rows"]
                if row["method"] == method
                and int(row["partitions_per_axis"]) == int(partitions)
            ]
            abort_steps = np.asarray(
                [float(row["median_abort_step"]) for row in selected], dtype=float
            )
            aggregate_rows.append(
                {
                    "method": method,
                    "partitions_per_axis": int(partitions),
                    "merge_cell_combinations": len(selected),
                    "samples_per_cell": int(selected[0]["samples"]),
                    "complete_cell_input_count": int(
                        sum(int(row["complete_inputs"]) for row in selected)
                    ),
                    "mean_cell_median_abort_step": float(np.mean(abort_steps)),
                    "min_cell_median_abort_step": float(np.min(abort_steps)),
                    "max_cell_median_abort_step": float(np.max(abort_steps)),
                    "maximum_state_branches_observed": int(
                        max(int(row["maximum_state_branches_observed"]) for row in selected)
                    ),
                }
            )

    finest = {
        row["method"]: row
        for row in aggregate_rows
        if row["partitions_per_axis"] == max(report["partitions"])
    }
    summary = {
        "schema_version": "SHDRepairedBranchSetCellAggregate/v1",
        "status": (
            "single-seed representative-cell kill diagnostic; cap failures are "
            "inconclusive and cells do not cover the family"
        ),
        "seed": report["seed"],
        "condition": report["condition"],
        "relative_radius": report["relative_radius"],
        "max_branches": report["max_branches"],
        "aggregate_rows": aggregate_rows,
        "route_assessment": {
            "any_complete_cell_input": any(
                row["complete_cell_input_count"] > 0 for row in aggregate_rows
            ),
            "finest_no_repair_mean_abort_step": finest["no_repair"][
                "mean_cell_median_abort_step"
            ],
            "finest_certificate_directed_mean_abort_step": finest[
                "certificate_directed"
            ]["mean_cell_median_abort_step"],
            "finest_logit_only_mean_abort_step": finest["logit_only"][
                "mean_cell_median_abort_step"
            ],
            "certificate_directed_improves_explicit_proof_depth": finest[
                "certificate_directed"
            ]["mean_cell_median_abort_step"]
            > finest["no_repair"]["mean_cell_median_abort_step"],
        },
        "interpretation": (
            "Both label-free repairs increase sampled family identity but reduce the depth "
            "reached by explicit sound branch analysis on these cells. The current repair "
            "objective recovers decisions; it does not restore static certifiability."
        ),
        "input_report_hash": sha256_file(source_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    labels = {
        "no_repair": "no repair",
        "certificate_directed": "certificate-directed",
        "logit_only": "logit-only",
    }
    colors = {
        "no_repair": "0.35",
        "certificate_directed": "#d95f02",
        "logit_only": "#1b9e77",
    }
    fig, axis = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    for method in METHODS:
        selected = [row for row in aggregate_rows if row["method"] == method]
        axis.plot(
            [row["partitions_per_axis"] for row in selected],
            [row["mean_cell_median_abort_step"] for row in selected],
            marker="o",
            color=colors[method],
            label=labels[method],
        )
    axis.axhline(50, linestyle="--", color="#7570b3", label="complete horizon")
    axis.set_xscale("log", base=2)
    axis.set_xticks((64, 128), ("64", "128"))
    axis.set_ylim(30, 52)
    axis.set_xlabel("Uniform partitions per parameter axis")
    axis.set_ylabel("Mean cell-median abort timestep")
    axis.set_title("Repair does not simplify explicit proof search")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
