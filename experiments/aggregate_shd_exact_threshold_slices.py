from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = (
        root
        / "artifacts/shd_v45_exact_threshold_slices/seed_1701/"
        "reset_to_value_exact_threshold_slices.json"
    )
    output_path = root / "results/shd_v1/exact_threshold_slices_summary.json"
    with input_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    rows = report["rows"]
    cell_counts = np.asarray(
        [len(row["result"]["cells"]) for row in rows], dtype=np.int64
    )
    trace_counts = np.asarray(
        [row["result"]["unique_trace_count"] for row in rows], dtype=np.int64
    )
    counterexamples = [
        row for row in rows if row["result"]["counterexample_scale"] is not None
    ]
    summary = {
        "schema_version": "SHDExactThresholdSlicesSummary/v1",
        "source_report": str(input_path.relative_to(root)).replace("\\", "/"),
        "source_report_hash": sha256_file(input_path),
        "source_code_revision": report["code_revision"],
        "code_revision": code_revision(root),
        "seed": report["seed"],
        "condition": report["condition"],
        "relative_radius": report["relative_radius"],
        "selected_indices": report["selected_indices"],
        "timestep_slice_count": report["timestep_slice_count"],
        "threshold_scale_scope": (
            "every representable binary64 value in the closed interval"
        ),
        "certified_slices": report["certified_slices"],
        "conclusive_slices": report["conclusive_slices"],
        "all_slices_certified": report["all_slices_certified"],
        "counterexample_count": len(counterexamples),
        "total_exact_cells": report["total_exact_cells"],
        "minimum_cells_per_slice": int(np.min(cell_counts)),
        "mean_cells_per_slice": float(np.mean(cell_counts)),
        "maximum_cells_per_slice": int(np.max(cell_counts)),
        "minimum_unique_traces_per_slice": int(np.min(trace_counts)),
        "mean_unique_traces_per_slice": float(np.mean(trace_counts)),
        "maximum_unique_traces_per_slice": int(np.max(trace_counts)),
        "seconds": report["seconds"],
        "route_assessment": {
            "threshold_boundary_relaxation_removed_on_all_slices": bool(
                report["all_slices_certified"]
            ),
            "claim_joint_continuous_certificate": False,
            "advance_to_more_timestep_grid_slices": False,
            "advance_to_joint_timestep_threshold_oracle": True,
            "advance_to_five_seeds": False,
        },
        "interpretation": (
            "At each of 257 fixed timestep factors, exact trace-cell enumeration "
            "covers every binary64 threshold scale in the plus/minus one-percent "
            "range and preserves the reference class. This removes threshold-only "
            "guard relaxation as the cause of the polygonal residue at those "
            "slices. It is not a proof between timestep slices, so no joint "
            "continuous-family certificate is issued."
        ),
    }
    write_json_immutable(output_path, summary)


if __name__ == "__main__":
    main()
