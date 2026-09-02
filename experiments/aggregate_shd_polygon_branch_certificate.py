from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, file_reference, write_json


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cover_path = (
        root
        / "artifacts/shd_v50_rounding_guard_residuals/seed_1701/"
        "reset_to_value_guard_cuts.json"
    )
    fixed_trace_path = (
        root
        / "artifacts/shd_v54_residual_fixed_trace_all/seed_1701/"
        "reset_to_value_fixed_trace.json"
    )
    branch_path = (
        root
        / "artifacts/shd_v56_residual_polygon_branches_all/seed_1701/"
        "reset_to_value_polygon_branches.json"
    )
    cover = _load(cover_path)
    fixed_trace = _load(fixed_trace_path)
    branch = _load(branch_path)
    cover_row = cover["rows"][0]
    branch_rows = branch["rows"]
    if branch["residual_geometry_reference"] != cover["residual_polygon_artifact_reference"]:
        raise ValueError("branch result does not target the cover residual")
    if branch["box_description"] != cover["box_description"]:
        raise ValueError("branch and cover semantics boxes differ")
    if branch["selected_polygon_count"] != cover_row["unresolved_leaves"]:
        raise ValueError("branch result does not include every residual polygon")
    if fixed_trace["selected_polygon_count"] != cover_row["unresolved_leaves"]:
        raise ValueError("fixed-trace result does not include every residual polygon")

    source_fraction = float(cover_row["certified_parameter_fraction"])
    residual_fraction = float(cover_row["unresolved_parameter_fraction"])
    area_sum = source_fraction + residual_fraction
    area_cover_valid = bool(np.isclose(area_sum, 1.0, rtol=1e-9, atol=1e-9))
    residual_complete = all(row["complete"] for row in branch_rows)
    residual_certified = all(row["certified"] for row in branch_rows)
    full_certificate = area_cover_valid and residual_complete and residual_certified
    active_branches = np.asarray(
        [row["maximum_active_branches"] for row in branch_rows], dtype=np.int64
    )
    possible_predictions = sorted(
        {
            prediction
            for row in branch_rows
            for prediction in row["possible_predictions"]
        }
    )
    summary = {
        "schema_version": "SHDPolygonBranchCertificateSummary/v1",
        "code_revision": code_revision(root),
        "source_code_revisions": {
            "cover": cover["code_revision"],
            "fixed_trace": fixed_trace["code_revision"],
            "polygon_branches": branch["code_revision"],
        },
        "source_report_descriptions": {
            str(cover_path.relative_to(root)).replace("\\", "/"): file_reference(
                cover_path
            ),
            str(fixed_trace_path.relative_to(root)).replace(
                "\\", "/"
            ): file_reference(fixed_trace_path),
            str(branch_path.relative_to(root)).replace("\\", "/"): file_reference(
                branch_path
            ),
        },
        "seed": cover["seed"],
        "condition": cover["condition"],
        "dataset_index": int(cover_row["dataset_index"]),
        "scope": {
            "selection": "single finite-grid-stable development input",
            "integration": "forward Euler",
            "reset": "reset-to-value",
            "state_format": "float32 with explicit conversion envelopes",
            "timestep_relative_bounds": [-0.01, 0.01],
            "threshold_scale_relative_bounds": [-0.01, 0.01],
            "synaptic_delay": 0,
            "output_delay": 0,
        },
        "box_description": cover["box_description"],
        "reference_prediction": branch_rows[0]["reference_prediction"],
        "possible_predictions": possible_predictions,
        "cover": {
            "root_certified_parameter_fraction": source_fraction,
            "root_residual_parameter_fraction": residual_fraction,
            "area_sum_before_clamp": area_sum,
            "area_sum_error": area_sum - 1.0,
            "area_cover_valid": area_cover_valid,
            "residual_polygon_count": cover_row["unresolved_leaves"],
        },
        "fixed_trace_diagnostic": {
            "robust_polygon_count": fixed_trace["robust_trace_count"],
            "uncertain_guard_count_quantiles": fixed_trace[
                "uncertain_guard_count_quantiles"
            ],
            "uncertain_timestep_count_quantiles": fixed_trace[
                "uncertain_timestep_count_quantiles"
            ],
        },
        "branch_closure": {
            "complete_polygon_count": branch["complete_polygon_count"],
            "certified_polygon_count": branch["certified_polygon_count"],
            "residual_complete": residual_complete,
            "residual_certified": residual_certified,
            "maximum_active_branch_quantiles": {
                "minimum": int(np.min(active_branches)),
                "median": float(np.median(active_branches)),
                "q90": float(np.quantile(active_branches, 0.90)),
                "q99": float(np.quantile(active_branches, 0.99)),
                "maximum": int(np.max(active_branches)),
            },
            "maximum_total_branch_splits": max(
                row["total_branch_splits"] for row in branch_rows
            ),
            "seconds": branch["seconds"],
        },
        "per_input_family_certified": full_certificate,
        "certified_parameter_fraction": 1.0 if full_certificate else source_fraction,
        "route_assessment": {
            "advance_to_five_seed_audit_expansion": full_certificate,
            "claim_population_certified_fraction": False,
            "claim_all_execution_semantics_axes": False,
            "claim_physical_certificate": False,
            "advance_to_more_leaf_scaling": False,
        },
        "interpretation": (
            "The sound polygon cover proves 99.7557% of the joint continuous box. "
            "Every one of its 3,365 residual polygons is then closed by enumerating "
            "all guard-uncertain spike outcomes; every complete branch predicts "
            "class 2, with median peak branch count 2 and maximum 14. The area "
            "cover closes to one within 6.8e-12, yielding the first full per-input "
            "continuous-family certificate. Because the input was selected during "
            "method development, this result gates advancement to a frozen "
            "five-seed audit and is not itself population-level evidence."
        ),
    }
    output_path = root / "results/shd_v1/polygon_branch_certificate_summary.json"
    write_json(output_path, summary)


if __name__ == "__main__":
    main()
