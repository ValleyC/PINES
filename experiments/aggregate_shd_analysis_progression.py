from __future__ import annotations

import csv
import json
from pathlib import Path

from pines.artifacts import code_revision, file_reference, write_json


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    affine_path = (
        root
        / "artifacts/shd_v31_guard_guided_affine/seed_1701/"
        "reset_to_value_adaptive_affine.json"
    )
    polygon_path = (
        root
        / "artifacts/shd_v50_rounding_guard_residuals/seed_1701/"
        "reset_to_value_guard_cuts.json"
    )
    branch_path = (
        root
        / "artifacts/shd_v56_residual_polygon_branches_all/seed_1701/"
        "reset_to_value_polygon_branches.json"
    )
    affine = _load(affine_path)
    polygon = _load(polygon_path)
    branch = _load(branch_path)
    model_descriptions = {
        affine["model_description"], polygon["model_description"], branch["model_description"]
    }
    box_descriptions = {affine["box_description"], polygon["box_description"], branch["box_description"]}
    if len(model_descriptions) != 1 or len(box_descriptions) != 1:
        raise ValueError("analysis stages do not use the same model and contract")
    if affine["selected_indices"] != [6] or polygon["selected_indices"] != [6]:
        raise ValueError("analysis stages do not use the frozen development input")
    if int(branch["dataset_index"]) != 6:
        raise ValueError("branch closure uses a different development input")

    affine_row = next(
        row for row in affine["rows"] if int(row["max_leaves"]) == 4096
    )
    polygon_row = next(
        row
        for row in polygon["rows"]
        if int(row["max_leaves"]) == 4096
        and int(row["guard_band_splits"]) == 512
    )
    branch_rows = branch["rows"]
    if not branch_rows or not all(row["complete"] for row in branch_rows):
        raise ValueError("local branch closure is incomplete")
    p99_active = sorted(int(row["maximum_active_branches"]) for row in branch_rows)[
        int(0.99 * (len(branch_rows) - 1))
    ]
    rows = [
        {
            "analysis": "shared affine boxes",
            "certified_parameter_fraction": float(
                affine_row["certified_parameter_fraction"]
            ),
            "full_certificate": bool(affine_row["certified"]),
            "seconds": float(affine_row["seconds"]),
            "unresolved_regions": int(affine_row["unresolved_leaves"]),
            "p99_active_branches": 0,
        },
        {
            "analysis": "polygonal guard cuts",
            "certified_parameter_fraction": float(
                polygon_row["certified_parameter_fraction"]
            ),
            "full_certificate": bool(polygon_row["certified"]),
            "seconds": float(polygon_row["seconds"]),
            "unresolved_regions": int(polygon_row["unresolved_leaves"]),
            "p99_active_branches": 0,
        },
        {
            "analysis": "polygonal cuts and local branch closure",
            "certified_parameter_fraction": 1.0,
            "full_certificate": True,
            "seconds": float(polygon_row["seconds"] + branch["seconds"]),
            "unresolved_regions": 0,
            "p99_active_branches": p99_active,
        },
    ]

    rows_path = root / "results/shd_v1/analysis_progression_rows.csv"
    summary_path = root / "results/shd_v1/analysis_progression_summary.json"
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "SHDAnalysisProgressionAggregate/v1",
        "status": (
            "sound same-input method-development comparison; not a population "
            "certificate"
        ),
        "seed": 1701,
        "dataset_index": 6,
        "model_description": next(iter(model_descriptions)),
        "box_description": next(iter(box_descriptions)),
        "contract": (
            "reset-to-value with joint plus/minus one-percent timestep and "
            "threshold ranges"
        ),
        "rows": rows,
        "input_report_references": {
            str(path.relative_to(root)).replace("\\", "/"): file_reference(path)
            for path in (affine_path, polygon_path, branch_path)
        },
        "rows_csv_reference": file_reference(rows_path),
        "code_revision": code_revision(root),
        "interpretation": (
            "Shared affine boxes preserve semantic dependence but leave guard "
            "ambiguity. Polygonal cuts close most of the parameter area. Exact "
            "local enumeration of the remaining guard branches completes the "
            "certificate for the same input and contract."
        ),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
