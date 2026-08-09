from __future__ import annotations

import json
from pathlib import Path

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results" / "shd_v1" / "polygonal_guard_cut_summary.json"
    if output_path.exists():
        raise FileExistsError("polygonal guard-cut summary already exists")

    variants = {
        "axis_control_1024": (
            "artifacts/shd_v37_polygon_axis_scaled/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
        "guard_cap_8_1024": (
            "artifacts/shd_v38_staged_guard_cap8/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
        "guard_cap_32_1024": (
            "artifacts/shd_v39_staged_guard_cap32/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
        "guard_cap_128_1024": (
            "artifacts/shd_v40_staged_guard_cap128/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
        "guard_cap_128_4096": (
            "artifacts/shd_v41_staged_guard_cap128_p4096/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
        "guard_cap_512_4096": (
            "artifacts/shd_v42_staged_guard_cap512_p4096/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
        "guard_cap_2048_16384": (
            "artifacts/shd_v43_staged_guard_cap2048_p16384/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
    }
    rows = []
    input_hashes = {}
    for name, relative in variants.items():
        path = root / relative
        report = _read(path)
        row = report["rows"][0]
        rows.append(
            {
                "variant": name,
                "max_leaves": int(row["max_leaves"]),
                "guard_split_cap": report["max_guard_band_splits"],
                "certified": bool(row["certified"]),
                "certified_parameter_fraction": float(
                    row["certified_parameter_fraction"]
                ),
                "unresolved_parameter_fraction": float(
                    row["unresolved_parameter_fraction"]
                ),
                "analyzed_polygons": int(row["analyzed_polygons"]),
                "final_leaves": int(row["final_leaves"]),
                "certified_leaves": int(row["certified_leaves"]),
                "unresolved_leaves": int(row["unresolved_leaves"]),
                "guard_band_splits": int(row["guard_band_splits"]),
                "axis_fallback_splits": int(row["axis_fallback_splits"]),
                "maximum_depth": int(row["maximum_depth"]),
                "maximum_polygon_vertices": int(
                    row["maximum_polygon_vertices"]
                ),
                "seconds": float(row["seconds"]),
            }
        )
        input_hashes[relative] = sha256_file(path)

    axis_16384_relative = (
        "artifacts/shd_v30_adaptive_affine_p16384/seed_1701/"
        "reset_to_value_adaptive_affine.json"
    )
    axis_16384_path = root / axis_16384_relative
    axis_16384 = _read(axis_16384_path)["rows"][0]
    input_hashes[axis_16384_relative] = sha256_file(axis_16384_path)
    by_name = {row["variant"]: row for row in rows}
    final = by_name["guard_cap_2048_16384"]
    axis_unresolved = float(axis_16384["unresolved_parameter_fraction"])

    summary = {
        "schema_version": "SHDPolygonalGuardCutSummary/v1",
        "status": (
            "sound single-input method-development result; finite-grid input selection "
            "and certified parameter fraction are diagnostics, and no full certificate "
            "is claimed while any polygon remains unresolved"
        ),
        "seed": 1701,
        "condition": "reset_to_value",
        "family_radius": 0.01,
        "rows": rows,
        "matched_axis_16384": {
            "certified_parameter_fraction": float(
                axis_16384["certified_parameter_fraction"]
            ),
            "unresolved_parameter_fraction": axis_unresolved,
            "seconds": float(axis_16384["seconds"]),
        },
        "route_assessment": {
            "best_certified_parameter_fraction": final[
                "certified_parameter_fraction"
            ],
            "remaining_unresolved_parameter_fraction": final[
                "unresolved_parameter_fraction"
            ],
            "unresolved_reduction_factor_vs_axis_16384": axis_unresolved
            / final["unresolved_parameter_fraction"],
            "any_full_certificate": any(row["certified"] for row in rows),
            "advance_to_more_leaf_budget": False,
            "advance_to_exact_guard_boundary_oracle": True,
            "advance_current_method_to_five_seeds": False,
        },
        "interpretation": (
            "Polygonal robust guard-band cuts are the strongest sound continuous-family "
            "method so far. A staged leaves-over-eight guard budget reaches 93.57% at "
            "1,024 leaves, 99.68% at 4,096, and 99.9477% at 16,384. At the largest "
            "budget the unresolved area is about 84 times smaller than axis-aligned "
            "affine refinement, but 6,625 tiny polygons remain unresolved and the input "
            "is not certified. Further brute-force leaves are stopped; an exact guard-"
            "boundary oracle or constrained solver is required to close the residue."
        ),
        "input_report_hashes": input_hashes,
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, summary)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
