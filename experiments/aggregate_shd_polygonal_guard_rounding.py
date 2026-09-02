from __future__ import annotations

import json
from pathlib import Path

from pines.artifacts import code_revision, file_reference, write_json


def _load_row(root: Path, relative_path: str, max_leaves: int) -> tuple[dict, str]:
    path = root / relative_path
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    row = next(row for row in report["rows"] if row["max_leaves"] == max_leaves)
    return row, file_reference(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    specifications = (
        (
            "guard_cap_128_1024",
            "artifacts/shd_v47_rounding_guard_cap128/seed_1701/"
            "reset_to_value_guard_cuts.json",
            1024,
        ),
        (
            "guard_cap_128_4096",
            "artifacts/shd_v47_rounding_guard_cap128/seed_1701/"
            "reset_to_value_guard_cuts.json",
            4096,
        ),
        (
            "guard_cap_512_4096",
            "artifacts/shd_v48_rounding_guard_cap512/seed_1701/"
            "reset_to_value_guard_cuts.json",
            4096,
        ),
        (
            "guard_cap_2048_16384",
            "artifacts/shd_v49_rounding_guard_cap2048/seed_1701/"
            "reset_to_value_guard_cuts.json",
            16384,
        ),
    )
    rows = []
    references = {}
    for variant, path, leaves in specifications:
        row, report_description = _load_row(root, path, leaves)
        rows.append({"variant": variant, **row})
        references[path] = report_description
    staged_4096 = next(row for row in rows if row["variant"] == "guard_cap_512_4096")
    staged_16384 = next(
        row for row in rows if row["variant"] == "guard_cap_2048_16384"
    )
    previous_summary = root / "results/shd_v1/polygonal_guard_cut_summary.json"
    output_path = root / "results/shd_v1/polygonal_guard_rounding_summary.json"
    summary = {
        "schema_version": "SHDPolygonalGuardRoundingSummary/v1",
        "seed": 1701,
        "condition": "reset_to_value",
        "relative_radius": 0.01,
        "code_revision": code_revision(root),
        "input_report_references": references,
        "superseded_arithmetic_idealized_summary": str(
            previous_summary.relative_to(root)
        ).replace("\\", "/"),
        "superseded_summary_reference": file_reference(previous_summary),
        "rows": rows,
        "rounding_model": (
            "state/current/output/logit float32 conversions receive a conservative "
            "epsilon-times-magnitude plus minimum-subnormal enclosure; output "
            "rounding errors are added to direct pairwise margins so shared spike "
            "terms remain correlated"
        ),
        "route_assessment": {
            "any_full_certificate": any(row["certified"] for row in rows),
            "best_certified_parameter_fraction": staged_16384[
                "certified_parameter_fraction"
            ],
            "remaining_unresolved_parameter_fraction": staged_16384[
                "unresolved_parameter_fraction"
            ],
            "remaining_unresolved_polygons": staged_16384["unresolved_leaves"],
            "coverage_gain_4096_to_16384_percentage_points": 100.0
            * (
                staged_16384["certified_parameter_fraction"]
                - staged_4096["certified_parameter_fraction"]
            ),
            "unresolved_reduction_factor_4096_to_16384": staged_4096[
                "unresolved_parameter_fraction"
            ]
            / staged_16384["unresolved_parameter_fraction"],
            "advance_to_more_leaf_budget": False,
            "advance_to_joint_timestep_threshold_oracle": True,
            "advance_current_method_to_five_seeds": False,
        },
        "interpretation": (
            "After explicit floating-roundoff accounting, staged polygonal guard "
            "cuts cover 82.26% at 1,024 leaves, 99.7557% at 4,096, and 99.8331% "
            "at 16,384. The largest run leaves 0.1669% area across 13,158 polygons "
            "and is not a certificate. Four times the leaves improve coverage by "
            "only 0.077 percentage points, so leaf scaling stops and the route "
            "advances to a joint timestep-threshold boundary oracle."
        ),
    }
    write_json(output_path, summary)


if __name__ == "__main__":
    main()
