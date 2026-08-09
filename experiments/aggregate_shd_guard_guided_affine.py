from __future__ import annotations

import json
from pathlib import Path

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results" / "shd_v1" / "guard_guided_affine_summary.json"
    if output_path.exists():
        raise FileExistsError("guard-guided affine summary already exists")
    width_relative = (
        "artifacts/shd_v29_adaptive_affine/seed_1701/"
        "reset_to_value_adaptive_affine.json"
    )
    guard_relative = (
        "artifacts/shd_v31_guard_guided_affine/seed_1701/"
        "reset_to_value_adaptive_affine.json"
    )
    width = _read(root / width_relative)
    guard = _read(root / guard_relative)
    width_by_budget = {int(row["max_leaves"]): row for row in width["rows"]}
    guard_by_budget = {int(row["max_leaves"]): row for row in guard["rows"]}
    budgets = sorted(set(width_by_budget) & set(guard_by_budget))
    rows = []
    for budget in budgets:
        width_fraction = float(
            width_by_budget[budget]["certified_parameter_fraction"]
        )
        guard_fraction = float(
            guard_by_budget[budget]["certified_parameter_fraction"]
        )
        rows.append(
            {
                "max_leaves": budget,
                "width_certified_parameter_fraction": width_fraction,
                "guard_certified_parameter_fraction": guard_fraction,
                "guard_minus_width_fraction": guard_fraction - width_fraction,
                "width_seconds": float(width_by_budget[budget]["seconds"]),
                "guard_seconds": float(guard_by_budget[budget]["seconds"]),
            }
        )
    summary = {
        "schema_version": "SHDGuardGuidedAffineSummary/v1",
        "status": (
            "sound single-input matched-budget split-policy development comparison; "
            "certified volume is diagnostic and neither policy completes the input"
        ),
        "seed": 1701,
        "condition": "reset_to_value",
        "family_radius": 0.01,
        "rows": rows,
        "route_assessment": {
            "maximum_absolute_fraction_gain": max(
                abs(row["guard_minus_width_fraction"]) for row in rows
            ),
            "guard_better_at_finest_shared_budget": rows[-1][
                "guard_minus_width_fraction"
            ]
            > 0.0,
            "guard_finest_fraction": rows[-1][
                "guard_certified_parameter_fraction"
            ],
            "width_finest_fraction": rows[-1][
                "width_certified_parameter_fraction"
            ],
            "any_full_certificate": False,
            "advance_guard_axis_heuristic_to_five_seeds": False,
            "advance_to_non_axis_aligned_guard_constraints": True,
        },
        "interpretation": (
            "Choosing the split axis from accumulated uncertain-guard generator scores "
            "is matched at 64 leaves, slightly worse at 256, and improves certified "
            "volume by 1.37 points at 1,024 and 0.63 points at 4,096. The small, "
            "non-monotone gain does not remove unresolved guard-surface residue. Further "
            "axis-policy tuning is stopped in favor of explicit non-axis-aligned guard "
            "constraints."
        ),
        "input_report_hashes": {
            width_relative: sha256_file(root / width_relative),
            guard_relative: sha256_file(root / guard_relative),
        },
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, summary)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
