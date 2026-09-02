from __future__ import annotations

import json
from pathlib import Path

from pines.artifacts import code_revision, file_reference, write_json


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results" / "shd_v1" / "affine_guard_domain_summary.json"
    if output_path.exists():
        raise FileExistsError("affine-guard summary already exists")

    cover_relatives = (
        "artifacts/shd_v27_affine_guard/seed_1701/"
        "reset_to_value_affine_guard.json",
        "artifacts/shd_v28_affine_guard_p16/seed_1701/"
        "reset_to_value_affine_guard.json",
    )
    adaptive_relatives = (
        "artifacts/shd_v29_adaptive_affine/seed_1701/"
        "reset_to_value_adaptive_affine.json",
        "artifacts/shd_v30_adaptive_affine_p16384/seed_1701/"
        "reset_to_value_adaptive_affine.json",
    )
    cover_rows = []
    adaptive_rows = []
    input_references = {}
    for relative in cover_relatives:
        path = root / relative
        report = _read(path)
        cover_rows.extend(report["rows"])
        input_references[relative] = file_reference(path)
    for relative in adaptive_relatives:
        path = root / relative
        report = _read(path)
        adaptive_rows.extend(report["rows"])
        input_references[relative] = file_reference(path)

    finest_cover = max(
        cover_rows, key=lambda row: int(row["partitions_per_axis"])
    )
    finest_adaptive = max(
        adaptive_rows, key=lambda row: int(row["max_leaves"])
    )
    summary = {
        "schema_version": "SHDAffineGuardDomainSummary/v1",
        "status": (
            "sound single-seed method-development result; local cell coverage and "
            "parameter volume are diagnostics, and no full-input certificate is claimed"
        ),
        "seed": 1701,
        "condition": "reset_to_value",
        "family_radius": 0.01,
        "uniform_cover_rows": sorted(
            cover_rows, key=lambda row: int(row["partitions_per_axis"])
        ),
        "adaptive_rows": sorted(
            adaptive_rows, key=lambda row: int(row["max_leaves"])
        ),
        "route_assessment": {
            "strict_synthetic_improvement_witness_in_unit_tests": True,
            "sampled_recurrent_soundness_checks_in_unit_tests": True,
            "finest_uniform_partitions_per_axis": int(
                finest_cover["partitions_per_axis"]
            ),
            "finest_uniform_affine_cell_input_fraction": float(
                finest_cover["affine_guard_certified_cell_input_fraction"]
            ),
            "finest_uniform_decision_interval_cell_input_fraction": float(
                finest_cover[
                    "decision_margin_certified_cell_input_fraction"
                ]
            ),
            "maximum_adaptive_leaves": int(finest_adaptive["max_leaves"]),
            "maximum_adaptive_certified_parameter_fraction": float(
                finest_adaptive["certified_parameter_fraction"]
            ),
            "any_full_input_certificate": any(
                bool(row["certified"]) for row in adaptive_rows
            )
            or any(
                int(row["affine_guard_full_cover_inputs"]) > 0
                for row in cover_rows
            ),
            "advance_to_symbolic_guard_surface_cuts": True,
            "advance_current_axis_aligned_cover_to_five_seeds": False,
        },
        "interpretation": (
            "Retaining shared timestep and threshold generators through recurrent state "
            "is the first domain to prove SHD reset-family sub-boxes. Uniform local "
            "coverage rises from 0.15% at four-by-four to 3.74% at eight-by-eight and "
            "18.46% at sixteen-by-sixteen, while decision intervals remain at zero. "
            "Adaptive certified parameter volume rises from 3.13% at 64 leaves to "
            "95.61% at 16,384 leaves on a grid-stable input, but unresolved guard-surface "
            "residue prevents a full certificate. The next method should cut or constrain "
            "symbolic guard surfaces rather than add more axis-aligned leaves."
        ),
        "input_report_references": input_references,
        "code_revision": code_revision(root),
    }
    write_json(output_path, summary)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
