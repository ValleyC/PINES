from __future__ import annotations

import json
from pathlib import Path

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = (
        root / "results" / "shd_v1" / "sampled_family_repair_development_summary.json"
    )
    if output_path.exists():
        raise FileExistsError("sampled-family repair summary already exists")

    variants = {
        "four_corner_10_epochs": (
            "artifacts/shd_v18_family_margin_smoke/seed_1701/reset_to_value/"
            "family_margin/repair_report.json",
            "artifacts/shd_v19_family_margin_smoke_grid/seed_1701/"
            "reset_to_value_family_grid.json",
        ),
        "four_corner_40_epochs": (
            "artifacts/shd_v18_family_margin/seed_1701/reset_to_value/"
            "family_margin/repair_report.json",
            "artifacts/shd_v19_family_margin_grid/seed_1701/"
            "reset_to_value_family_grid.json",
        ),
        "three_by_three_10_epochs": (
            "artifacts/shd_v20_family_margin_grid3_smoke/seed_1701/reset_to_value/"
            "family_margin/repair_report.json",
            "artifacts/shd_v21_family_margin_grid3_smoke_grid/seed_1701/"
            "reset_to_value_family_grid.json",
        ),
        "three_by_three_40_epochs": (
            "artifacts/shd_v20_family_margin_grid3_full/seed_1701/reset_to_value/"
            "family_margin/repair_report.json",
            "artifacts/shd_v21_family_margin_grid3_full_grid/seed_1701/"
            "reset_to_value_family_grid.json",
        ),
    }
    rows = []
    input_hashes = {}
    for name, (repair_relative, grid_relative) in variants.items():
        repair_path = root / repair_relative
        grid_path = root / grid_relative
        repair = _read(repair_path)
        grid = _read(grid_path)
        grid_row = grid["rows"][0]
        rows.append(
            {
                "variant": name,
                "training_grid_resolution": int(
                    repair["config"].get("family_grid_resolution", 2)
                ),
                "epochs": int(repair["config"]["epochs"]),
                "optimization_steps": int(repair["optimization_steps"]),
                "elapsed_seconds": float(repair["elapsed_seconds"]),
                "best_calibration_family_disagreements": int(
                    repair["selected"]["best_calibration_disagreements"]
                ),
                "audit_disagreement_rate": float(
                    repair["after"]["audit_disagreement_rate"]
                ),
                "certificate_upper_bound": float(
                    repair["after"]["certificate_upper_bound"]
                ),
                "accuracy_recovery_fraction": float(
                    repair["accuracy_recovery_fraction"]
                ),
                "center_identity_fraction": float(
                    grid_row["center_target_identity_fraction"]
                ),
                "nine_by_nine_family_identity_fraction": float(
                    grid_row["grid_family_identity_fraction"]
                ),
            }
        )
        input_hashes[repair_relative] = sha256_file(repair_path)
        input_hashes[grid_relative] = sha256_file(grid_path)

    baseline_relative = (
        "artifacts/shd_v12_repair_family_grid/seed_1701/"
        "reset_to_value_family_grid.json"
    )
    baseline_path = root / baseline_relative
    baseline = _read(baseline_path)
    baseline_identity = {
        str(row["method"]): float(row["grid_family_identity_fraction"])
        for row in baseline["rows"]
    }
    input_hashes[baseline_relative] = sha256_file(baseline_path)
    row_by_name = {row["variant"]: row for row in rows}
    best = max(rows, key=lambda row: row["nine_by_nine_family_identity_fraction"])
    short_grid = row_by_name["three_by_three_10_epochs"]
    full_grid = row_by_name["three_by_three_40_epochs"]

    summary = {
        "schema_version": "SHDSampledFamilyRepairDevelopment/v1",
        "status": (
            "single-seed calibration-only method-development and kill decision; "
            "not sound, five-seed, or primary evidence"
        ),
        "seed": 1701,
        "condition": "reset_to_value",
        "family_radius": 0.01,
        "evaluation_grid_resolution": 9,
        "rows": rows,
        "existing_seed_1701_baselines": baseline_identity,
        "route_assessment": {
            "best_variant": best["variant"],
            "best_family_identity_fraction": best[
                "nine_by_nine_family_identity_fraction"
            ],
            "best_exceeds_certificate_directed": best[
                "nine_by_nine_family_identity_fraction"
            ]
            > baseline_identity["certificate_directed"],
            "best_exceeds_logit_only": best[
                "nine_by_nine_family_identity_fraction"
            ]
            > baseline_identity["logit_only"],
            "full_training_improves_calibration_count": full_grid[
                "best_calibration_family_disagreements"
            ]
            < short_grid["best_calibration_family_disagreements"],
            "full_training_improves_audit_family_identity": full_grid[
                "nine_by_nine_family_identity_fraction"
            ]
            > short_grid["nine_by_nine_family_identity_fraction"],
            "run_sound_branch_analysis": False,
            "advance_to_five_seed": False,
        },
        "gate_rule": (
            "Sound branch analysis and five-seed expansion require reliable audit-family "
            "improvement over both certificate-directed and logit-only baselines."
        ),
        "interpretation": (
            "Worst-point training on four corners or a three-by-three grid restores "
            "accuracy but does not reliably generalize to the untouched nine-by-nine "
            "family. The best development checkpoint beats certificate-directed repair "
            "by one of 128 audit inputs but remains below logit-only; longer training "
            "improves calibration disagreement while reducing audit-family identity. "
            "A sampled execution loss is therefore not a certificate-restoring proxy."
        ),
        "input_report_hashes": input_hashes,
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, summary)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
