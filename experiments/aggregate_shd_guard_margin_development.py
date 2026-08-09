from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _proof_depth(path: Path, partitions: int = 128) -> float:
    report = _read(path)
    selected = [
        row
        for row in report["rows"]
        if int(row["partitions_per_axis"]) == partitions
    ]
    return float(np.mean([float(row["median_abort_step"]) for row in selected]))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results" / "shd_v1" / "guard_margin_development_summary.json"
    if output_path.exists():
        raise FileExistsError("guard-margin development summary already exists")

    smoke_roots = {
        "0.1": "shd_v14_guard_margin_smoke",
        "0.5": "shd_v14_guard_margin_w05",
        "1.0": "shd_v14_guard_margin_w10",
    }
    proof_roots = {
        "0.1": "shd_v16_guard_proof_w01",
        "0.5": "shd_v16_guard_proof_w05",
        "1.0": "shd_v16_guard_proof_w10",
    }
    input_hashes = {}
    smoke_rows = []
    for weight in smoke_roots:
        repair_path = (
            root
            / "artifacts"
            / smoke_roots[weight]
            / "seed_1701"
            / "reset_to_value"
            / "guard_margin"
            / "repair_report.json"
        )
        proof_path = (
            root
            / "artifacts"
            / proof_roots[weight]
            / "seed_1701"
            / "reset_to_value_branch_cells.json"
        )
        repair = _read(repair_path)
        best = min(repair["history"], key=lambda row: float(row["selection_score"]))
        smoke_rows.append(
            {
                "guard_weight": float(weight),
                "epochs": 10,
                "best_calibration_disagreements": int(
                    best["calibration_disagreements"]
                ),
                "best_guard_near_fraction": float(best["guard_near_fraction"]),
                "audit_disagreement_rate": float(
                    repair["after"]["audit_disagreement_rate"]
                ),
                "accuracy_recovery_fraction": float(
                    repair["accuracy_recovery_fraction"]
                ),
                "certificate_upper_bound": float(
                    repair["after"]["certificate_upper_bound"]
                ),
                "mean_abort_step_at_128_partitions": _proof_depth(proof_path),
            }
        )
        input_hashes[str(repair_path.relative_to(root))] = sha256_file(repair_path)
        input_hashes[str(proof_path.relative_to(root))] = sha256_file(proof_path)

    full_repair_path = (
        root
        / "artifacts"
        / "shd_v15_guard_margin"
        / "seed_1701"
        / "reset_to_value"
        / "guard_margin"
        / "repair_report.json"
    )
    full_grid_path = (
        root
        / "artifacts"
        / "shd_v17_guard_family_grid"
        / "seed_1701"
        / "reset_to_value_family_grid.json"
    )
    full_proof_path = (
        root
        / "artifacts"
        / "shd_v17_guard_branch_cells"
        / "seed_1701"
        / "reset_to_value_branch_cells.json"
    )
    full_repair = _read(full_repair_path)
    full_grid = _read(full_grid_path)
    grid_row = full_grid["rows"][0]
    prior_proof = _read(
        root / "results" / "shd_v1" / "repaired_branch_set_cells_summary.json"
    )["route_assessment"]
    full_row = {
        "guard_weight": 0.1,
        "epochs": 40,
        "audit_disagreement_rate": float(
            full_repair["after"]["audit_disagreement_rate"]
        ),
        "accuracy_recovery_fraction": float(
            full_repair["accuracy_recovery_fraction"]
        ),
        "certificate_upper_bound": float(
            full_repair["after"]["certificate_upper_bound"]
        ),
        "grid_family_identity_fraction": float(
            grid_row["grid_family_identity_fraction"]
        ),
        "mean_abort_step_at_128_partitions": _proof_depth(full_proof_path),
        "no_repair_mean_abort_step": float(
            prior_proof["finest_no_repair_mean_abort_step"]
        ),
        "certificate_directed_mean_abort_step": float(
            prior_proof["finest_certificate_directed_mean_abort_step"]
        ),
        "logit_only_mean_abort_step": float(
            prior_proof["finest_logit_only_mean_abort_step"]
        ),
    }
    for path in (full_repair_path, full_grid_path, full_proof_path):
        input_hashes[str(path.relative_to(root))] = sha256_file(path)

    summary = {
        "schema_version": "SHDGuardMarginDevelopment/v1",
        "status": (
            "single-seed calibration-only method-development and kill decision; "
            "not five-seed or primary evidence"
        ),
        "seed": 1701,
        "condition": "reset_to_value",
        "smoke_rows": smoke_rows,
        "selected_full_run": full_row,
        "route_assessment": {
            "selected_weight": 0.1,
            "selected_weight_rule": (
                "best qualitative transport/proof tradeoff on calibration-only "
                "development; audit/test metrics are reported but were not selection inputs"
            ),
            "any_representative_cell_completed": False,
            "full_guard_exceeds_no_repair_proof_depth": full_row[
                "mean_abort_step_at_128_partitions"
            ]
            > full_row["no_repair_mean_abort_step"],
            "advance_to_five_seed": False,
        },
        "interpretation": (
            "A local near-threshold penalty partially recovers explicit proof depth relative "
            "to the prior certificate-directed proxy, but neither completes a cell nor "
            "matches the unrepaired proof depth. Stronger weights sacrifice transport "
            "accuracy without improving depth. Multi-step verified margins are required."
        ),
        "input_report_hashes": input_hashes,
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, summary)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
