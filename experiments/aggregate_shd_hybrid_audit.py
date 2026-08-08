from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable


def _exact_interval(successes: int, samples: int, confidence: float) -> list[float]:
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, samples - successes + 1)
    )
    upper = 1.0 if successes == samples else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, samples - successes)
    )
    return [lower, upper]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-report",
        default="artifacts/shd_v57_hybrid_audit_v1/hybrid_family_audit.json",
    )
    parser.add_argument(
        "--grid-report",
        default=(
            "artifacts/shd_v58_hybrid_audit_grid_v1/"
            "hybrid_audit_grid_validation.json"
        ),
    )
    parser.add_argument(
        "--output", default="results/shd_v1/hybrid_family_audit_summary.json"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    audit_path = root / args.audit_report
    grid_path = root / args.grid_report
    output_path = root / args.output
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    with grid_path.open("r", encoding="utf-8") as handle:
        grid = json.load(handle)
    if audit.get("schema_version") != "SHDHybridFamilyAuditResult/v1":
        raise ValueError("unsupported hybrid audit report")
    if grid.get("schema_version") != "SHDHybridAuditGridValidation/v1":
        raise ValueError("unsupported hybrid grid validation report")
    if grid["audit_report_hash"] != sha256_file(audit_path):
        raise ValueError("grid validation does not reference the supplied audit")

    audit_keys = {
        (row["seed"], row["audit_position"], row["dataset_index"])
        for row in audit["rows"]
    }
    grid_keys = {
        (row["seed"], row["audit_position"], row["dataset_index"])
        for row in grid["rows"]
    }
    if audit_keys != grid_keys:
        raise ValueError("audit and grid rows do not identify the same inputs")

    samples = int(audit["sample_count"])
    certified = int(audit["certified_input_count"])
    stable = int(grid["grid_identity_count"])
    violations = int(grid["certified_grid_violation_count"])
    counterexamples = samples - stable
    uncertified_stable = int(grid["uncertified_grid_identity_count"])
    confidence = 0.95
    rows = audit["rows"]
    summary = {
        "schema_version": "SHDHybridFamilyAuditSummary/v1",
        "status": (
            "frozen five-seed screen plus independent finite-grid falsification; "
            "the screen passed and triggered the separately frozen full audit"
        ),
        "scope": {
            "task": "SHD recurrent SNN",
            "target_reset": "reset-to-value",
            "integration": "forward Euler",
            "relative_timestep_bounds": [-0.01, 0.01],
            "relative_threshold_scale_bounds": [-0.01, 0.01],
            "state_arithmetic": "float32 with explicit sound rounding envelopes",
            "selection": (
                "first 32 frozen certificate-audit inputs per seed; no labels, "
                "predictions, margins, or grid behavior used for selection"
            ),
        },
        "sample_count": samples,
        "seed_count": len(audit["per_seed"]),
        "certified_input_count": certified,
        "certified_input_fraction": certified / samples,
        "certified_input_fraction_exact_95_percent_interval": _exact_interval(
            certified, samples, confidence
        ),
        "mean_certified_parameter_fraction": float(
            audit["mean_certified_parameter_fraction"]
        ),
        "median_certified_parameter_fraction": float(
            np.median([row["certified_parameter_fraction"] for row in rows])
        ),
        "per_seed": audit["per_seed"],
        "advance_gate": {
            "minimum_certified_input_fraction": audit["config"]["advance_gate"][
                "minimum_mean_certified_input_fraction"
            ],
            "passed": bool(audit["advance_gate_passed"]),
            "every_seed_nonzero": all(
                row["certified_input_count"] > 0 for row in audit["per_seed"]
            ),
            "area_partition_maximum_absolute_error": float(
                max(
                    abs(
                        row["certified_parameter_fraction"]
                        + row["unresolved_parameter_fraction"]
                        - 1.0
                    )
                    for row in rows
                )
            ),
        },
        "finite_grid_falsification": {
            "resolution_per_axis": int(grid["grid_resolution_per_axis"]),
            "points_per_input": int(grid["grid_point_count"]),
            "grid_identity_count": stable,
            "grid_identity_fraction": stable / samples,
            "grid_counterexample_input_count": counterexamples,
            "certified_grid_violation_count": violations,
            "uncertified_grid_identity_count": uncertified_stable,
            "uncertified_grid_counterexample_count": int(
                grid["uncertified_grid_counterexample_count"]
            ),
            "certified_fraction_of_grid_stable_inputs": certified / stable,
            "interpretation": (
                "Zero violations is a falsification result, not an additional proof. "
                "The gap between 70% grid identity and 36.25% full certification "
                "quantifies proof- and resource-budget conservatism on this screen."
            ),
        },
        "runtime": {
            "certificate_wall_seconds": float(audit["seconds"]),
            "certificate_median_seconds_per_input": float(
                np.median([row["seconds"] for row in rows])
            ),
            "grid_wall_seconds": float(grid["seconds"]),
        },
        "source_report_hashes": {
            str(audit_path.relative_to(root)).replace("\\", "/"): sha256_file(
                audit_path
            ),
            str(grid_path.relative_to(root)).replace("\\", "/"): sha256_file(
                grid_path
            ),
        },
        "source_code_revisions": {
            "certificate_audit": audit["code_revision"],
            "grid_validation": grid["code_revision"],
        },
        "code_revision": code_revision(root),
        "route_assessment": {
            "advance_to_full_frozen_audit": bool(audit["advance_gate_passed"]),
            "claim_population_certified_fraction_from_screen": False,
            "claim_all_execution_semantics_axes": False,
            "claim_physical_certificate": False,
        },
        "interpretation": (
            "The method advances beyond a selected development example: with all "
            "budgets frozen, it proves the entire joint continuous box for 58 of "
            "160 untouched screen inputs and for at least one input from every "
            "training seed. The result supports tractability of the proposed "
            "per-input family certificate for this bounded subfamily, while the "
            "full split, other axes, a second event task, and hardware remain open."
        ),
    }
    write_json_immutable(output_path, summary)


if __name__ == "__main__":
    main()
