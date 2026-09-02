from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, file_reference, write_json


def _check_row(row: dict, budgets: dict) -> list[str]:
    failures: list[str] = []
    prefix = f"seed={row.get('seed')} position={row.get('audit_position')}"
    certified_fraction = float(row["certified_parameter_fraction"])
    unresolved_fraction = float(row["unresolved_parameter_fraction"])
    if not np.isfinite(certified_fraction) or not 0.0 <= certified_fraction <= 1.0:
        failures.append(f"{prefix}: invalid certified fraction")
    if not np.isfinite(unresolved_fraction) or not 0.0 <= unresolved_fraction <= 1.0:
        failures.append(f"{prefix}: invalid unresolved fraction")
    if certified_fraction + unresolved_fraction != 1.0:
        failures.append(f"{prefix}: fractions do not close exactly")
    expected_certified = int(row["unresolved_leaves"]) == 0
    if bool(row["certified"]) != expected_certified:
        failures.append(f"{prefix}: Boolean certificate disagrees with leaves")
    if row["certified"] and (certified_fraction != 1.0 or unresolved_fraction != 0.0):
        failures.append(f"{prefix}: complete certificate does not cover full area")
    terminal_leaves = (
        int(row["branch_certified_leaves"])
        + int(row["affine_certified_leaves"])
        + int(row["unresolved_leaves"])
    )
    if terminal_leaves != int(row["final_leaves"]):
        failures.append(f"{prefix}: terminal leaf accounting mismatch")
    if terminal_leaves > int(budgets["maximum_polygon_leaves"]):
        failures.append(f"{prefix}: polygon leaf budget exceeded")
    if int(row["guard_band_splits"]) > int(
        budgets["maximum_guard_band_splits"]
    ):
        failures.append(f"{prefix}: guard split budget exceeded")
    if int(row["maximum_completed_branches"]) > int(
        budgets["maximum_local_branches"]
    ):
        failures.append(f"{prefix}: local branch budget exceeded")
    if int(row["branch_attempts"]) != int(row["analyzed_polygons"]):
        failures.append(f"{prefix}: analyzed/branch-attempt count mismatch")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--source-provenance")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    report_path = root / args.audit_report
    output_path = root / args.output
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema_version") != "SHDHybridFamilyFullAuditResult/v1":
        raise ValueError("unsupported full hybrid audit artifact")
    provenance = None
    if args.source_provenance:
        provenance_path = root / args.source_provenance
        with provenance_path.open("r", encoding="utf-8") as handle:
            provenance = json.load(handle)
        if provenance.get("schema_version") != (
            "SHDHybridAuditScientificSourceProvenance/v2"
        ):
            raise ValueError("unsupported scientific source provenance")
        if provenance.get("audit_report_description") != file_reference(report_path):
            raise ValueError("source provenance references a different audit report")
    config = report["config"]
    arithmetic = config.get("analysis_arithmetic", {})
    required_contract = {
        "input_drive_error": "binary64 gamma bound with independent reduction allowance",
        "polygon_partition": "shared intersections and no positive-area child deletion",
        "partial_area_reporting": "outward lower certified-area bound",
    }
    contract_failures = [
        key
        for key, expected in required_contract.items()
        if arithmetic.get(key) != expected
    ]

    failures: list[str] = []
    shard_rows: list[dict] = []
    shard_revisions: set[str] = set()
    seen_shards: set[tuple[int, int]] = set()
    for record in report["shards"]:
        shard_path = root / record["path"]
        if file_reference(shard_path) != record["reference"]:
            failures.append(f"reference mismatch: {record['path']}")
            continue
        with shard_path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        identity = (int(shard["seed"]), int(shard["shard_index"]))
        if identity in seen_shards:
            failures.append(f"duplicate shard: {identity}")
        seen_shards.add(identity)
        if shard.get("config_reference") != report["config_reference"]:
            failures.append(f"config mismatch: {record['path']}")
        if shard.get("box_description") != report["box_description"]:
            failures.append(f"box mismatch: {record['path']}")
        if len(shard["rows"]) != int(record["sample_count"]):
            failures.append(f"row-count mismatch: {record['path']}")
        shard_rows.extend(shard["rows"])
        shard_revisions.add(str(shard.get("code_revision")))

    row_key = lambda row: (int(row["seed"]), int(row["audit_position"]))
    report_rows = sorted(report["rows"], key=row_key)
    shard_rows = sorted(shard_rows, key=row_key)
    if report_rows != shard_rows:
        failures.append("top-level rows are not an exact copy of saved shard rows")
    if len({row_key(row) for row in report_rows}) != len(report_rows):
        failures.append("duplicate model-input row identity")
    if len(report_rows) != int(report["sample_count"]):
        failures.append("top-level sample count mismatch")
    revision_match = shard_revisions == {str(report["code_revision"])}
    revision_supplement_valid = bool(
        provenance
        and shard_revisions.issubset(
            {"uncommitted", str(report["code_revision"])}
        )
        and provenance.get("audit_report_code_revision")
        == str(report["code_revision"])
        and provenance.get("scientific_core_unchanged_since_audit_revision")
        and provenance.get(
            "scientific_driver_functions_unchanged_since_audit_revision"
        )
        and int(provenance.get("shard_count", -1)) == len(report["shards"])
        and provenance.get("config_reference") == report["config_reference"]
    )
    if not revision_match and not revision_supplement_valid:
        failures.append(
            "mixed shard revision metadata lacks a valid source provenance supplement"
        )

    for seed in config["seeds"]:
        positions = sorted(
            int(row["audit_position"])
            for row in report_rows
            if int(row["seed"]) == int(seed)
        )
        if positions != list(range(len(positions))):
            failures.append(f"seed {seed}: audit positions are not a complete prefix")
    for row in report_rows:
        failures.extend(_check_row(row, config["certificate_budgets"]))

    certified_count = sum(bool(row["certified"]) for row in report_rows)
    if certified_count != int(report["certified_input_count"]):
        failures.append("aggregate certified count mismatch")
    if certified_count / len(report_rows) != float(report["certified_input_fraction"]):
        failures.append("aggregate certified fraction mismatch")

    verification = {
        "schema_version": "SHDHybridAuditArtifactVerification/v1",
        "status": "passed" if not failures and not contract_failures else "failed",
        "audit_report": str(report_path.relative_to(root)).replace("\\", "/"),
        "audit_report_description": file_reference(report_path),
        "sample_count": len(report_rows),
        "shard_count": len(seen_shards),
        "code_revision_under_test": report["code_revision"],
        "shard_code_revision_values": sorted(shard_revisions),
        "revision_metadata_match": revision_match,
        "revision_supplement_valid": revision_supplement_valid,
        "required_arithmetic_contract": required_contract,
        "missing_or_mismatched_contract_fields": contract_failures,
        "failure_count": len(failures),
        "failures": failures,
        "checker_code_revision": code_revision(root),
        "interpretation": (
            "This independently checks references, row topology, exact leaf/fraction "
            "accounting, resource caps, aggregate recomputation, and the declared "
            "roundoff/partition contract. It does not replace the analyzer's "
            "mathematical soundness tests."
        ),
    }
    write_json(output_path, verification)
    if verification["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
