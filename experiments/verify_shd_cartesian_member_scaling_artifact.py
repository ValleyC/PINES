from __future__ import annotations

import argparse
import json
from pathlib import Path

from pines.artifacts import sha256_file, sha256_json, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    report_path = root / args.report
    config_path = root / args.config
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "SHDHybridCartesianMemberScalingResult/v1":
        raise ValueError("unsupported Cartesian member-scaling report")
    if report.get("config_hash") != sha256_file(config_path):
        raise ValueError("Cartesian report config hash mismatch")
    selection_path = root / report["selection_report"]
    if report.get("selection_report_hash") != sha256_file(selection_path):
        raise ValueError("Cartesian report selection hash mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection[config["selection"]["field"]]
    if int(report["seed"]) != int(selection["seed"]):
        raise ValueError("Cartesian report seed does not match selection")
    if int(report["dataset_index"]) != int(selected["dataset_index"]):
        raise ValueError("Cartesian report input does not match selection")

    source_hashes = report.get("scientific_source_hashes", {})
    if not source_hashes:
        raise ValueError("Cartesian report has no scientific source manifest")
    if sha256_json(source_hashes) != report.get("scientific_source_manifest_hash"):
        raise ValueError("Cartesian source manifest hash mismatch")
    changed_sources = [
        relative
        for relative, expected_hash in source_hashes.items()
        if not (root / relative).exists()
        or sha256_file(root / relative) != expected_hash
    ]
    if changed_sources:
        raise ValueError(
            "scientific sources changed since execution: " + ", ".join(changed_sources)
        )

    member_rows = {
        int(row["member_index"]): row for row in report.get("member_rows", [])
    }
    if len(member_rows) != int(report["member_count"]):
        raise ValueError("Cartesian report has duplicate or missing members")
    shard_records = report.get("shards", [])
    if len(shard_records) != len(member_rows):
        raise ValueError("Cartesian shard count does not match member count")
    shard_revisions = set()
    maximum_area_error = 0.0
    verified_rows = {}
    for record in shard_records:
        member_index = int(record["member_index"])
        shard_path = root / record["path"]
        if sha256_file(shard_path) != record["hash"]:
            raise ValueError(f"Cartesian shard hash mismatch: {shard_path}")
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if shard.get("schema_version") != "SHDHybridCartesianMemberScalingShard/v1":
            raise ValueError(f"unsupported Cartesian shard: {shard_path}")
        if int(shard["member_index"]) != member_index:
            raise ValueError(f"Cartesian shard member mismatch: {shard_path}")
        if shard["config_hash"] != report["config_hash"]:
            raise ValueError(f"Cartesian shard config mismatch: {shard_path}")
        if shard["selection_report_hash"] != report["selection_report_hash"]:
            raise ValueError(f"Cartesian shard selection mismatch: {shard_path}")
        if (
            shard["scientific_source_manifest_hash"]
            != report["scientific_source_manifest_hash"]
        ):
            raise ValueError(f"Cartesian shard source mismatch: {shard_path}")
        if shard["result"] != member_rows[member_index]:
            raise ValueError(f"Cartesian shard result mismatch: {shard_path}")
        shard_revisions.add(str(shard["code_revision"]))
        for budget_row in shard["result"]["budget_rows"]:
            area_error = abs(
                float(budget_row["certified_parameter_fraction"])
                + float(budget_row["unresolved_parameter_fraction"])
                - 1.0
            )
            maximum_area_error = max(maximum_area_error, area_error)
            if area_error > 5e-12:
                raise ValueError(f"Cartesian area accounting fails: {shard_path}")
            if bool(budget_row["certified"]) != (
                int(budget_row["unresolved_leaves"]) == 0
                and float(budget_row["unresolved_parameter_fraction"]) <= 5e-12
            ):
                raise ValueError(f"Cartesian certificate flag mismatch: {shard_path}")
        verified_rows[member_index] = True
    if set(verified_rows) != set(member_rows):
        raise ValueError("Cartesian shards do not cover every member exactly once")

    verification = {
        "schema_version": "SHDHybridCartesianMemberScalingArtifactVerification/v1",
        "status": "independent immutable-shard and scientific-source verification",
        "report": args.report.replace("\\", "/"),
        "report_hash": sha256_file(report_path),
        "config": args.config.replace("\\", "/"),
        "config_hash": sha256_file(config_path),
        "member_count": len(member_rows),
        "shard_count": len(shard_records),
        "shard_code_revision_values": sorted(shard_revisions),
        "scientific_source_manifest_hash": report[
            "scientific_source_manifest_hash"
        ],
        "scientific_sources_match_execution_manifest": True,
        "maximum_area_accounting_error": maximum_area_error,
        "all_member_rows_match_shards": True,
        "all_shard_hashes_match": True,
        "all_members_verified": True,
    }
    write_json_immutable(root / args.output, verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
