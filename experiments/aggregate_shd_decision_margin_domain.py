from __future__ import annotations

import json
from pathlib import Path

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results" / "shd_v1" / "decision_margin_domain_summary.json"
    if output_path.exists():
        raise FileExistsError("decision-margin summary already exists")

    relative_paths = (
        "artifacts/shd_v22_decision_margin/seed_1701/"
        "reset_to_value_decision_margin.json",
        "artifacts/shd_v23_decision_margin_fine/seed_1701/"
        "reset_to_value_decision_margin.json",
        "artifacts/shd_v24_decision_margin_p32/seed_1701/"
        "reset_to_value_decision_margin.json",
    )
    rows = []
    input_hashes = {}
    for relative in relative_paths:
        path = root / relative
        report = _read(path)
        for row in report["rows"]:
            rows.append(
                {
                    "sample_count": int(report["sample_count"]),
                    "partitions_per_axis": int(row["partitions_per_axis"]),
                    "subbox_count": int(row["subbox_count"]),
                    "interval_certified_inputs": int(
                        row["interval_certified_inputs"]
                    ),
                    "decision_margin_certified_inputs": int(
                        row["decision_margin_certified_inputs"]
                    ),
                    "interval_seconds": float(row["interval_seconds"]),
                    "decision_margin_seconds": float(
                        row["decision_margin_seconds"]
                    ),
                }
            )
        input_hashes[relative] = sha256_file(path)

    summary = {
        "schema_version": "SHDDecisionMarginDomainSummary/v1",
        "status": (
            "sound single-seed staged domain-development kill test; sample count is "
            "reduced as partition cost increases and the result is not five-seed evidence"
        ),
        "seed": 1701,
        "condition": "reset_to_value",
        "family_radius": 0.01,
        "rows": sorted(rows, key=lambda row: row["partitions_per_axis"]),
        "route_assessment": {
            "synthetic_strict_improvement_witness_in_unit_tests": True,
            "sampled_soundness_checks_in_unit_tests": True,
            "maximum_partitions_per_axis": max(
                row["partitions_per_axis"] for row in rows
            ),
            "any_interval_certificate": any(
                row["interval_certified_inputs"] > 0 for row in rows
            ),
            "any_decision_margin_certificate": any(
                row["decision_margin_certified_inputs"] > 0 for row in rows
            ),
            "advance_to_five_seed": False,
        },
        "interpretation": (
            "Direct pairwise output-margin propagation is strictly tighter than separate "
            "logit intervals on a constructed witness, but certifies no staged SHD input "
            "through 32-by-32 partitioning. Output correlation is therefore not the "
            "dominant loss; the next domain must retain correlations through recurrent "
            "hidden states and threshold guards. Zero coverage is an abstraction failure, "
            "not evidence of true prediction instability."
        ),
        "input_report_hashes": input_hashes,
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, summary)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
