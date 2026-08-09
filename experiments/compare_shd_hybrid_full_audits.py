from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-report", required=True)
    parser.add_argument("--corrected-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    prior_path = root / args.prior_report
    corrected_path = root / args.corrected_report
    with prior_path.open("r", encoding="utf-8") as handle:
        prior = json.load(handle)
    with corrected_path.open("r", encoding="utf-8") as handle:
        corrected = json.load(handle)
    expected_schema = "SHDHybridFamilyFullAuditResult/v1"
    if prior.get("schema_version") != expected_schema or corrected.get(
        "schema_version"
    ) != expected_schema:
        raise ValueError("unsupported full-audit report")

    unchanged_fields = (
        "condition",
        "relative_timestep_radius",
        "relative_threshold_radius",
        "seeds",
        "audit_selection",
        "certificate_budgets",
    )
    mismatched_fields = [
        field
        for field in unchanged_fields
        if prior["config"].get(field) != corrected["config"].get(field)
    ]
    if mismatched_fields:
        raise ValueError(f"scientific configurations differ: {mismatched_fields}")

    key = lambda row: (
        int(row["seed"]),
        int(row["audit_position"]),
        int(row["dataset_index"]),
    )
    prior_rows = {key(row): row for row in prior["rows"]}
    corrected_rows = {key(row): row for row in corrected["rows"]}
    if prior_rows.keys() != corrected_rows.keys():
        raise ValueError("audits do not contain the same model-input rows")

    transitions = {
        "remained_certified": 0,
        "lost_certificate": 0,
        "gained_certificate": 0,
        "remained_inconclusive": 0,
    }
    changed_rows = []
    for identity in sorted(prior_rows):
        before = bool(prior_rows[identity]["certified"])
        after = bool(corrected_rows[identity]["certified"])
        if before and after:
            transitions["remained_certified"] += 1
        elif before:
            transitions["lost_certificate"] += 1
        elif after:
            transitions["gained_certificate"] += 1
        else:
            transitions["remained_inconclusive"] += 1
        if before != after:
            changed_rows.append(
                {
                    "seed": identity[0],
                    "audit_position": identity[1],
                    "dataset_index": identity[2],
                    "prior_certified": before,
                    "corrected_certified": after,
                    "prior_certified_parameter_fraction": float(
                        prior_rows[identity]["certified_parameter_fraction"]
                    ),
                    "corrected_certified_parameter_fraction": float(
                        corrected_rows[identity]["certified_parameter_fraction"]
                    ),
                }
            )

    per_seed = []
    for seed in corrected["config"]["seeds"]:
        identities = [identity for identity in prior_rows if identity[0] == seed]
        before = sum(prior_rows[identity]["certified"] for identity in identities)
        after = sum(
            corrected_rows[identity]["certified"] for identity in identities
        )
        per_seed.append(
            {
                "seed": int(seed),
                "sample_count": len(identities),
                "prior_certified_count": int(before),
                "corrected_certified_count": int(after),
                "certified_count_delta": int(after - before),
                "prior_certified_fraction": before / len(identities),
                "corrected_certified_fraction": after / len(identities),
            }
        )

    report = {
        "schema_version": "SHDHybridFullAuditCorrectionComparison/v1",
        "status": "paired comparison with scientific domain and budgets unchanged",
        "prior_report": str(prior_path.relative_to(root)).replace("\\", "/"),
        "prior_report_hash": sha256_file(prior_path),
        "corrected_report": str(corrected_path.relative_to(root)).replace(
            "\\", "/"
        ),
        "corrected_report_hash": sha256_file(corrected_path),
        "sample_count": len(prior_rows),
        "transition_counts": transitions,
        "changed_row_count": len(changed_rows),
        "prior_certified_fraction": float(prior["certified_input_fraction"]),
        "corrected_certified_fraction": float(
            corrected["certified_input_fraction"]
        ),
        "certified_fraction_delta": float(
            corrected["certified_input_fraction"]
            - prior["certified_input_fraction"]
        ),
        "mean_partial_coverage_delta": float(
            np.mean(
                [
                    corrected_rows[identity]["certified_parameter_fraction"]
                    - prior_rows[identity]["certified_parameter_fraction"]
                    for identity in prior_rows
                ]
            )
        ),
        "per_seed": per_seed,
        "changed_rows": changed_rows,
        "prior_code_revision": prior["code_revision"],
        "corrected_code_revision": corrected["code_revision"],
        "comparison_code_revision": code_revision(root),
        "interpretation": (
            "The paired transition table isolates the empirical impact of the "
            "soundness correction because the model-input rows, semantic domain, "
            "and scientific resource caps are identical. The prior percentages "
            "are superseded regardless of the size or direction of the delta."
        ),
    }
    write_json_immutable(root / args.output, report)


if __name__ == "__main__":
    main()
