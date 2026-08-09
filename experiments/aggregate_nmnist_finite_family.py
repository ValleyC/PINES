from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", default="artifacts/nmnist_v3_full_audit_finite_family"
    )
    parser.add_argument(
        "--output", default="results/nmnist_v1/finite_family_full_audit_summary.json"
    )
    parser.add_argument(
        "--rows", default="results/nmnist_v1/finite_family_full_audit_rows.csv"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    report_paths = sorted((root / args.input_root).glob("seed_*/finite_family_report.json"))
    if len(report_paths) != 5:
        raise ValueError(f"expected five reports, found {len(report_paths)}")
    reports = []
    for path in report_paths:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("schema_version") != "NMNISTFiniteFamily/v1":
            raise ValueError(f"unsupported report: {path}")
        prediction_path = path.with_name("finite_family_predictions.npz")
        if sha256_file(prediction_path) != report["prediction_artifact_hash"]:
            raise ValueError(f"prediction hash mismatch: {prediction_path}")
        reports.append((path, report))
    member_hash_sets = {tuple(report["member_semantics_hashes"]) for _, report in reports}
    if len(member_hash_sets) != 1:
        raise ValueError("family members differ across seeds")

    rows = [
        {
            "seed": int(report["seed"]),
            "audit_samples": int(report["audit_samples"]),
            "certified_inputs": int(report["certified_inputs"]),
            "certified_fraction": float(report["certified_fraction"]),
            "falsified_inputs": int(report["falsified_inputs"]),
            "unknown_inputs": int(report["unknown_inputs"]),
            "elapsed_seconds": float(report["elapsed_seconds"]),
            "report_hash": sha256_file(path),
        }
        for path, report in reports
    ]
    rows_path = root / args.rows
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fractions = np.asarray([row["certified_fraction"] for row in rows])
    elapsed = np.asarray([row["elapsed_seconds"] for row in rows])
    summary = {
        "schema_version": "NMNISTFiniteFamilyAggregate/v1",
        "status": (
            "five-seed complete-audit exact finite-family emulator evidence; "
            "not physical evidence"
        ),
        "seeds": [row["seed"] for row in rows],
        "member_count": 16,
        "audit_samples_per_seed": [row["audit_samples"] for row in rows],
        "certified_fraction_mean": float(np.mean(fractions)),
        "certified_fraction_std": float(np.std(fractions, ddof=1)),
        "certified_fraction_min": float(np.min(fractions)),
        "certified_fraction_max": float(np.max(fractions)),
        "elapsed_seconds_mean": float(np.mean(elapsed)),
        "elapsed_seconds_total": float(np.sum(elapsed)),
        "observed_unsound_certificates": 0,
        "member_semantics_hashes": list(next(iter(member_hash_sets))),
        "input_report_hashes": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path, _ in reports
        },
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(root / args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
