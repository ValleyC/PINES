from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, file_reference, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", default="artifacts/dvs_gesture_v6_finite_family"
    )
    parser.add_argument(
        "--output", default="results/dvs_gesture_v3/finite_family_summary.json"
    )
    parser.add_argument(
        "--rows", default="results/dvs_gesture_v3/finite_family_rows.csv"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_root = root / args.input_root
    output_path = root / args.output
    rows_path = root / args.rows
    report_paths = sorted(input_root.glob("seed_*/finite_family_report.json"))
    if len(report_paths) != 5:
        raise ValueError(f"expected five finite-family reports, found {len(report_paths)}")

    reports = []
    for path in report_paths:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("schema_version") != "DVSGestureFiniteFamily/v1":
            raise ValueError(f"unsupported report: {path}")
        prediction_path = path.with_name("finite_family_predictions.npz")
        if file_reference(prediction_path) != report["prediction_file"]:
            raise ValueError(f"prediction reference mismatch: {prediction_path}")
        reports.append((path, report))

    member_reference_sets = {tuple(report["member_semantics_descriptions"]) for _, report in reports}
    if len(member_reference_sets) != 1:
        raise ValueError("finite-family members differ across seeds")
    fractions = np.asarray(
        [report["certified_fraction"] for _, report in reports], dtype=np.float64
    )
    elapsed = np.asarray(
        [report["elapsed_seconds"] for _, report in reports], dtype=np.float64
    )
    rows = [
        {
            "seed": int(report["seed"]),
            "audit_samples": int(report["audit_samples"]),
            "certified_inputs": int(report["certified_inputs"]),
            "certified_fraction": float(report["certified_fraction"]),
            "falsified_inputs": int(report["falsified_inputs"]),
            "unknown_inputs": int(report["unknown_inputs"]),
            "elapsed_seconds": float(report["elapsed_seconds"]),
            "report_description": file_reference(path),
        }
        for path, report in reports
    ]
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "DVSGestureFiniteFamilyAggregate/v1",
        "status": (
            "five-seed exact finite-family emulator evidence from the frozen "
            "development pipeline; not a continuous or physical certificate"
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
        "member_semantics_descriptions": list(next(iter(member_reference_sets))),
        "input_report_references": {
            str(path.relative_to(root)).replace("\\", "/"): file_reference(path)
            for path, _ in reports
        },
        "rows_csv_reference": file_reference(rows_path),
        "code_revision": code_revision(root),
        "interpretation": (
            "Exact enumeration proves prediction invariance only for the 16 "
            "listed point semantics. It does not cover continuous parameter "
            "tolerances or emulator-to-hardware error."
        ),
    }
    write_json(output_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
