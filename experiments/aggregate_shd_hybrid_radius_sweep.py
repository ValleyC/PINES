from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pines.artifacts import code_revision, file_reference, write_json

if __package__:
    from .plot_shd_hybrid_radius_sweep import plot_radius_sweep
else:
    from plot_shd_hybrid_radius_sweep import plot_radius_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/shd_hybrid_radius_sweep_v1.json",
    )
    parser.add_argument("--output-root", default="results/shd_v1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "SHDHybridRadiusSweep/v1":
        raise ValueError("unsupported hybrid radius-sweep configuration")

    rows: list[dict[str, object]] = []
    report_descriptions: dict[str, str] = {}
    expected_seeds: list[int] | None = None
    expected_indices: dict[int, list[int]] | None = None
    for variant in config["variants"]:
        radius = float(variant["relative_radius"])
        audit_path = root / variant["audit_report"]
        grid_path = root / variant["grid_report"]
        variant_config_path = root / variant["config"]
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        variant_config = json.loads(variant_config_path.read_text(encoding="utf-8"))

        if audit.get("schema_version") != "SHDHybridFamilyAuditResult/v1":
            raise ValueError(f"unsupported audit report: {audit_path}")
        if grid.get("schema_version") != "SHDHybridAuditGridValidation/v1":
            raise ValueError(f"unsupported grid report: {grid_path}")
        if audit["config_reference"] != file_reference(variant_config_path):
            raise ValueError(f"audit config reference mismatch: {audit_path}")
        if grid["audit_report_description"] != file_reference(audit_path):
            raise ValueError(f"grid does not bind the audit report: {grid_path}")
        if not np.isclose(
            float(variant_config["relative_timestep_radius"]), radius
        ) or not np.isclose(
            float(variant_config["relative_threshold_radius"]), radius
        ):
            raise ValueError(f"radius mismatch: {variant_config_path}")
        if int(grid["grid_resolution_per_axis"]) != int(
            config["grid_resolution_per_axis"]
        ):
            raise ValueError(f"grid resolution mismatch: {grid_path}")
        if int(grid["certified_grid_violation_count"]) != 0:
            raise ValueError(f"certified grid violation: {grid_path}")

        seeds = [int(seed) for seed in variant_config["seeds"]]
        indices = {
            seed: [
                int(row["dataset_index"])
                for row in audit["rows"]
                if int(row["seed"]) == seed
            ]
            for seed in seeds
        }
        if expected_seeds is None:
            expected_seeds = seeds
            expected_indices = indices
        elif seeds != expected_seeds or indices != expected_indices:
            raise ValueError("the radius variants do not use identical model-input rows")

        samples = int(audit["sample_count"])
        if samples != int(grid["sample_count"]):
            raise ValueError(f"audit/grid sample mismatch: radius={radius}")
        certified = int(audit["certified_input_count"])
        falsified = int(grid["uncertified_grid_counterexample_count"])
        unresolved = int(grid["uncertified_grid_identity_count"])
        if certified + falsified + unresolved != samples:
            raise ValueError(f"outcome partition mismatch: radius={radius}")
        seconds = np.asarray(
            [float(row["seconds"]) for row in audit["rows"]], dtype=np.float64
        )
        per_seed_certified = []
        for seed in seeds:
            selected = [
                bool(row["certified"])
                for row in audit["rows"]
                if int(row["seed"]) == seed
            ]
            per_seed_certified.append(float(np.mean(selected)))
        row = {
            "relative_radius": radius,
            "radius_percent": radius * 100.0,
            "samples": samples,
            "seed_count": len(seeds),
            "certified_fraction": certified / samples,
            "grid_falsified_fraction": falsified / samples,
            "stable_unresolved_fraction": unresolved / samples,
            "grid_identity_fraction": float(grid["grid_identity_fraction"]),
            "proof_gap": unresolved / samples,
            "mean_certified_parameter_fraction": float(
                audit["mean_certified_parameter_fraction"]
            ),
            "median_seconds_per_input": float(np.median(seconds)),
            "p90_seconds_per_input": float(np.percentile(seconds, 90)),
            "minimum_seed_certified_fraction": float(
                np.min(per_seed_certified)
            ),
            "maximum_seed_certified_fraction": float(
                np.max(per_seed_certified)
            ),
        }
        rows.append(row)
        report_descriptions[f"audit_{radius:.4g}"] = file_reference(audit_path)
        report_descriptions[f"grid_{radius:.4g}"] = file_reference(grid_path)

    rows.sort(key=lambda row: float(row["relative_radius"]))
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "hybrid_radius_sweep_rows.csv"
    summary_path = output_root / "hybrid_radius_sweep_summary.json"
    figure_path = output_root / "shd_hybrid_radius_sweep.pdf"
    if rows_path.exists() or summary_path.exists() or figure_path.exists():
        raise FileExistsError("hybrid radius-sweep aggregate already exists")

    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "SHDHybridRadiusSweepAggregate/v1",
        "status": (
            "five-seed matched-input continuous-width sensitivity study with "
            "sound certification and independent finite-grid falsification"
        ),
        "scope": {
            "task": config["task"],
            "condition": config["condition"],
            "sample_count_per_seed": int(config["sample_count_per_seed"]),
            "grid_resolution_per_axis": int(
                config["grid_resolution_per_axis"]
            ),
            "selection": (
                "same first frozen certificate-audit inputs for every radius, "
                "without prediction or label selection"
            ),
        },
        "rows": rows,
        "monotonicity_checks": {
            "certified_fraction_nonincreasing": all(
                float(left["certified_fraction"])
                >= float(right["certified_fraction"])
                for left, right in zip(rows, rows[1:], strict=False)
            ),
            "grid_identity_nonincreasing": all(
                float(left["grid_identity_fraction"])
                >= float(right["grid_identity_fraction"])
                for left, right in zip(rows, rows[1:], strict=False)
            ),
        },
        "interpretation": (
            "Certified fraction measures complete proof at the fixed resource "
            "budget. Grid falsification provides concrete changed executions. "
            "Stable unresolved inputs isolate proof or resource conservatism."
        ),
        "config_reference": file_reference(config_path),
        "input_report_references": report_descriptions,
        "rows_csv_reference": file_reference(rows_path),
        "code_revision": code_revision(root),
    }
    plot_radius_sweep(rows, figure_path)
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
