from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from transportcert.abstract import SemanticsBox
from transportcert.affine import AdaptiveHybridPolygonCertifier
from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.models import DenseRecurrentSNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/shd_hybrid_audit_v1.json"
    )
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v57_hybrid_audit_v1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "SHDHybridFamilyAudit/v1":
        raise ValueError("unsupported hybrid audit configuration")
    output_dir = root / args.output_root
    output_path = output_dir / "hybrid_family_audit.json"
    if output_path.exists():
        raise FileExistsError(f"hybrid audit output exists: {output_path}")

    store = PackedSHD(root / args.data_root / "train.npz")
    semantics = primary_semantic_conditions()
    condition = config["condition"]
    if condition not in semantics or condition == "reference":
        raise ValueError("configuration must name a target condition")
    reference = semantics["reference"]
    target = semantics[condition]
    timestep_radius = float(config["relative_timestep_radius"])
    threshold_radius = float(config["relative_threshold_radius"])
    box = SemanticsBox(
        base=target,
        timestep_bounds=(
            reference.timestep * (1.0 - timestep_radius),
            reference.timestep * (1.0 + timestep_radius),
        ),
        threshold_scale_bounds=(
            1.0 - threshold_radius,
            1.0 + threshold_radius,
        ),
        integration_rules=(target.integration_rule,),
        threshold_timings=(target.threshold_timing,),
        reset_rules=(target.reset_rule,),
        synaptic_delays=(target.synaptic_delay_steps,),
        output_delays=(target.output_delay_steps,),
        name=f"{condition}-hybrid-audit-v1",
    )
    budgets = config["certificate_budgets"]
    certifier = AdaptiveHybridPolygonCertifier(
        max_branches=int(budgets["maximum_local_branches"]),
        max_guard_band_splits=int(budgets["maximum_guard_band_splits"]),
    )
    sample_count = int(config["audit_selection"]["sample_count_per_seed"])
    rows = []
    seed_provenance = {}
    run_started = time.perf_counter()
    total_inputs = sample_count * len(config["seeds"])
    completed_inputs = 0

    for seed in config["seeds"]:
        seed_dir = root / args.artifact_root / f"seed_{seed}"
        model_path = seed_dir / "model.npz"
        split_path = seed_dir / "split_indices.npz"
        model = DenseRecurrentSNN.load(model_path)
        with np.load(split_path, allow_pickle=False) as splits:
            audit_indices = np.asarray(
                splits[config["audit_selection"]["split"]], dtype=np.int64
            )
        if len(audit_indices) < sample_count:
            raise ValueError(f"seed {seed} has too few audit inputs")
        selected_indices = audit_indices[:sample_count]
        frames = store.frames(selected_indices)
        seed_provenance[str(seed)] = {
            "model_hash": model.model_hash,
            "model_artifact_hash": sha256_file(model_path),
            "split_indices_hash": sha256_file(split_path),
            "selected_indices": selected_indices,
            "selected_indices_hash": array_hash(selected_indices),
        }
        for position, (dataset_index, frame) in enumerate(
            zip(selected_indices, frames, strict=True)
        ):
            started = time.perf_counter()
            result = certifier.certify(
                model,
                frame[None, ...],
                reference,
                box,
                max_leaves=int(budgets["maximum_polygon_leaves"]),
            )
            rows.append(
                {
                    "seed": int(seed),
                    "audit_position": position,
                    "dataset_index": int(dataset_index),
                    "certified": result.certified,
                    "certified_parameter_fraction": (
                        result.certified_parameter_fraction
                    ),
                    "unresolved_parameter_fraction": (
                        result.unresolved_parameter_fraction
                    ),
                    "analyzed_polygons": result.analyzed_polygons,
                    "final_leaves": result.final_leaves,
                    "branch_certified_leaves": result.branch_certified_leaves,
                    "affine_certified_leaves": result.affine_certified_leaves,
                    "unresolved_leaves": result.unresolved_leaves,
                    "branch_attempts": result.branch_attempts,
                    "branch_cap_hits": result.branch_cap_hits,
                    "branch_prediction_rejections": (
                        result.branch_prediction_rejections
                    ),
                    "maximum_completed_branches": (
                        result.maximum_completed_branches
                    ),
                    "guard_band_splits": result.guard_band_splits,
                    "axis_fallback_splits": result.axis_fallback_splits,
                    "seconds": time.perf_counter() - started,
                }
            )
            completed_inputs += 1
            print(
                f"hybrid audit seed={seed} input={position + 1}/{sample_count} "
                f"overall={completed_inputs}/{total_inputs} "
                f"certified={result.certified} "
                f"covered={result.certified_parameter_fraction:.4f}",
                flush=True,
            )

    per_seed = []
    for seed in config["seeds"]:
        seed_rows = [row for row in rows if row["seed"] == seed]
        per_seed.append(
            {
                "seed": seed,
                "sample_count": len(seed_rows),
                "certified_input_count": sum(row["certified"] for row in seed_rows),
                "certified_input_fraction": float(
                    np.mean([row["certified"] for row in seed_rows])
                ),
                "mean_certified_parameter_fraction": float(
                    np.mean(
                        [row["certified_parameter_fraction"] for row in seed_rows]
                    )
                ),
                "median_seconds": float(
                    np.median([row["seconds"] for row in seed_rows])
                ),
            }
        )
    certified_fraction = float(np.mean([row["certified"] for row in rows]))
    gate = config["advance_gate"]
    gate_passed = (
        certified_fraction >= float(gate["minimum_mean_certified_input_fraction"])
        and (
            not gate["require_every_seed_nonzero"]
            or all(row["certified_input_count"] > 0 for row in per_seed)
        )
        and (
            not gate["require_zero_area_cover_failures"]
            or all(
                np.isclose(
                    row["certified_parameter_fraction"]
                    + row["unresolved_parameter_fraction"],
                    1.0,
                )
                for row in rows
            )
        )
    )
    report = {
        "schema_version": "SHDHybridFamilyAuditResult/v1",
        "status": (
            "five-seed frozen audit subset with no prediction- or label-based "
            "input filtering; only zero unresolved area is certified"
        ),
        "config": config,
        "config_hash": sha256_file(config_path),
        "condition": condition,
        "box_hash": box.box_hash,
        "sample_count": len(rows),
        "certified_input_count": sum(row["certified"] for row in rows),
        "certified_input_fraction": certified_fraction,
        "mean_certified_parameter_fraction": float(
            np.mean([row["certified_parameter_fraction"] for row in rows])
        ),
        "per_seed": per_seed,
        "advance_gate_passed": gate_passed,
        "rows": rows,
        "seconds": time.perf_counter() - run_started,
        "seed_provenance": seed_provenance,
        "train_store_hash": store.data_hash,
        "reference_semantics_hash": reference.semantics_hash,
        "target_semantics_hash": target.semantics_hash,
        "code_revision": code_revision(root),
        "interpretation": (
            "This audit estimates certificate tractability on untouched split "
            "positions after freezing method budgets. It is still a development "
            "subset until the declared gate triggers the full frozen audit."
        ),
    }
    write_json_immutable(output_path, report)


if __name__ == "__main__":
    main()
