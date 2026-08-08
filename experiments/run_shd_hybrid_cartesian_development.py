from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from transportcert.abstract import SemanticsBox
from transportcert.affine import AdaptiveHybridPolygonCertifier
from transportcert.artifacts import code_revision, sha256_file, write_json_immutable
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.emulator import VectorizedEmulator
from transportcert.models import DenseRecurrentSNN
from transportcert.parameter_batch import ReferenceParameterSweepEmulator
from transportcert.semantics import (
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)


def _ordering(timing: ThresholdTiming) -> UpdateOrdering:
    if timing is ThresholdTiming.PRE_INTEGRATION:
        return UpdateOrdering.THRESHOLD_RESET_INTEGRATE
    return UpdateOrdering.INTEGRATE_THRESHOLD_RESET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/shd_hybrid_cartesian_development_v2.json",
    )
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root",
        default="artifacts/shd_v64_hybrid_cartesian_development_v2",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") not in {
        "SHDHybridCartesianDevelopment/v1",
        "SHDHybridCartesianDevelopment/v2",
    }:
        raise ValueError("unsupported Cartesian development configuration")
    output_dir = root / args.output_root
    output_path = output_dir / "hybrid_cartesian_development.json"
    if output_path.exists():
        raise FileExistsError(f"Cartesian development output exists: {output_path}")

    selection = config["selection"]
    selection_report_path = None
    selection_report_hash = None
    if config["schema_version"] == "SHDHybridCartesianDevelopment/v2":
        if selection.get("uses_labels"):
            raise ValueError("Cartesian development selection must be label-free")
        selection_report_path = root / selection["report"]
        with selection_report_path.open("r", encoding="utf-8") as handle:
            selection_report = json.load(handle)
        if (
            selection_report.get("schema_version")
            != "SHDCartesianDevelopmentSelectionResult/v1"
        ):
            raise ValueError("unsupported Cartesian selection report")
        if selection_report["split"] != selection["split"]:
            raise ValueError("selection report split does not match configuration")
        selected = selection_report[selection["field"]]
        if selected is None:
            raise ValueError("selection report has no full-family-stable input")
        seed = int(selection_report["seed"])
        dataset_index = int(selected["dataset_index"])
        selection_report_hash = sha256_file(selection_report_path)
    else:
        seed = int(selection["seed"])
        dataset_index = int(selection["dataset_index"])
    seed_dir = root / args.artifact_root / f"seed_{seed}"
    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    frame = store.frames(np.asarray([dataset_index], dtype=np.int64))
    reference = primary_semantic_conditions()["reference"]
    family = config["family"]
    integrations = tuple(
        IntegrationRule(value) for value in family["integration_rules"]
    )
    timings = tuple(
        ThresholdTiming(value) for value in family["threshold_timings"]
    )
    resets = tuple(ResetRule(value) for value in family["reset_rules"])
    timestep_radius = float(family["relative_timestep_radius"])
    threshold_radius = float(family["relative_threshold_radius"])
    box = SemanticsBox(
        base=reference,
        timestep_bounds=(
            reference.timestep * (1.0 - timestep_radius),
            reference.timestep * (1.0 + timestep_radius),
        ),
        threshold_scale_bounds=(
            1.0 - threshold_radius,
            1.0 + threshold_radius,
        ),
        integration_rules=integrations,
        threshold_timings=timings,
        reset_rules=resets,
        synaptic_delays=tuple(int(v) for v in family["synaptic_delays"]),
        output_delays=tuple(int(v) for v in family["output_delays"]),
        name="shd-hybrid-cartesian-development-v1",
    )

    budgets = config["certificate_budgets"]
    certifier = AdaptiveHybridPolygonCertifier(
        max_branches=int(budgets["maximum_local_branches_per_member"]),
        max_guard_band_splits=int(budgets["maximum_guard_band_splits"]),
    )
    certificate_rows = []
    for max_leaves in budgets["maximum_polygon_leaves"]:
        started = time.perf_counter()
        result = certifier.certify(
            model, frame, reference, box, max_leaves=int(max_leaves)
        )
        certificate_rows.append(
            {
                "maximum_polygon_leaves": int(max_leaves),
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
                "maximum_completed_branches_across_members": (
                    result.maximum_completed_branches
                ),
                "guard_band_splits": result.guard_band_splits,
                "axis_fallback_splits": result.axis_fallback_splits,
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"Cartesian development leaves={max_leaves} "
            f"certified={result.certified} "
            f"covered={result.certified_parameter_fraction:.6f}",
            flush=True,
        )
        if result.certified and budgets.get("stop_after_first_certificate", False):
            break

    resolution = int(config["falsification_grid_resolution_per_axis"])
    factors = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    dt_factor, threshold_factor = np.meshgrid(factors, factors, indexing="ij")
    timesteps = reference.timestep * (
        1.0 + timestep_radius * dt_factor.ravel()
    )
    threshold_scales = 1.0 + threshold_radius * threshold_factor.ravel()
    reference_emulator = VectorizedEmulator()
    reference_prediction = int(
        reference_emulator.run(model, frame, reference).predictions[0]
    )
    parameter_engine = ReferenceParameterSweepEmulator(reference_emulator)
    member_rows = []
    all_identity = True
    for integration, timing, reset, synaptic_delay, output_delay in (
        itertools.product(
            integrations,
            timings,
            resets,
            box.synaptic_delays,
            box.output_delays,
        )
    ):
        member = replace(
            reference,
            integration_rule=integration,
            threshold_timing=timing,
            update_ordering=_ordering(timing),
            reset_rule=reset,
            synaptic_delay_steps=synaptic_delay,
            output_delay_steps=output_delay,
        )
        execution = parameter_engine.run(
            model,
            frame,
            member,
            timesteps,
            threshold_scales,
            np.asarray([reference_prediction], dtype=np.int16),
        )
        identity = bool(np.all(execution.predictions == reference_prediction))
        all_identity &= identity
        member_rows.append(
            {
                "semantics_hash": member.semantics_hash,
                "integration_rule": integration.value,
                "threshold_timing": timing.value,
                "reset_rule": reset.value,
                "synaptic_delay": synaptic_delay,
                "output_delay": output_delay,
                "grid_identity": identity,
                "counterexample_point_count": int(
                    np.count_nonzero(
                        execution.predictions != reference_prediction
                    )
                ),
                "observed_predictions": [
                    int(value) for value in np.unique(execution.predictions)
                ],
            }
        )

    report = {
        "schema_version": "SHDHybridCartesianDevelopmentResult/v1",
        "status": (
            "development-only 16-member continuous-family feasibility result; "
            "not population evidence and not a frozen audit"
        ),
        "config": config,
        "config_hash": sha256_file(config_path),
        "selection_report": (
            None
            if selection_report_path is None
            else str(selection_report_path.relative_to(root)).replace("\\", "/")
        ),
        "selection_report_hash": selection_report_hash,
        "seed": seed,
        "dataset_index": dataset_index,
        "box_hash": box.box_hash,
        "discrete_member_count": (
            len(integrations)
            * len(timings)
            * len(resets)
            * len(box.synaptic_delays)
            * len(box.output_delays)
        ),
        "reference_prediction": reference_prediction,
        "certificate_rows": certificate_rows,
        "falsification_grid": {
            "resolution_per_axis": resolution,
            "points_per_member": len(timesteps),
            "full_family_identity": all_identity,
            "member_rows": member_rows,
        },
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "split_indices_hash": sha256_file(split_path),
        "train_store_hash": store.data_hash,
        "reference_semantics_hash": reference.semantics_hash,
        "executor": "VectorizedEmulator canonical operational semantics",
        "device": "cpu",
        "code_revision": code_revision(root),
        "interpretation": (
            "A certificate requires all 16 mutually exclusive members and every "
            "continuous point to preserve the reference class. Grid identity is "
            "only a falsification ceiling. This selected input may guide method "
            "and resource development but cannot estimate audit coverage."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_immutable(output_path, report)


if __name__ == "__main__":
    main()
