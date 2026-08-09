from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from pines.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.emulator import VectorizedEmulator
from pines.models import DenseRecurrentSNN
from pines.parameter_batch import ReferenceParameterSweepEmulator
from pines.semantics import (
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
        default="configs/experiments/shd_cartesian_development_selection_v1.json",
    )
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root",
        default="artifacts/shd_v61_cartesian_development_selection_v1",
    )
    parser.add_argument("--input-batch-size", type=int, default=100)
    parser.add_argument("--parameter-batch-size", type=int, default=64)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "SHDCartesianDevelopmentSelection/v1":
        raise ValueError("unsupported Cartesian selection configuration")
    if config["selection"].get("uses_labels"):
        raise ValueError("development selector must remain label-free")

    output_dir = root / args.output_root
    output_path = output_dir / "cartesian_development_selection.json"
    if output_path.exists():
        raise FileExistsError(f"selection output exists: {output_path}")

    seed = int(config["seed"])
    seed_dir = root / args.artifact_root / f"seed_{seed}"
    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    with np.load(split_path, allow_pickle=False) as splits:
        indices = np.asarray(splits[config["split"]], dtype=np.int64)
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    frames = store.frames(indices)
    reference = primary_semantic_conditions()["reference"]

    family = config["family"]
    integrations = tuple(
        IntegrationRule(value) for value in family["integration_rules"]
    )
    timings = tuple(
        ThresholdTiming(value) for value in family["threshold_timings"]
    )
    resets = tuple(ResetRule(value) for value in family["reset_rules"])
    synaptic_delays = tuple(int(value) for value in family["synaptic_delays"])
    output_delays = tuple(int(value) for value in family["output_delays"])
    members = [
        replace(
            reference,
            integration_rule=integration,
            threshold_timing=timing,
            update_ordering=_ordering(timing),
            reset_rule=reset,
            synaptic_delay_steps=synaptic_delay,
            output_delay_steps=output_delay,
        )
        for integration, timing, reset, synaptic_delay, output_delay in itertools.product(
            integrations,
            timings,
            resets,
            synaptic_delays,
            output_delays,
        )
    ]

    resolution = int(config["selection"]["grid_resolution_per_axis"])
    factors = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    timestep_factor, threshold_factor = np.meshgrid(
        factors, factors, indexing="ij"
    )
    normalized_points = np.stack(
        [timestep_factor.ravel(), threshold_factor.ravel()], axis=1
    )
    timesteps = reference.timestep * (
        1.0
        + float(family["relative_timestep_radius"])
        * normalized_points[:, 0]
    )
    threshold_scales = (
        1.0
        + float(family["relative_threshold_radius"])
        * normalized_points[:, 1]
    )

    reference_emulator = VectorizedEmulator()
    reference_predictions = np.asarray(
        reference_emulator.run(model, frames, reference).predictions,
        dtype=np.int16,
    )
    engine = ReferenceParameterSweepEmulator(reference_emulator)
    counterexample_points = np.zeros(len(indices), dtype=np.int64)
    counterexample_members = np.zeros(len(indices), dtype=np.int64)
    minimum_margins = np.full(len(indices), np.inf, dtype=np.float64)
    member_rows = []
    started = time.perf_counter()
    for member_index, member in enumerate(members):
        execution = engine.run(
            model,
            frames,
            member,
            timesteps,
            threshold_scales,
            reference_predictions,
        )
        minimum_margins = np.minimum(
            minimum_margins, execution.minimum_reference_margin
        )
        mismatches = execution.predictions != reference_predictions[:, None]
        mismatch_counts = np.count_nonzero(mismatches, axis=1)
        counterexample_points += mismatch_counts
        counterexample_members += mismatch_counts > 0
        member_rows.append(
            {
                "member_index": member_index,
                "semantics_hash": member.semantics_hash,
                "semantics": member.to_dict(),
                "identity_input_count": int(np.count_nonzero(mismatch_counts == 0)),
                "counterexample_input_count": int(
                    np.count_nonzero(mismatch_counts > 0)
                ),
            }
        )
        print(
            f"Cartesian selection member={member_index + 1}/{len(members)} "
            f"identity={member_rows[-1]['identity_input_count']}/{len(indices)}",
            flush=True,
        )

    stable = counterexample_points == 0
    stable_positions = np.flatnonzero(stable)
    ranking = sorted(
        stable_positions,
        key=lambda position: (-minimum_margins[position], int(indices[position])),
    )
    rows = [
        {
            "development_position": int(position),
            "dataset_index": int(dataset_index),
            "reference_prediction": int(reference_predictions[position]),
            "full_family_grid_identity": bool(stable[position]),
            "counterexample_member_count": int(counterexample_members[position]),
            "counterexample_point_count": int(counterexample_points[position]),
            "minimum_reference_margin": float(minimum_margins[position]),
        }
        for position, dataset_index in enumerate(indices)
    ]
    selected_position = ranking[0] if ranking else None
    report = {
        "schema_version": "SHDCartesianDevelopmentSelectionResult/v1",
        "status": (
            "label-free open-calibration development selection; sampled identity "
            "is not a certificate and is not population evidence"
        ),
        "config": config,
        "config_hash": sha256_file(config_path),
        "seed": seed,
        "split": config["split"],
        "sample_count": len(indices),
        "discrete_member_count": len(members),
        "grid_resolution_per_axis": resolution,
        "points_per_member": len(normalized_points),
        "normalized_points_hash": array_hash(normalized_points),
        "full_family_grid_identity_count": int(np.count_nonzero(stable)),
        "full_family_grid_identity_fraction": float(np.mean(stable)),
        "selected_input": (
            None
            if selected_position is None
            else {
                "development_position": int(selected_position),
                "dataset_index": int(indices[selected_position]),
                "reference_prediction": int(
                    reference_predictions[selected_position]
                ),
                "minimum_reference_margin": float(
                    minimum_margins[selected_position]
                ),
            }
        ),
        "top_stable_candidates": [rows[position] for position in ranking[:20]],
        "member_rows": member_rows,
        "rows": rows,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "split_indices_hash": sha256_file(split_path),
        "train_store_hash": store.data_hash,
        "reference_semantics_hash": reference.semantics_hash,
        "executor": "VectorizedEmulator canonical operational semantics",
        "device": "cpu",
        "seconds": time.perf_counter() - started,
        "code_revision": code_revision(root),
        "interpretation": (
            "The selected input may be used to develop the sound Cartesian-family "
            "analyzer. It was chosen without labels from the repair-calibration "
            "split, so any later population claim requires a separately frozen audit."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_immutable(output_path, report)


if __name__ == "__main__":
    main()
