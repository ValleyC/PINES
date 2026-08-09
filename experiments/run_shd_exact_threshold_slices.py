from __future__ import annotations

import argparse
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
from pines.exact_boundary import ExactThresholdBoundaryOracle
from pines.models import DenseRecurrentSNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--timestep-slices", type=int, default=257)
    parser.add_argument("--max-cells", type=int, default=100_000)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--selection-report",
        default=(
            "artifacts/shd_v43_staged_guard_cap2048_p16384/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
    )
    parser.add_argument(
        "--output-root", default="artifacts/shd_v44_exact_threshold_slices"
    )
    args = parser.parse_args()
    if not 0.0 < args.radius < 1.0:
        raise ValueError("radius must be between zero and one")
    if args.timestep_slices < 2 or args.max_cells < 1:
        raise ValueError("slice and cell budgets must be positive")

    root = Path(__file__).resolve().parents[1]
    model_path = root / args.artifact_root / f"seed_{args.seed}" / "model.npz"
    split_path = (
        root / args.artifact_root / f"seed_{args.seed}" / "split_indices.npz"
    )
    selection_path = root / args.selection_report
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_path = output_dir / f"{args.condition}_exact_threshold_slices.json"
    if output_path.exists():
        raise FileExistsError(f"exact-threshold output exists: {output_path}")

    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if int(selection["seed"]) != args.seed:
        raise ValueError("selection report seed does not match")
    if selection["condition"] != args.condition:
        raise ValueError("selection report condition does not match")
    selected_indices = np.asarray(selection["selected_indices"], dtype=np.int64)

    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    frames = store.frames(selected_indices)
    semantics = primary_semantic_conditions()
    if args.condition == "reference" or args.condition not in semantics:
        raise ValueError("condition must name a declared target semantics")
    reference = semantics["reference"]
    target = semantics[args.condition]
    timestep_factors = np.linspace(
        1.0 - args.radius,
        1.0 + args.radius,
        args.timestep_slices,
        dtype=np.float64,
    )
    threshold_bounds = (1.0 - args.radius, 1.0 + args.radius)
    oracle = ExactThresholdBoundaryOracle()
    rows = []
    started = time.perf_counter()
    for input_position, (dataset_index, frame) in enumerate(
        zip(selected_indices, frames, strict=True)
    ):
        for slice_index, factor in enumerate(timestep_factors):
            fixed_target = replace(
                target,
                timestep=reference.timestep * float(factor),
            )
            result = oracle.certify(
                model,
                frame[None, ...],
                reference,
                fixed_target,
                threshold_bounds,
                max_cells=args.max_cells,
            )
            rows.append(
                {
                    "input_position": input_position,
                    "dataset_index": int(dataset_index),
                    "slice_index": slice_index,
                    "timestep_factor": float(factor),
                    "target_semantics_hash": fixed_target.semantics_hash,
                    "result": result,
                }
            )
            if (slice_index + 1) % 32 == 0 or slice_index + 1 == len(
                timestep_factors
            ):
                print(
                    f"exact threshold input={input_position + 1}/{len(frames)} "
                    f"slice={slice_index + 1}/{len(timestep_factors)} "
                    f"certified={result.certified} cells={len(result.cells)}",
                    flush=True,
                )

    certified_slices = sum(row["result"].certified for row in rows)
    conclusive_slices = sum(row["result"].conclusive for row in rows)
    report = {
        "schema_version": "SHDExactThresholdSlices/v1",
        "status": (
            "exact over every binary64 threshold scale at each enumerated fixed "
            "timestep; the timestep grid is diagnostic and is not a joint "
            "continuous-family certificate"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "relative_radius": args.radius,
        "timestep_slice_count": len(timestep_factors),
        "timestep_factors_hash": array_hash(timestep_factors),
        "threshold_scale_bounds": threshold_bounds,
        "selected_indices": selected_indices,
        "selected_indices_hash": array_hash(selected_indices),
        "certified_slices": certified_slices,
        "conclusive_slices": conclusive_slices,
        "all_slices_certified": certified_slices == len(rows),
        "total_exact_cells": sum(len(row["result"].cells) for row in rows),
        "maximum_cells_per_slice": max(
            len(row["result"].cells) for row in rows
        ),
        "maximum_unique_traces_per_slice": max(
            row["result"].unique_trace_count for row in rows
        ),
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "selection_report_hash": sha256_file(selection_path),
        "reference_semantics_hash": reference.semantics_hash,
        "base_target_semantics_hash": target.semantics_hash,
        "code_revision": code_revision(root),
        "interpretation": (
            "The exact slice oracle removes threshold-boundary relaxation as an "
            "explanation at the sampled timesteps. It does not cover the open "
            "regions between timestep slices; joint certification still requires "
            "a sound timestep-continuous boundary method."
        ),
    }
    write_json_immutable(output_path, report)


if __name__ == "__main__":
    main()
