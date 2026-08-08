from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from transportcert.abstract import SemanticsBox
from transportcert.affine import AdaptiveAffineGuardCutCertifier
from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.models import DenseRecurrentSNN
from transportcert.torch_emulator import TorchEmulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--selection-pool", type=int, default=128)
    parser.add_argument("--selection-grid", type=int, default=9)
    parser.add_argument("--max-leaves", type=int, nargs="+", default=(64, 256))
    parser.add_argument("--max-guard-band-splits", type=int, default=-1)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v32_guard_cuts")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1 or args.selection_pool < args.sample_count:
        raise ValueError("selection pool must cover the positive sample count")
    if args.selection_grid < 2 or any(value < 1 for value in args.max_leaves):
        raise ValueError("grid and leaf budgets must be positive")
    if args.max_guard_band_splits < -1:
        raise ValueError("maximum guard-band splits must be -1 or nonnegative")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.condition}_guard_cuts.json"
    if report_path.exists():
        raise FileExistsError(f"guard-cut output exists: {report_path}")

    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(split_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    pool_indices = audit_indices[: min(args.selection_pool, len(audit_indices))]
    pool_frames = store.frames(pool_indices)
    semantics = primary_semantic_conditions()
    if args.condition == "reference" or args.condition not in semantics:
        raise ValueError("condition must name a declared non-reference semantics")
    reference = semantics["reference"]
    target = semantics[args.condition]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = TorchEmulator(device=device, dtype=torch.float32)
    reference_predictions = np.asarray(
        emulator.run(model, pool_frames, reference).numpy().predictions
    )
    factors = np.linspace(
        1.0 - args.radius,
        1.0 + args.radius,
        args.selection_grid,
    )
    grid_stable = np.ones(len(pool_indices), dtype=bool)
    for timestep_factor in factors:
        target_semantics = replace(
            target, timestep=reference.timestep * float(timestep_factor)
        )
        for threshold_factor in factors:
            candidate = model.with_parameters(
                threshold=model.threshold * float(threshold_factor)
            )
            predictions = np.asarray(
                emulator.run(candidate, pool_frames, target_semantics)
                .numpy()
                .predictions
            )
            grid_stable &= predictions == reference_predictions
    stable_positions = np.flatnonzero(grid_stable)
    if len(stable_positions) < args.sample_count:
        raise RuntimeError("selection pool contains too few grid-stable inputs")
    selected_positions = stable_positions[: args.sample_count]
    selected_indices = pool_indices[selected_positions]
    selected_frames = pool_frames[selected_positions]

    box = SemanticsBox(
        base=target,
        timestep_bounds=(
            reference.timestep * (1.0 - args.radius),
            reference.timestep * (1.0 + args.radius),
        ),
        threshold_scale_bounds=(1.0 - args.radius, 1.0 + args.radius),
        integration_rules=(target.integration_rule,),
        threshold_timings=(target.threshold_timing,),
        reset_rules=(target.reset_rule,),
        synaptic_delays=(target.synaptic_delay_steps,),
        output_delays=(target.output_delay_steps,),
        name=f"{args.condition}-joint-pm-{args.radius:.6g}",
    )
    certifier = AdaptiveAffineGuardCutCertifier(
        max_guard_band_splits=(
            None
            if args.max_guard_band_splits == -1
            else args.max_guard_band_splits
        )
    )
    rows = []
    for max_leaves in args.max_leaves:
        for position, (dataset_index, frame) in enumerate(
            zip(selected_indices, selected_frames, strict=True)
        ):
            started = time.perf_counter()
            result = certifier.certify(
                model,
                frame[None, ...],
                reference,
                box,
                max_leaves=max_leaves,
            )
            rows.append(
                {
                    "selected_position": position,
                    "dataset_index": int(dataset_index),
                    "max_leaves": max_leaves,
                    "certified": result.certified,
                    "certified_parameter_fraction": result.certified_parameter_fraction,
                    "unresolved_parameter_fraction": result.unresolved_parameter_fraction,
                    "analyzed_polygons": result.analyzed_polygons,
                    "final_leaves": result.final_leaves,
                    "certified_leaves": result.certified_leaves,
                    "unresolved_leaves": result.unresolved_leaves,
                    "maximum_depth": result.maximum_depth,
                    "maximum_polygon_vertices": result.maximum_polygon_vertices,
                    "guard_band_splits": result.guard_band_splits,
                    "axis_fallback_splits": result.axis_fallback_splits,
                    "seconds": time.perf_counter() - started,
                }
            )
            print(
                f"guard cuts seed={args.seed} leaves={max_leaves} "
                f"input={position + 1}/{len(selected_indices)} "
                f"certified={result.certified} "
                f"covered={result.certified_parameter_fraction:.4f}",
                flush=True,
            )

    report = {
        "schema_version": "SHDGuardCutCertificate/v1",
        "status": (
            "sound single-seed polygonal method-development diagnostic selected from "
            "finite-grid-stable inputs; only zero unresolved area is a certificate"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "relative_radius": args.radius,
        "selection_pool": len(pool_indices),
        "selection_grid_resolution": args.selection_grid,
        "grid_stable_pool_inputs": int(np.count_nonzero(grid_stable)),
        "sample_count": len(selected_indices),
        "selected_indices": selected_indices.tolist(),
        "selected_indices_hash": array_hash(selected_indices),
        "max_leaf_budgets": list(args.max_leaves),
        "max_guard_band_splits": (
            None
            if args.max_guard_band_splits == -1
            else args.max_guard_band_splits
        ),
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "reference_semantics_hash": reference.semantics_hash,
        "target_semantics_hash": target.semantics_hash,
        "box_hash": box.box_hash,
        "rows": rows,
        "device_for_grid_selection": device,
        "torch_version": torch.__version__,
        "code_revision": code_revision(root),
        "interpretation": (
            "Each cut partitions the normalized timestep/threshold polygon into robust "
            "quiet, residual-band, and spiking regions. Polygon areas form a cover; "
            "certified area is diagnostic until the unresolved area is zero."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
