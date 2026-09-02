from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pines.artifacts import (
    array_description,
    code_revision,
    file_reference,
    write_json,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.models import DenseRecurrentSNN
from pines.parameter_batch import TorchParameterBatchEmulator
from pines.torch_emulator import TorchEmulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--random-points-per-polygon", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=5017)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--residual-report",
        default=(
            "artifacts/shd_v50_rounding_guard_residuals/seed_1701/"
            "reset_to_value_guard_cuts.json"
        ),
    )
    parser.add_argument(
        "--output-root", default="artifacts/shd_v51_residual_counterexample_search"
    )
    args = parser.parse_args()
    if args.random_points_per_polygon < 0 or args.batch_size < 1:
        raise ValueError("sampling and batch budgets must be nonnegative")

    root = Path(__file__).resolve().parents[1]
    report_path = root / args.residual_report
    with report_path.open("r", encoding="utf-8") as handle:
        residual_report = json.load(handle)
    if residual_report["condition"] != args.condition:
        raise ValueError("residual report condition does not match")
    geometry_path = root / residual_report["residual_polygon_artifact"]
    if file_reference(geometry_path) != residual_report["residual_polygon_artifact_reference"]:
        raise ValueError("residual geometry reference mismatch")
    with np.load(geometry_path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"], dtype=np.float64)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
    polygons = [
        vertices[offsets[index] : offsets[index + 1]]
        for index in range(len(offsets) - 1)
    ]

    rng = np.random.default_rng(args.random_seed)
    point_parts = []
    for polygon in polygons:
        point_parts.append(polygon)
        point_parts.append(np.mean(polygon, axis=0, keepdims=True))
        point_parts.append((polygon + np.roll(polygon, -1, axis=0)) / 2.0)
        if args.random_points_per_polygon:
            weights = rng.dirichlet(
                np.ones(len(polygon)), size=args.random_points_per_polygon
            )
            point_parts.append(weights @ polygon)
    proposed_points = np.concatenate(point_parts, axis=0)
    normalized_points = np.unique(proposed_points, axis=0)
    if np.any(normalized_points < -1.0 - 1e-12) or np.any(
        normalized_points > 1.0 + 1e-12
    ):
        raise AssertionError("residual sample escaped normalized semantics domain")

    model_path = root / args.artifact_root / f"seed_{args.seed}" / "model.npz"
    split_path = (
        root / args.artifact_root / f"seed_{args.seed}" / "split_indices.npz"
    )
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    dataset_index = int(residual_report["rows"][0]["dataset_index"])
    frame = store.frames(np.asarray([dataset_index], dtype=np.int64))
    semantics = primary_semantic_conditions()
    reference = semantics["reference"]
    target = semantics[args.condition]
    radius = float(residual_report["relative_radius"])
    timesteps = reference.timestep * (1.0 + radius * normalized_points[:, 0])
    threshold_scales = 1.0 + radius * normalized_points[:, 1]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    reference_prediction = int(
        TorchEmulator(device=device, dtype=dtype)
        .run(model, frame, reference)
        .predictions[0]
        .item()
    )
    started = time.perf_counter()
    execution = TorchParameterBatchEmulator(device=device, dtype=dtype).run(
        model,
        frame,
        target,
        timesteps,
        threshold_scales,
        batch_size=args.batch_size,
    )
    seconds = time.perf_counter() - started
    competing_logits = execution.final_logits.copy()
    competing_logits[:, reference_prediction] = -np.inf
    margins = (
        execution.final_logits[:, reference_prediction]
        - np.max(competing_logits, axis=1)
    )
    counterexample_mask = execution.predictions != reference_prediction
    nearest_indices = np.argsort(margins)[: min(20, len(margins))]

    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"{args.condition}_residual_samples.npz"
    report_output = output_dir / f"{args.condition}_residual_search.json"
    if sample_path.exists() or report_output.exists():
        raise FileExistsError(f"residual-search output exists: {output_dir}")
    with sample_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            normalized_points=normalized_points,
            timesteps=timesteps,
            threshold_scales=threshold_scales,
            final_logits=execution.final_logits,
            predictions=execution.predictions,
            reference_margins=margins,
        )
    report = {
        "schema_version": "SHDResidualCounterexampleSearch/v1",
        "status": (
            "unlabeled counterexample search over residual polygon vertices, "
            "centroids, edge midpoints, and deterministic random convex points; "
            "absence of a sampled counterexample is not a certificate"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "dataset_index": dataset_index,
        "relative_radius": radius,
        "polygon_count": len(polygons),
        "proposed_point_count": len(proposed_points),
        "unique_point_count": len(normalized_points),
        "random_points_per_polygon": args.random_points_per_polygon,
        "random_seed": args.random_seed,
        "normalized_points_reference": array_description(normalized_points),
        "reference_prediction": reference_prediction,
        "observed_predictions": sorted(
            int(value) for value in np.unique(execution.predictions)
        ),
        "counterexample_count": int(np.count_nonzero(counterexample_mask)),
        "minimum_reference_margin": float(np.min(margins)),
        "margin_quantiles": {
            "q01": float(np.quantile(margins, 0.01)),
            "q10": float(np.quantile(margins, 0.10)),
            "median": float(np.median(margins)),
            "q90": float(np.quantile(margins, 0.90)),
        },
        "nearest_points": [
            {
                "sample_index": int(index),
                "normalized_timestep": float(normalized_points[index, 0]),
                "normalized_threshold": float(normalized_points[index, 1]),
                "timestep": float(timesteps[index]),
                "threshold_scale": float(threshold_scales[index]),
                "prediction": int(execution.predictions[index]),
                "reference_margin": float(margins[index]),
            }
            for index in nearest_indices
        ],
        "seconds": seconds,
        "device": device,
        "torch_version": torch.__version__,
        "sample_artifact": str(sample_path.relative_to(root)).replace("\\", "/"),
        "sample_artifact_reference": file_reference(sample_path),
        "residual_report_description": file_reference(report_path),
        "residual_geometry_reference": file_reference(geometry_path),
        "model_description": model.model_description,
        "model_file": file_reference(model_path),
        "train_store_reference": store.data_description,
        "split_indices_file": file_reference(split_path),
        "reference_semantics": reference.semantics_description,
        "target_semantics": target.semantics_description,
        "code_revision": code_revision(root),
        "interpretation": (
            "This search targets the exact polygons left by the sound analyzer. "
            "It can falsify the candidate family certificate but cannot prove it; "
            "a no-counterexample outcome only supports developing a tighter joint "
            "oracle."
        ),
    }
    write_json(report_output, report)


if __name__ == "__main__":
    main()
