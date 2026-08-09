from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pines.abstract import SemanticsBox
from pines.affine import FixedTraceAffineAnalyzer, polygon_area
from pines.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.models import DenseRecurrentSNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--max-polygons", type=int, default=256)
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
        "--output-root", default="artifacts/shd_v53_residual_fixed_trace"
    )
    args = parser.parse_args()
    if args.max_polygons < 1:
        raise ValueError("max-polygons must be positive")

    root = Path(__file__).resolve().parents[1]
    residual_report_path = root / args.residual_report
    with residual_report_path.open("r", encoding="utf-8") as handle:
        residual_report = json.load(handle)
    if residual_report["condition"] != args.condition:
        raise ValueError("residual condition does not match")
    geometry_path = root / residual_report["residual_polygon_artifact"]
    if sha256_file(geometry_path) != residual_report["residual_polygon_artifact_hash"]:
        raise ValueError("residual geometry hash mismatch")
    with np.load(geometry_path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"], dtype=np.float64)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
    polygons = [
        vertices[offsets[index] : offsets[index + 1]]
        for index in range(len(offsets) - 1)
    ]
    areas = np.asarray([polygon_area(polygon) for polygon in polygons])
    selected_polygon_indices = np.argsort(-areas)[: min(args.max_polygons, len(polygons))]

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
    box = SemanticsBox(
        base=target,
        timestep_bounds=(
            reference.timestep * (1.0 - radius),
            reference.timestep * (1.0 + radius),
        ),
        threshold_scale_bounds=(1.0 - radius, 1.0 + radius),
        integration_rules=(target.integration_rule,),
        threshold_timings=(target.threshold_timing,),
        reset_rules=(target.reset_rule,),
        synaptic_delays=(target.synaptic_delay_steps,),
        output_delays=(target.output_delay_steps,),
        name=f"{args.condition}-joint-pm-{radius:.6g}",
    )
    analyzer = FixedTraceAffineAnalyzer()
    rows = []
    started = time.perf_counter()
    for position, polygon_index in enumerate(selected_polygon_indices):
        polygon = polygons[int(polygon_index)]
        result = analyzer.analyze(model, frame, reference, box, polygon)
        rows.append(
            {
                "selection_position": position,
                "polygon_index": int(polygon_index),
                "normalized_area": float(areas[polygon_index]),
                "vertex_count": len(polygon),
                "trace_robust": result.trace_robust,
                "certified": result.certified,
                "reference_prediction": result.reference_prediction,
                "trace_prediction": result.trace_prediction,
                "total_guard_count": result.total_guard_count,
                "uncertain_guard_count": result.uncertain_guard_count,
                "uncertain_timestep_count": result.uncertain_timestep_count,
                "first_uncertain_timestep": result.first_uncertain_timestep,
                "minimum_signed_guard_margin": result.minimum_signed_guard_margin,
            }
        )
        if (position + 1) % 32 == 0 or position + 1 == len(
            selected_polygon_indices
        ):
            print(
                f"fixed trace polygon={position + 1}/{len(selected_polygon_indices)} "
                f"robust={sum(row['trace_robust'] for row in rows)}",
                flush=True,
            )

    uncertain_counts = np.asarray(
        [row["uncertain_guard_count"] for row in rows], dtype=np.int64
    )
    uncertain_timesteps = np.asarray(
        [row["uncertain_timestep_count"] for row in rows], dtype=np.int64
    )
    report = {
        "schema_version": "SHDResidualFixedTraceDiagnostic/v1",
        "status": (
            "sound fixed-center-trace diagnostic on largest unresolved polygons; "
            "non-robust traces are inconclusive and not certificates"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "dataset_index": dataset_index,
        "relative_radius": radius,
        "source_polygon_count": len(polygons),
        "selected_polygon_count": len(selected_polygon_indices),
        "selection_rule": "largest normalized area first",
        "selected_polygon_indices": selected_polygon_indices,
        "selected_polygon_indices_hash": array_hash(selected_polygon_indices),
        "selected_area_fraction_of_residue": float(
            np.sum(areas[selected_polygon_indices]) / np.sum(areas)
        ),
        "robust_trace_count": int(sum(row["trace_robust"] for row in rows)),
        "certified_polygon_count": int(sum(row["certified"] for row in rows)),
        "uncertain_guard_count_quantiles": {
            "minimum": int(np.min(uncertain_counts)),
            "q25": float(np.quantile(uncertain_counts, 0.25)),
            "median": float(np.median(uncertain_counts)),
            "q75": float(np.quantile(uncertain_counts, 0.75)),
            "q90": float(np.quantile(uncertain_counts, 0.90)),
            "maximum": int(np.max(uncertain_counts)),
        },
        "uncertain_timestep_count_quantiles": {
            "minimum": int(np.min(uncertain_timesteps)),
            "median": float(np.median(uncertain_timesteps)),
            "q90": float(np.quantile(uncertain_timesteps, 0.90)),
            "maximum": int(np.max(uncertain_timesteps)),
        },
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "residual_report_hash": sha256_file(residual_report_path),
        "residual_geometry_hash": sha256_file(geometry_path),
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "reference_semantics_hash": reference.semantics_hash,
        "target_semantics_hash": target.semantics_hash,
        "box_hash": box.box_hash,
        "code_revision": code_revision(root),
        "interpretation": (
            "The distribution of uncertain guards measures the minimum branching "
            "burden seen by a center-trace affine proof. It decides whether local "
            "branch enumeration is a plausible next sound solver."
        ),
    }
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_path = output_dir / f"{args.condition}_fixed_trace.json"
    write_json_immutable(output_path, report)


if __name__ == "__main__":
    main()
