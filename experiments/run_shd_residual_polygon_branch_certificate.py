from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pines.abstract import SemanticsBox
from pines.affine import PolygonBranchCertifier, polygon_area
from pines.artifacts import (
    array_description,
    code_revision,
    file_reference,
    write_json,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.models import DenseRecurrentSNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--max-polygons", type=int, default=256)
    parser.add_argument("--max-branches", type=int, default=1024)
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
        "--output-root", default="artifacts/shd_v55_residual_polygon_branches"
    )
    args = parser.parse_args()
    if args.max_polygons < 1 or args.max_branches < 1:
        raise ValueError("polygon and branch budgets must be positive")

    root = Path(__file__).resolve().parents[1]
    residual_report_path = root / args.residual_report
    with residual_report_path.open("r", encoding="utf-8") as handle:
        residual_report = json.load(handle)
    if residual_report["condition"] != args.condition:
        raise ValueError("residual condition does not match")
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
    certifier = PolygonBranchCertifier()
    rows = []
    started = time.perf_counter()
    for position, polygon_index in enumerate(selected_polygon_indices):
        polygon = polygons[int(polygon_index)]
        result = certifier.certify(
            model,
            frame,
            reference,
            box,
            polygon,
            max_branches=args.max_branches,
        )
        rows.append(
            {
                "selection_position": position,
                "polygon_index": int(polygon_index),
                "normalized_area": float(areas[polygon_index]),
                "vertex_count": len(polygon),
                "certified": result.certified,
                "complete": result.complete,
                "reference_prediction": result.reference_prediction,
                "possible_predictions": list(result.possible_predictions),
                "final_branch_count": result.final_branch_count,
                "maximum_active_branches": result.maximum_active_branches,
                "total_branch_splits": result.total_branch_splits,
                "maximum_uncertain_neurons_at_step": (
                    result.maximum_uncertain_neurons_at_step
                ),
                "first_cap_timestep": result.first_cap_timestep,
            }
        )
        if (position + 1) % 32 == 0 or position + 1 == len(
            selected_polygon_indices
        ):
            print(
                f"polygon branches={position + 1}/{len(selected_polygon_indices)} "
                f"certified={sum(row['certified'] for row in rows)} "
                f"incomplete={sum(not row['complete'] for row in rows)}",
                flush=True,
            )

    certified_area = sum(row["normalized_area"] for row in rows if row["certified"])
    complete_area = sum(row["normalized_area"] for row in rows if row["complete"])
    branch_counts = np.asarray(
        [row["maximum_active_branches"] for row in rows], dtype=np.int64
    )
    source_certified_fraction = float(
        residual_report["rows"][0]["certified_parameter_fraction"]
    )
    root_area = 4.0
    report = {
        "schema_version": "SHDResidualPolygonBranchCertificate/v1",
        "status": (
            "sound polygon-local spike-branch enumeration; every guard-uncertain "
            "outcome is propagated and only complete all-reference branches certify"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "dataset_index": dataset_index,
        "relative_radius": radius,
        "source_polygon_count": len(polygons),
        "selected_polygon_count": len(selected_polygon_indices),
        "selection_rule": "largest normalized area first",
        "selected_polygon_indices": selected_polygon_indices,
        "selected_polygon_indices_reference": array_description(selected_polygon_indices),
        "selected_area_fraction_of_residue": float(
            np.sum(areas[selected_polygon_indices]) / np.sum(areas)
        ),
        "max_branches": args.max_branches,
        "complete_polygon_count": int(sum(row["complete"] for row in rows)),
        "certified_polygon_count": int(sum(row["certified"] for row in rows)),
        "selected_certified_area_fraction": float(
            certified_area / np.sum(areas[selected_polygon_indices])
        ),
        "selected_complete_area_fraction": float(
            complete_area / np.sum(areas[selected_polygon_indices])
        ),
        "added_root_parameter_fraction": float(certified_area / root_area),
        "combined_root_parameter_fraction": float(
            source_certified_fraction + certified_area / root_area
        ),
        "maximum_active_branch_quantiles": {
            "minimum": int(np.min(branch_counts)),
            "median": float(np.median(branch_counts)),
            "q90": float(np.quantile(branch_counts, 0.90)),
            "q99": float(np.quantile(branch_counts, 0.99)),
            "maximum": int(np.max(branch_counts)),
        },
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "residual_report_description": file_reference(residual_report_path),
        "residual_geometry_reference": file_reference(geometry_path),
        "model_description": model.model_description,
        "model_file": file_reference(model_path),
        "train_store_reference": store.data_description,
        "split_indices_file": file_reference(split_path),
        "reference_semantics": reference.semantics_description,
        "target_semantics": target.semantics_description,
        "box_description": box.box_description,
        "code_revision": code_revision(root),
        "interpretation": (
            "A complete result covers the entire selected polygon, including "
            "branches that may be infeasible. Incomplete cap hits are inconclusive; "
            "a non-reference possible prediction prevents certification even if "
            "that relaxed branch is not sampled."
        ),
    }
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_path = output_dir / f"{args.condition}_polygon_branches.json"
    write_json(output_path, report)


if __name__ == "__main__":
    main()
