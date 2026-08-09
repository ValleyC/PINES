from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pines.abstract import SemanticsBox, partition_semantics_box
from pines.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.branch_set import BranchSetMemberCertifier
from pines.emulator import VectorizedEmulator
from pines.models import DenseRecurrentSNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--radius", type=float, default=1e-2)
    parser.add_argument("--partitions", type=int, nargs="+", default=(32, 64, 128))
    parser.add_argument("--max-branches", type=int, default=65536)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v10_branch_set_cells")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1 or args.max_branches < 1:
        raise ValueError("sample and branch limits must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "merged" if args.merge else "unmerged"
    report_path = output_dir / f"branch_set_cells_{suffix}.json"
    if report_path.exists():
        raise FileExistsError(f"branch-set cell output exists: {report_path}")

    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(split_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    selected_indices = audit_indices[: min(args.sample_count, len(audit_indices))]
    frames = store.frames(selected_indices)
    reference = primary_semantic_conditions()["reference"]
    reference_predictions = VectorizedEmulator().run(
        model, frames, reference
    ).predictions
    full_box = SemanticsBox(
        base=reference,
        timestep_bounds=(1.0 - args.radius, 1.0 + args.radius),
        threshold_scale_bounds=(1.0 - args.radius, 1.0 + args.radius),
        integration_rules=(reference.integration_rule,),
        threshold_timings=(reference.threshold_timing,),
        reset_rules=(reference.reset_rule,),
        synaptic_delays=(0,),
        output_delays=(0,),
        name=f"reference-joint-pm-{args.radius:.6g}",
    )
    certifier = BranchSetMemberCertifier()
    rows = []
    for partitions in args.partitions:
        boxes = partition_semantics_box(full_box, partitions, partitions)
        cell_indices = {
            "lower_left": 0,
            "near_reference_lower": (partitions // 2 - 1) * partitions
            + partitions // 2
            - 1,
            "near_reference_upper": (partitions // 2) * partitions
            + partitions // 2,
        }
        for cell_name, cell_index in cell_indices.items():
            results = []
            started = time.perf_counter()
            for input_index in range(len(selected_indices)):
                results.append(
                    certifier.certify(
                        model,
                        frames[input_index : input_index + 1],
                        boxes[cell_index],
                        int(reference_predictions[input_index]),
                        max_branches=args.max_branches,
                        merge_equivalent=args.merge,
                    )
                )
            elapsed = time.perf_counter() - started
            complete = [result for result in results if result.complete]
            abort_steps = [
                result.aborted_step
                for result in results
                if result.aborted_step is not None
            ]
            rows.append(
                {
                    "partitions_per_axis": partitions,
                    "subbox_count_for_full_cover": partitions * partitions,
                    "cell": cell_name,
                    "cell_index": cell_index,
                    "cell_timestep_bounds": list(boxes[cell_index].timestep_bounds),
                    "cell_threshold_bounds": list(
                        boxes[cell_index].threshold_scale_bounds
                    ),
                    "samples": len(selected_indices),
                    "complete_inputs": len(complete),
                    "complete_fraction": len(complete) / len(selected_indices),
                    "certified_inputs": sum(result.certified for result in results),
                    "maximum_state_branches_observed": max(
                        result.maximum_state_branches for result in results
                    ),
                    "median_abort_step": (
                        None if not abort_steps else float(np.median(abort_steps))
                    ),
                    "seconds": elapsed,
                }
            )
            print(
                f"cell sweep seed={args.seed} partitions={partitions} cell={cell_name} "
                f"complete={len(complete)}/{len(selected_indices)} seconds={elapsed:.2f}",
                flush=True,
            )

    report = {
        "schema_version": "SHDBranchSetCellSweep/v1",
        "status": (
            "representative-cell complexity diagnostic only; it neither covers the "
            "continuous box nor constitutes a family certificate"
        ),
        "seed": args.seed,
        "relative_radius": args.radius,
        "partitions": list(args.partitions),
        "max_branches": args.max_branches,
        "merge_equivalent": args.merge,
        "sample_count": len(selected_indices),
        "selected_indices_hash": array_hash(selected_indices),
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "reference_semantics_hash": reference.semantics_hash,
        "full_box_hash": full_box.box_hash,
        "rows": rows,
        "code_revision": code_revision(root),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
