from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from transportcert.abstract import SemanticsBox, partition_semantics_box
from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.branch_set import BranchSetMemberCertifier
from transportcert.emulator import VectorizedEmulator
from transportcert.models import DenseRecurrentSNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--radius", type=float, default=1e-2)
    parser.add_argument("--partitions", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--max-branches", type=int, default=4096)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v6_branch_set_partition")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1 or args.max_branches < 1:
        raise ValueError("sample and branch limits must be positive")
    if any(value < 1 for value in args.partitions):
        raise ValueError("partition counts must be positive")
    if tuple(sorted(set(args.partitions))) != tuple(args.partitions):
        raise ValueError("partitions must be unique and increasing")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "branch_set_partition.json"
    if report_path.exists():
        raise FileExistsError(f"branch-set output exists: {report_path}")

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
        certified = np.ones(len(selected_indices), dtype=bool)
        all_cells_complete = np.ones(len(selected_indices), dtype=bool)
        saw_alternative_prediction = np.zeros(len(selected_indices), dtype=bool)
        evaluated_cells = np.zeros(len(selected_indices), dtype=np.int64)
        maximum_branches = np.zeros(len(selected_indices), dtype=np.int64)
        earliest_abort_step = np.full(len(selected_indices), 10**9, dtype=np.int64)
        started = time.perf_counter()
        for input_index in range(len(selected_indices)):
            for box in boxes:
                result = certifier.certify(
                    model,
                    frames[input_index : input_index + 1],
                    box,
                    int(reference_predictions[input_index]),
                    max_branches=args.max_branches,
                )
                evaluated_cells[input_index] += 1
                maximum_branches[input_index] = max(
                    maximum_branches[input_index], result.maximum_state_branches
                )
                if not result.complete:
                    all_cells_complete[input_index] = False
                    certified[input_index] = False
                    if result.aborted_step is not None:
                        earliest_abort_step[input_index] = min(
                            earliest_abort_step[input_index], result.aborted_step
                        )
                    break
                if not result.certified:
                    saw_alternative_prediction[input_index] = True
                    certified[input_index] = False
                    break
            print(
                f"branch set seed={args.seed} partitions={partitions} "
                f"inputs={input_index + 1}/{len(selected_indices)}",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        finite_abort_steps = earliest_abort_step[earliest_abort_step < 10**9]
        rows.append(
            {
                "partitions_per_axis": partitions,
                "subbox_count": len(boxes),
                "samples": len(selected_indices),
                "certified_inputs": int(np.count_nonzero(certified)),
                "certified_fraction": float(np.mean(certified)),
                "all_cells_complete_inputs": int(
                    np.count_nonzero(all_cells_complete)
                ),
                "alternative_prediction_abstract_inputs": int(
                    np.count_nonzero(saw_alternative_prediction)
                ),
                "branch_cap_inputs": int(np.count_nonzero(~all_cells_complete)),
                "mean_evaluated_cells": float(np.mean(evaluated_cells)),
                "maximum_state_branches_observed": int(np.max(maximum_branches)),
                "median_earliest_abort_step": (
                    None
                    if len(finite_abort_steps) == 0
                    else float(np.median(finite_abort_steps))
                ),
                "seconds": elapsed,
            }
        )
        print(
            f"seed={args.seed} partitions={partitions} "
            f"coverage={np.mean(certified):.4f} seconds={elapsed:.2f}",
            flush=True,
        )

    report = {
        "schema_version": "SHDBranchSetPartitionDiagnostic/v1",
        "status": (
            "sound reference-member branch-set diagnostic on a deterministic audit "
            "subset; branch-cap failures are inconclusive and this is not physical evidence"
        ),
        "seed": args.seed,
        "relative_radius": args.radius,
        "max_branches": args.max_branches,
        "partitions": list(args.partitions),
        "sample_count": len(selected_indices),
        "sample_selection": "first entries of the frozen certificate-audit order",
        "selected_indices_hash": array_hash(selected_indices),
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "reference_semantics_hash": reference.semantics_hash,
        "box_hash": full_box.box_hash,
        "rows": rows,
        "code_revision": code_revision(root),
        "interpretation": (
            "The analyzer separates possible spike vectors and merges only states with "
            "identical current spikes and accumulated logits. Spatial partitioning may "
            "reduce branch-cap failures; alternative abstract predictions are sound "
            "possibilities but need not be concretely feasible."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
