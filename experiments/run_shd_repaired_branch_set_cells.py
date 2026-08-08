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


METHODS = (
    "no_repair",
    "certificate_directed",
    "guard_margin",
    "family_margin",
    "logit_only",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--radius", type=float, default=1e-2)
    parser.add_argument("--partitions", type=int, nargs="+", default=(64, 128))
    parser.add_argument("--max-branches", type=int, default=65536)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--repair-root", default="artifacts/shd_v3_repairs_matched")
    parser.add_argument("--guard-root", default="artifacts/shd_v15_guard_margin")
    parser.add_argument("--family-root", default="artifacts/shd_v18_family_margin")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output-root", default="artifacts/shd_v13_repaired_branch_cells")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1 or args.max_branches < 1:
        raise ValueError("sample and branch limits must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    repair_seed_dir = root / args.repair_root / f"seed_{args.seed}" / args.condition
    guard_seed_dir = root / args.guard_root / f"seed_{args.seed}" / args.condition
    family_seed_dir = root / args.family_root / f"seed_{args.seed}" / args.condition
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.condition}_branch_cells.json"
    if report_path.exists():
        raise FileExistsError(f"repaired branch-cell output exists: {report_path}")

    source_model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    source_model = DenseRecurrentSNN.load(source_model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(split_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    selected_indices = audit_indices[: min(args.sample_count, len(audit_indices))]
    frames = store.frames(selected_indices)
    semantics = primary_semantic_conditions()
    reference = semantics["reference"]
    target = semantics[args.condition]
    source_predictions = VectorizedEmulator().run(
        source_model, frames, reference
    ).predictions
    full_box = SemanticsBox(
        base=target,
        timestep_bounds=(1.0 - args.radius, 1.0 + args.radius),
        threshold_scale_bounds=(1.0 - args.radius, 1.0 + args.radius),
        integration_rules=(target.integration_rule,),
        threshold_timings=(target.threshold_timing,),
        reset_rules=(target.reset_rule,),
        synaptic_delays=(target.synaptic_delay_steps,),
        output_delays=(target.output_delay_steps,),
        name=f"{args.condition}-joint-pm-{args.radius:.6g}",
    )
    certifier = BranchSetMemberCertifier()
    rows = []
    model_metadata = {}
    for method in args.methods:
        if method == "no_repair":
            deployed_model = source_model
            model_path = source_model_path
            repair_report_hash = None
        else:
            method_dir = (
                guard_seed_dir / method
                if method == "guard_margin"
                else family_seed_dir / method
                if method == "family_margin"
                else repair_seed_dir / method
            )
            model_path = method_dir / "repaired_model.npz"
            repair_report_path = method_dir / "repair_report.json"
            deployed_model = DenseRecurrentSNN.load(model_path)
            repair_report_hash = sha256_file(repair_report_path)
        model_metadata[method] = {
            "model_hash": deployed_model.model_hash,
            "model_artifact_hash": sha256_file(model_path),
            "repair_report_hash": repair_report_hash,
        }
        for merge_equivalent in (False, True):
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
                                deployed_model,
                                frames[input_index : input_index + 1],
                                boxes[cell_index],
                                int(source_predictions[input_index]),
                                max_branches=args.max_branches,
                                merge_equivalent=merge_equivalent,
                            )
                        )
                    elapsed = time.perf_counter() - started
                    abort_steps = [
                        result.aborted_step
                        for result in results
                        if result.aborted_step is not None
                    ]
                    rows.append(
                        {
                            "method": method,
                            "merge_equivalent": merge_equivalent,
                            "partitions_per_axis": partitions,
                            "subbox_count_for_full_cover": partitions * partitions,
                            "cell": cell_name,
                            "cell_index": cell_index,
                            "samples": len(selected_indices),
                            "complete_inputs": sum(result.complete for result in results),
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
                        f"repaired branch cells seed={args.seed} method={method} "
                        f"merge={merge_equivalent} partitions={partitions} cell={cell_name} "
                        f"complete={sum(result.complete for result in results)}/{len(results)}",
                        flush=True,
                    )

    report = {
        "schema_version": "SHDRepairedBranchSetCellDiagnostic/v2",
        "status": (
            "single-seed representative-cell sound-analysis diagnostic; cells do not "
            "cover the full family and cap failures are inconclusive"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "relative_radius": args.radius,
        "partitions": list(args.partitions),
        "max_branches": args.max_branches,
        "methods": list(args.methods),
        "sample_count": len(selected_indices),
        "selected_indices_hash": array_hash(selected_indices),
        "source_model_hash": source_model.model_hash,
        "source_model_artifact_hash": sha256_file(source_model_path),
        "deployed_models": model_metadata,
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "reference_semantics_hash": reference.semantics_hash,
        "target_semantics_hash": target.semantics_hash,
        "full_box_hash": full_box.box_hash,
        "rows": rows,
        "code_revision": code_revision(root),
        "interpretation": (
            "Comparing identical representative cells tests whether a selected repair "
            "reduces explicit abstract branch complexity. Completion is necessary but not "
            "sufficient for a full-family certificate because most cells are not executed."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
