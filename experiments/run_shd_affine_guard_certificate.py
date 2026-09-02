from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pines.abstract import (
    DecisionMarginFamilyCertifier,
    SemanticsBox,
    partition_semantics_box,
)
from pines.affine import AffineGuardFamilyCertifier
from pines.artifacts import (
    array_description,
    code_revision,
    file_reference,
    write_json,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.models import DenseRecurrentSNN


def _evaluate_cover(
    certifier: object,
    model: DenseRecurrentSNN,
    frames: np.ndarray,
    reference: object,
    boxes: tuple[SemanticsBox, ...],
    batch_size: int,
) -> tuple[int, int, float]:
    full_cover = np.ones(len(frames), dtype=bool)
    certified_cell_inputs = 0
    started = time.perf_counter()
    for box in boxes:
        offset = 0
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            result = certifier.certify(model, batch, reference, box)
            count = len(batch)
            full_cover[offset : offset + count] &= result.certified
            certified_cell_inputs += int(np.count_nonzero(result.certified))
            offset += count
    return (
        int(np.count_nonzero(full_cover)),
        certified_cell_inputs,
        time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--partitions", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v27_affine_guard")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1 or args.batch_size < 1:
        raise ValueError("sample and batch counts must be positive")
    if any(value < 1 for value in args.partitions):
        raise ValueError("partition counts must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.condition}_affine_guard.json"
    if report_path.exists():
        raise FileExistsError(f"affine-guard output exists: {report_path}")

    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(split_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    selected_indices = audit_indices[: min(args.sample_count, len(audit_indices))]
    frames = store.frames(selected_indices)
    semantics = primary_semantic_conditions()
    if args.condition == "reference" or args.condition not in semantics:
        raise ValueError("condition must name a declared non-reference semantics")
    reference = semantics["reference"]
    target = semantics[args.condition]
    root_box = SemanticsBox(
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

    rows = []
    for partitions in args.partitions:
        boxes = partition_semantics_box(root_box, partitions, partitions)
        decision_full, decision_cells, decision_seconds = _evaluate_cover(
            DecisionMarginFamilyCertifier(),
            model,
            frames,
            reference,
            boxes,
            args.batch_size,
        )
        affine_full, affine_cells, affine_seconds = _evaluate_cover(
            AffineGuardFamilyCertifier(),
            model,
            frames,
            reference,
            boxes,
            args.batch_size,
        )
        cell_input_total = len(boxes) * len(selected_indices)
        rows.append(
            {
                "partitions_per_axis": partitions,
                "subbox_count": len(boxes),
                "decision_margin_full_cover_inputs": decision_full,
                "decision_margin_certified_cell_inputs": decision_cells,
                "decision_margin_certified_cell_input_fraction": decision_cells
                / cell_input_total,
                "affine_guard_full_cover_inputs": affine_full,
                "affine_guard_certified_cell_inputs": affine_cells,
                "affine_guard_certified_cell_input_fraction": affine_cells
                / cell_input_total,
                "affine_cell_input_gain": affine_cells - decision_cells,
                "decision_margin_seconds": decision_seconds,
                "affine_guard_seconds": affine_seconds,
            }
        )
        print(
            f"affine guard seed={args.seed} partitions={partitions} "
            f"decision_full={decision_full} affine_full={affine_full} "
            f"decision_cells={decision_cells}/{cell_input_total} "
            f"affine_cells={affine_cells}/{cell_input_total}",
            flush=True,
        )

    report = {
        "schema_version": "SHDAffineGuardCertificate/v1",
        "status": (
            "sound single-seed floating-point method-development diagnostic; affine "
            "forms retain shared timestep and threshold generators through recurrence"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "relative_radius": args.radius,
        "partitions": list(args.partitions),
        "sample_count": len(selected_indices),
        "batch_size": args.batch_size,
        "sample_selection": "first entries of the frozen certificate-audit order",
        "selected_indices_reference": array_description(selected_indices),
        "model_description": model.model_description,
        "model_file": file_reference(model_path),
        "train_store_reference": store.data_description,
        "split_indices_file": file_reference(split_path),
        "reference_semantics": reference.semantics_description,
        "target_semantics": target.semantics_description,
        "root_box_description": root_box.box_description,
        "rows": rows,
        "code_revision": code_revision(root),
        "interpretation": (
            "Full-cover certification requires every cell for an input. Cell-input "
            "coverage isolates whether the affine recurrent domain proves local boxes; "
            "it is not itself a per-input family certificate."
        ),
    }
    write_json(report_path, report)


if __name__ == "__main__":
    main()
