from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from transportcert.abstract import (
    DecisionMarginFamilyCertifier,
    IntervalFamilyCertifier,
    SemanticsBox,
)
from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.models import DenseRecurrentSNN


def _count_certified(
    certifier: object,
    model: DenseRecurrentSNN,
    frames: np.ndarray,
    reference: object,
    box: SemanticsBox,
    partitions: int,
    batch_size: int,
) -> tuple[int, float]:
    certified = 0
    started = time.perf_counter()
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        if partitions == 1:
            result = certifier.certify(model, batch, reference, box)
        else:
            result = certifier.certify_partitioned(
                model,
                batch,
                reference,
                box,
                partitions,
                partitions,
            )
        certified += int(np.count_nonzero(result.certified))
    return certified, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--partitions", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root", default="artifacts/shd_v22_decision_margin"
    )
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1 or args.batch_size < 1:
        raise ValueError("sample and batch counts must be positive")
    if any(partitions < 1 for partitions in args.partitions):
        raise ValueError("partition counts must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.condition}_decision_margin.json"
    if report_path.exists():
        raise FileExistsError(f"decision-margin output exists: {report_path}")

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

    rows = []
    for partitions in args.partitions:
        interval_count, interval_seconds = _count_certified(
            IntervalFamilyCertifier(),
            model,
            frames,
            reference,
            box,
            partitions,
            args.batch_size,
        )
        margin_count, margin_seconds = _count_certified(
            DecisionMarginFamilyCertifier(),
            model,
            frames,
            reference,
            box,
            partitions,
            args.batch_size,
        )
        rows.append(
            {
                "partitions_per_axis": partitions,
                "subbox_count": partitions * partitions,
                "interval_certified_inputs": interval_count,
                "interval_certified_fraction": interval_count / len(selected_indices),
                "decision_margin_certified_inputs": margin_count,
                "decision_margin_certified_fraction": margin_count
                / len(selected_indices),
                "certified_input_gain": margin_count - interval_count,
                "interval_seconds": interval_seconds,
                "decision_margin_seconds": margin_seconds,
            }
        )
        print(
            f"decision margin seed={args.seed} partitions={partitions} "
            f"interval={interval_count}/{len(selected_indices)} "
            f"margin={margin_count}/{len(selected_indices)}",
            flush=True,
        )

    report = {
        "schema_version": "SHDDecisionMarginCertificate/v1",
        "status": (
            "sound single-member floating-point certificate; pairwise output margins "
            "retain shared-spike correlation while hidden reachability remains interval"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "relative_radius": args.radius,
        "partitions": list(args.partitions),
        "sample_count": len(selected_indices),
        "batch_size": args.batch_size,
        "sample_selection": "first entries of the frozen certificate-audit order",
        "selected_indices_hash": array_hash(selected_indices),
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "reference_semantics_hash": reference.semantics_hash,
        "target_semantics_hash": target.semantics_hash,
        "box_hash": box.box_hash,
        "rows": rows,
        "code_revision": code_revision(root),
        "interpretation": (
            "A positive gain isolates looseness caused by independently bounding output "
            "logits. Zero coverage means hidden-state/spike interval relaxation remains "
            "dominant; it does not establish true family instability."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
