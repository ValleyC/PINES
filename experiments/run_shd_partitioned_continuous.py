from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_shd_static_family import _families
from pines.abstract import IntervalFamilyCertifier, SemanticsBox
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
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root", default="artifacts/shd_v1_partitioned_continuous"
    )
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--partitions", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument(
        "--vary",
        choices=("joint", "timestep", "threshold"),
        default="joint",
    )
    args = parser.parse_args()
    if not 0 < args.radius < 1:
        raise ValueError("radius must lie strictly between zero and one")
    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    if any(value <= 0 for value in args.partitions):
        raise ValueError("partition counts must be positive")
    if tuple(sorted(set(args.partitions))) != tuple(args.partitions):
        raise ValueError("partition counts must be unique and increasing")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "partitioned_continuous_report.json"
    if output_path.exists():
        raise FileExistsError(f"partition report exists: {output_path}")

    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(seed_dir / "split_indices.npz", allow_pickle=False) as splits:
        complete_audit_indices = np.asarray(
            splits["certificate_audit"], dtype=np.int64
        )
    audit_indices = complete_audit_indices[: args.max_samples]
    inputs = store.frames(audit_indices)
    model = DenseRecurrentSNN.load(seed_dir / "model.npz")
    reference = primary_semantic_conditions()["reference"]
    discrete_box, _ = _families()["full"]
    timestep_bounds = (
        (1.0 - args.radius, 1.0 + args.radius)
        if args.vary in ("joint", "timestep")
        else (1.0, 1.0)
    )
    threshold_bounds = (
        (1.0 - args.radius, 1.0 + args.radius)
        if args.vary in ("joint", "threshold")
        else (1.0, 1.0)
    )
    box = SemanticsBox(
        base=reference,
        timestep_bounds=timestep_bounds,
        threshold_scale_bounds=threshold_bounds,
        integration_rules=discrete_box.integration_rules,
        threshold_timings=discrete_box.threshold_timings,
        reset_rules=discrete_box.reset_rules,
        synaptic_delays=discrete_box.synaptic_delays,
        output_delays=discrete_box.output_delays,
        name=f"full-continuous-{args.vary}-pm-{args.radius:.4f}",
    )
    certifier = IntervalFamilyCertifier()
    rows = []
    previous_certified: np.ndarray | None = None
    for partitions in args.partitions:
        timestep_partitions = (
            partitions if args.vary in ("joint", "timestep") else 1
        )
        threshold_partitions = (
            partitions if args.vary in ("joint", "threshold") else 1
        )
        started = time.perf_counter()
        result = certifier.certify_partitioned(
            model,
            inputs,
            reference,
            box,
            timestep_partitions=timestep_partitions,
            threshold_partitions=threshold_partitions,
        )
        elapsed = time.perf_counter() - started
        lost_from_previous = (
            0
            if previous_certified is None
            else int(np.count_nonzero(previous_certified & ~result.certified))
        )
        rows.append(
            {
                "partitions_per_axis": partitions,
                "timestep_partitions": timestep_partitions,
                "threshold_partitions": threshold_partitions,
                "subbox_count": timestep_partitions * threshold_partitions,
                "discrete_members_per_subbox": 16,
                "total_abstract_members": timestep_partitions
                * threshold_partitions
                * 16,
                "samples": len(audit_indices),
                "certified_inputs": int(np.count_nonzero(result.certified)),
                "certified_fraction": result.certified_fraction,
                "lost_certificates_from_previous_partition": lost_from_previous,
                "seconds": elapsed,
            }
        )
        previous_certified = result.certified
        print(
            f"seed={args.seed} partitions={partitions} "
            f"subboxes={timestep_partitions * threshold_partitions} "
            f"coverage={result.certified_fraction:.4f} "
            f"lost={lost_from_previous} seconds={elapsed:.2f}",
            flush=True,
        )

    report = {
        "schema_version": "SHDPartitionedContinuousExperiment/v1",
        "status": (
            "sound continuous-box partition diagnostic on a deterministic audit "
            "subset; not primary or physical evidence"
        ),
        "seed": args.seed,
        "varied_axes": args.vary,
        "relative_radius": args.radius,
        "box_hash": box.box_hash,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(seed_dir / "model.npz"),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(seed_dir / "split_indices.npz"),
        "complete_audit_samples": len(complete_audit_indices),
        "audit_samples": len(audit_indices),
        "audit_sample_indices_hash": array_hash(audit_indices),
        "subset_rule": "first max_samples entries of the frozen certificate audit split",
        "rows": rows,
        "gate_assessment": {
            "finest_partition_certifies_at_least_20_percent": bool(
                rows[-1]["certified_fraction"] >= 0.20
            ),
            "coverage_is_monotone": all(
                row["lost_certificates_from_previous_partition"] == 0
                for row in rows
            ),
        },
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, report)
    print(json.dumps(report["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
