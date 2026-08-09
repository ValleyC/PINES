from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

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
        "--output-root", default="artifacts/shd_v1_continuous_radius_sweep"
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
    )
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if any(not 0 < radius < 1 for radius in args.radii):
        raise ValueError("radii must lie strictly between zero and one")
    if tuple(sorted(set(args.radii))) != tuple(args.radii):
        raise ValueError("radii must be unique and increasing")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "radius_sweep_report.json"
    if output_path.exists():
        raise FileExistsError(f"radius sweep report exists: {output_path}")

    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(seed_dir / "split_indices.npz", allow_pickle=False) as splits:
        complete_audit_indices = np.asarray(
            splits["certificate_audit"], dtype=np.int64
        )
    audit_indices = complete_audit_indices
    if args.max_samples is not None:
        audit_indices = audit_indices[: args.max_samples]
    inputs = store.frames(audit_indices)
    model = DenseRecurrentSNN.load(seed_dir / "model.npz")
    reference = primary_semantic_conditions()["reference"]
    certifier = IntervalFamilyCertifier()
    rows = []
    for varied_axis in ("timestep", "threshold", "joint"):
        for radius in args.radii:
            box = SemanticsBox(
                base=reference,
                timestep_bounds=(
                    (1.0 - radius, 1.0 + radius)
                    if varied_axis in ("timestep", "joint")
                    else (1.0, 1.0)
                ),
                threshold_scale_bounds=(
                    (1.0 - radius, 1.0 + radius)
                    if varied_axis in ("threshold", "joint")
                    else (1.0, 1.0)
                ),
                integration_rules=(reference.integration_rule,),
                threshold_timings=(reference.threshold_timing,),
                reset_rules=(reference.reset_rule,),
                synaptic_delays=(reference.synaptic_delay_steps,),
                output_delays=(reference.output_delay_steps,),
                name=f"reference-{varied_axis}-pm-{radius:.8g}",
            )
            started = time.perf_counter()
            result = certifier.certify(model, inputs, reference, box)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "varied_axis": varied_axis,
                    "relative_radius": radius,
                    "box_hash": box.box_hash,
                    "samples": len(audit_indices),
                    "certified_inputs": int(np.count_nonzero(result.certified)),
                    "certified_fraction": result.certified_fraction,
                    "seconds": elapsed,
                }
            )
            print(
                f"seed={args.seed} axis={varied_axis} radius={radius:.1e} "
                f"coverage={result.certified_fraction:.4f}",
                flush=True,
            )

    report = {
        "schema_version": "SHDContinuousRadiusSweep/v1",
        "status": (
            "sound reference-member radius diagnostic; not a full discrete-family "
            "or physical certificate"
        ),
        "seed": args.seed,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(seed_dir / "model.npz"),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(seed_dir / "split_indices.npz"),
        "complete_audit_samples": len(complete_audit_indices),
        "audit_samples": len(audit_indices),
        "audit_sample_indices_hash": array_hash(audit_indices),
        "subset_rule": (
            "complete frozen certificate audit split"
            if args.max_samples is None
            else "first max_samples entries of the frozen certificate audit split"
        ),
        "reference_semantics_hash": reference.semantics_hash,
        "rows": rows,
        "gate_assessment": {
            "one_percent_reference_member_certifies_at_least_20_percent": all(
                row["certified_fraction"] >= 0.20
                for row in rows
                if row["relative_radius"] == max(args.radii)
            ),
        },
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, report)
    print(json.dumps(report["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
