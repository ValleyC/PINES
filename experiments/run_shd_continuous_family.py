from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_shd_static_family import _families, _ordering
from transportcert.abstract import IntervalFamilyCertifier, SemanticsBox
from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.models import DenseRecurrentSNN
from transportcert.torch_emulator import TorchEmulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root", default="artifacts/shd_v1_continuous_family"
    )
    parser.add_argument("--radii", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(not 0 < radius < 1 for radius in args.radii):
        raise ValueError("radii must lie strictly between zero and one")
    if args.validation_samples <= 0:
        raise ValueError("validation-samples must be positive")

    import torch

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "continuous_family_report.json"
    if output_path.exists():
        raise FileExistsError(f"continuous-family report exists: {output_path}")

    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(seed_dir / "split_indices.npz", allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    if args.validation_samples > len(audit_indices):
        raise ValueError("validation subset exceeds audit split")
    inputs = store.frames(audit_indices)
    validation_inputs = inputs[: args.validation_samples]
    model = DenseRecurrentSNN.load(seed_dir / "model.npz")
    reference = primary_semantic_conditions()["reference"]
    discrete_box, _ = _families()["full"]
    certifier = IntervalFamilyCertifier()
    emulator = TorchEmulator(device=args.device, dtype=torch.float64)
    reference_predictions = (
        emulator.run(model, validation_inputs, reference)
        .predictions.detach()
        .cpu()
        .numpy()
    )

    rows = []
    for radius in args.radii:
        box = SemanticsBox(
            base=reference,
            timestep_bounds=(1.0 - radius, 1.0 + radius),
            threshold_scale_bounds=(1.0 - radius, 1.0 + radius),
            integration_rules=discrete_box.integration_rules,
            threshold_timings=discrete_box.threshold_timings,
            reset_rules=discrete_box.reset_rules,
            synaptic_delays=discrete_box.synaptic_delays,
            output_delays=discrete_box.output_delays,
            name=f"full-continuous-pm-{radius:.4f}",
        )
        started = time.perf_counter()
        static = certifier.certify(model, inputs, reference, box)
        static_seconds = time.perf_counter() - started

        sampled_agreement = np.ones(args.validation_samples, dtype=bool)
        sampled_runs = 0
        sampled_started = time.perf_counter()
        for integration, timing, reset, delay, dt, threshold_scale in itertools.product(
            box.integration_rules,
            box.threshold_timings,
            box.reset_rules,
            box.synaptic_delays,
            box.timestep_bounds,
            box.threshold_scale_bounds,
        ):
            semantics = replace(
                reference,
                timestep=dt,
                integration_rule=integration,
                threshold_timing=timing,
                update_ordering=_ordering(timing),
                reset_rule=reset,
                synaptic_delay_steps=delay,
            )
            candidate = model.with_parameters(
                threshold=model.threshold * threshold_scale,
                name=f"{model.name}-threshold-scale-{threshold_scale:.6f}",
            )
            predictions = (
                emulator.run(candidate, validation_inputs, semantics)
                .predictions.detach()
                .cpu()
                .numpy()
            )
            sampled_agreement &= predictions == reference_predictions
            sampled_runs += 1
        sampled_seconds = time.perf_counter() - sampled_started
        static_validation = static.certified[: args.validation_samples]
        sampled_counterexamples = int(
            np.count_nonzero(static_validation & ~sampled_agreement)
        )
        rows.append(
            {
                "relative_radius": radius,
                "timestep_bounds": list(box.timestep_bounds),
                "threshold_scale_bounds": list(box.threshold_scale_bounds),
                "box_hash": box.box_hash,
                "discrete_member_count": 16,
                "audit_samples": len(audit_indices),
                "static_certified_inputs": int(np.count_nonzero(static.certified)),
                "static_certified_fraction": static.certified_fraction,
                "validation_samples": args.validation_samples,
                "sampled_corner_runs": sampled_runs,
                "sampled_corner_agreement_inputs": int(
                    np.count_nonzero(sampled_agreement)
                ),
                "sampled_corner_agreement_fraction": float(
                    np.mean(sampled_agreement)
                ),
                "sampled_corner_counterexamples_to_static": sampled_counterexamples,
                "static_seconds": static_seconds,
                "sampled_corner_seconds": sampled_seconds,
            }
        )
        print(
            f"seed={args.seed} radius={radius:.3f} "
            f"static={static.certified_fraction:.4f} "
            f"corner_agreement={np.mean(sampled_agreement):.4f} "
            f"counterexamples={sampled_counterexamples}",
            flush=True,
        )

    report = {
        "schema_version": "SHDContinuousFamilyExperiment/v1",
        "status": (
            "software sound-box evidence with endpoint-grid validation; endpoint "
            "agreement is diagnostic and not a proof over the continuous interior"
        ),
        "seed": args.seed,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(seed_dir / "model.npz"),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(seed_dir / "split_indices.npz"),
        "audit_sample_indices_hash": array_hash(audit_indices),
        "audit_samples": len(audit_indices),
        "validation_sample_indices_hash": array_hash(
            audit_indices[: args.validation_samples]
        ),
        "validation_samples": args.validation_samples,
        "validation_rule": (
            "first validation_samples entries of the frozen audit split; all four "
            "timestep/threshold box corners times all 16 discrete members"
        ),
        "reference_semantics_hash": reference.semantics_hash,
        "device": args.device,
        "torch_dtype": "float32",
        "rows": rows,
        "gate_assessment": {
            "every_radius_certifies_at_least_20_percent": all(
                row["static_certified_fraction"] >= 0.20 for row in rows
            ),
            "no_sampled_corner_counterexample": all(
                row["sampled_corner_counterexamples_to_static"] == 0 for row in rows
            ),
        },
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, report)
    print(json.dumps(report["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
