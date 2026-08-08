from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.emulator import VectorizedEmulator
from transportcert.models import DenseRecurrentSNN
from transportcert.parameter_batch import ReferenceParameterSweepEmulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-report",
        default="artifacts/shd_v57_hybrid_audit_v1/hybrid_family_audit.json",
    )
    parser.add_argument("--grid-resolution", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--input-batch-size", type=int, default=64)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root", default="artifacts/shd_v58_hybrid_audit_grid_v1"
    )
    args = parser.parse_args()
    if (
        args.grid_resolution < 2
        or args.batch_size < 1
        or args.input_batch_size < 1
    ):
        raise ValueError("grid resolution and batch size must be positive")

    root = Path(__file__).resolve().parents[1]
    audit_path = root / args.audit_report
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("schema_version") not in {
        "SHDHybridFamilyAuditResult/v1",
        "SHDHybridFamilyFullAuditResult/v1",
    }:
        raise ValueError("unsupported hybrid audit report")

    output_dir = root / args.output_root
    output_path = output_dir / "hybrid_audit_grid_validation.json"
    if output_path.exists():
        raise FileExistsError(f"grid validation output exists: {output_path}")

    condition = audit["condition"]
    config = audit["config"]
    timestep_radius = float(config["relative_timestep_radius"])
    threshold_radius = float(config["relative_threshold_radius"])
    factors = np.linspace(
        -1.0, 1.0, args.grid_resolution, dtype=np.float64
    )
    normalized_timestep, normalized_threshold = np.meshgrid(
        factors, factors, indexing="ij"
    )
    normalized_timestep = normalized_timestep.ravel()
    normalized_threshold = normalized_threshold.ravel()

    semantics = primary_semantic_conditions()
    reference = semantics["reference"]
    target = semantics[condition]
    timesteps = reference.timestep * (
        1.0 + timestep_radius * normalized_timestep
    )
    threshold_scales = 1.0 + threshold_radius * normalized_threshold
    store = PackedSHD(root / args.data_root / "train.npz")

    reference_engine = VectorizedEmulator()
    family_engine = ReferenceParameterSweepEmulator(reference_engine)
    output_rows = []
    started = time.perf_counter()

    for seed in config["seeds"]:
        seed_rows = [row for row in audit["rows"] if row["seed"] == seed]
        seed_dir = root / args.artifact_root / f"seed_{seed}"
        model_path = seed_dir / "model.npz"
        model = DenseRecurrentSNN.load(model_path)
        indices = np.asarray(
            [row["dataset_index"] for row in seed_rows], dtype=np.int64
        )
        frames = store.frames(indices)
        reference_predictions = np.asarray(
            reference_engine.run(model, frames, reference).predictions,
            dtype=np.int16,
        )

        execution = family_engine.run(
            model,
            frames,
            target,
            timesteps,
            threshold_scales,
            reference_predictions,
        )
        mismatches = execution.predictions != reference_predictions[:, None]
        for position, audit_row in enumerate(seed_rows):
            output_rows.append(
                {
                    "seed": int(seed),
                    "audit_position": int(audit_row["audit_position"]),
                    "dataset_index": int(audit_row["dataset_index"]),
                    "certificate_result": bool(audit_row["certified"]),
                    "certified_parameter_fraction": float(
                        audit_row["certified_parameter_fraction"]
                    ),
                    "reference_prediction": int(reference_predictions[position]),
                    "grid_identity": not bool(np.any(mismatches[position])),
                    "counterexample_point_count": int(
                        np.count_nonzero(mismatches[position])
                    ),
                    "observed_predictions": [
                        int(value)
                        for value in np.unique(execution.predictions[position])
                    ],
                    "minimum_reference_margin": float(
                        execution.minimum_reference_margin[position]
                    ),
                }
            )
        print(f"grid validation seed={seed} complete", flush=True)

    certified_rows = [row for row in output_rows if row["certificate_result"]]
    uncertified_rows = [row for row in output_rows if not row["certificate_result"]]
    violations = [row for row in certified_rows if not row["grid_identity"]]
    grid_identity_count = sum(row["grid_identity"] for row in output_rows)
    report = {
        "schema_version": "SHDHybridAuditGridValidation/v1",
        "status": (
            "independent finite-grid falsification diagnostic; passing the grid is "
            "not a proof, while any certified-row mismatch is a soundness violation"
        ),
        "audit_report": str(audit_path.relative_to(root)).replace("\\", "/"),
        "audit_report_hash": sha256_file(audit_path),
        "audit_code_revision": audit["code_revision"],
        "condition": condition,
        "grid_resolution_per_axis": args.grid_resolution,
        "grid_point_count": len(timesteps),
        "normalized_axis_values": factors,
        "sample_count": len(output_rows),
        "certificate_count": len(certified_rows),
        "certificate_fraction": len(certified_rows) / len(output_rows),
        "grid_identity_count": grid_identity_count,
        "grid_identity_fraction": grid_identity_count / len(output_rows),
        "certified_grid_violation_count": len(violations),
        "uncertified_grid_identity_count": sum(
            row["grid_identity"] for row in uncertified_rows
        ),
        "uncertified_grid_counterexample_count": sum(
            not row["grid_identity"] for row in uncertified_rows
        ),
        "rows": output_rows,
        "seconds": time.perf_counter() - started,
        "executor": "VectorizedEmulator canonical operational semantics",
        "device": "cpu",
        "train_store_hash": store.data_hash,
        "reference_semantics_hash": reference.semantics_hash,
        "target_semantics_hash": target.semantics_hash,
        "code_revision": code_revision(root),
        "interpretation": (
            "The grid checks the same joint timestep/threshold box as the sound "
            "certificate. Its identity fraction is an empirical ceiling at this "
            "resolution. A gap between grid identity and certification measures "
            "proof or resource-budget conservatism, not observed semantic failure."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_immutable(output_path, report)


if __name__ == "__main__":
    main()
