from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.models import DenseRecurrentSNN
from transportcert.parameter_batch import TorchParameterBatchEmulator
from transportcert.torch_emulator import TorchEmulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--sobol-power", type=int, default=10)
    parser.add_argument("--sobol-seed", type=int, default=7319)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--input-batch-size", type=int, default=32)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.sobol_power <= 20
        or args.batch_size < 1
        or args.input_batch_size < 1
    ):
        raise ValueError("invalid Sobol or batch budget")

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
    output_path = output_dir / "hybrid_audit_sobol_validation.json"
    if output_path.exists():
        raise FileExistsError(f"Sobol validation output exists: {output_path}")

    normalized_points = (
        2.0
        * qmc.Sobol(d=2, scramble=True, seed=args.sobol_seed).random_base2(
            m=args.sobol_power
        )
        - 1.0
    )
    config = audit["config"]
    timestep_radius = float(config["relative_timestep_radius"])
    threshold_radius = float(config["relative_threshold_radius"])
    semantics = primary_semantic_conditions()
    reference = semantics["reference"]
    target = semantics[audit["condition"]]
    timesteps = reference.timestep * (
        1.0 + timestep_radius * normalized_points[:, 0]
    )
    threshold_scales = 1.0 + threshold_radius * normalized_points[:, 1]
    store = PackedSHD(root / args.data_root / "train.npz")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if target.state_format.kind == "float32" else torch.float64
    reference_engine = TorchEmulator(device=device, dtype=dtype)
    family_engine = TorchParameterBatchEmulator(device=device, dtype=dtype)
    output_rows = []
    started = time.perf_counter()
    for seed in config["seeds"]:
        seed_rows = [row for row in audit["rows"] if row["seed"] == seed]
        model = DenseRecurrentSNN.load(
            root / args.artifact_root / f"seed_{seed}" / "model.npz"
        )
        indices = np.asarray(
            [row["dataset_index"] for row in seed_rows], dtype=np.int64
        )
        frames = store.frames(indices)
        reference_predictions = np.asarray(
            reference_engine.run(model, frames, reference).numpy().predictions,
            dtype=np.int16,
        )
        execution = family_engine.run_cartesian(
            model,
            frames,
            target,
            timesteps,
            threshold_scales,
            input_batch_size=args.input_batch_size,
            parameter_batch_size=args.batch_size,
        )
        row_indices = np.arange(len(seed_rows))[:, None]
        point_indices = np.arange(len(timesteps))[None, :]
        competing = execution.final_logits.copy()
        competing[
            row_indices,
            point_indices,
            reference_predictions[:, None],
        ] = -np.inf
        reference_logits = execution.final_logits[
            row_indices,
            point_indices,
            reference_predictions[:, None],
        ]
        margins = reference_logits - np.max(competing, axis=2)
        mismatches = execution.predictions != reference_predictions[:, None]
        for position, audit_row in enumerate(seed_rows):
            output_rows.append(
                {
                    "seed": int(seed),
                    "audit_position": int(audit_row["audit_position"]),
                    "dataset_index": int(audit_row["dataset_index"]),
                    "certificate_result": bool(audit_row["certified"]),
                    "sobol_identity": not bool(np.any(mismatches[position])),
                    "counterexample_point_count": int(
                        np.count_nonzero(mismatches[position])
                    ),
                    "minimum_reference_margin": float(
                        np.min(margins[position])
                    ),
                }
            )
        print(f"Sobol validation seed={seed} complete", flush=True)

    certified_rows = [row for row in output_rows if row["certificate_result"]]
    violations = [row for row in certified_rows if not row["sobol_identity"]]
    report = {
        "schema_version": "SHDHybridAuditSobolValidation/v1",
        "status": (
            "scrambled Sobol falsification diagnostic; passing sampled points is "
            "not a proof, while any certified-row mismatch is a soundness violation"
        ),
        "audit_report": str(audit_path.relative_to(root)).replace("\\", "/"),
        "audit_report_hash": sha256_file(audit_path),
        "audit_code_revision": audit["code_revision"],
        "condition": audit["condition"],
        "sobol_power": args.sobol_power,
        "sobol_seed": args.sobol_seed,
        "points_per_input": len(normalized_points),
        "normalized_points_hash": array_hash(normalized_points),
        "sample_count": len(output_rows),
        "certificate_count": len(certified_rows),
        "certified_sobol_violation_count": len(violations),
        "sobol_identity_count": sum(row["sobol_identity"] for row in output_rows),
        "sobol_identity_fraction": float(
            np.mean([row["sobol_identity"] for row in output_rows])
        ),
        "uncertified_sobol_identity_count": sum(
            not row["certificate_result"] and row["sobol_identity"]
            for row in output_rows
        ),
        "uncertified_sobol_counterexample_count": sum(
            not row["certificate_result"] and not row["sobol_identity"]
            for row in output_rows
        ),
        "rows": output_rows,
        "seconds": time.perf_counter() - started,
        "device": device,
        "torch_version": torch.__version__,
        "train_store_hash": store.data_hash,
        "reference_semantics_hash": reference.semantics_hash,
        "target_semantics_hash": target.semantics_hash,
        "code_revision": code_revision(root),
        "interpretation": (
            "This diagnostic complements the structured grid with a deterministic "
            "low-discrepancy interior design. It can reveal a false certificate or "
            "additional model counterexamples but cannot certify unsampled points."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_immutable(output_path, report)


if __name__ == "__main__":
    main()
