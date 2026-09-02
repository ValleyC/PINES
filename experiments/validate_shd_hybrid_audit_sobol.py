from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from pines.artifacts import (
    array_description,
    code_revision,
    file_reference,
    write_json,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.emulator import VectorizedEmulator
from pines.models import DenseRecurrentSNN
from pines.parameter_batch import ReferenceParameterSweepEmulator


def _run_seed(task: dict[str, Any]) -> list[dict[str, Any]]:
    from threadpoolctl import threadpool_limits

    root = Path(task["root"])
    seed = int(task["seed"])
    seed_rows = task["seed_rows"]
    semantics = primary_semantic_conditions()
    reference = semantics["reference"]
    target = semantics[task["condition"]]
    store = PackedSHD(root / task["data_root"] / "train.npz")
    model = DenseRecurrentSNN.load(
        root / task["artifact_root"] / f"seed_{seed}" / "model.npz"
    )
    indices = np.asarray(
        [row["dataset_index"] for row in seed_rows], dtype=np.int64
    )
    frames = store.frames(indices)
    reference_engine = VectorizedEmulator()
    with threadpool_limits(limits=1, user_api="blas"):
        reference_predictions = np.asarray(
            reference_engine.run(model, frames, reference).predictions,
            dtype=np.int16,
        )
        execution = ReferenceParameterSweepEmulator(reference_engine).run(
            model,
            frames,
            target,
            np.asarray(task["timesteps"], dtype=np.float64),
            np.asarray(task["threshold_scales"], dtype=np.float64),
            reference_predictions,
        )
    mismatches = execution.predictions != reference_predictions[:, None]
    rows = []
    for position, audit_row in enumerate(seed_rows):
        rows.append(
            {
                "seed": seed,
                "audit_position": int(audit_row["audit_position"]),
                "dataset_index": int(audit_row["dataset_index"]),
                "certificate_result": bool(audit_row["certified"]),
                "sobol_identity": not bool(np.any(mismatches[position])),
                "counterexample_point_count": int(
                    np.count_nonzero(mismatches[position])
                ),
                "minimum_reference_margin": float(
                    execution.minimum_reference_margin[position]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--sobol-power", type=int, default=10)
    parser.add_argument("--sobol-seed", type=int, default=7319)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--input-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.sobol_power <= 20
        or args.batch_size < 1
        or args.input_batch_size < 1
        or args.workers < 1
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
    output_rows = []
    started = time.perf_counter()
    tasks = [
        {
            "root": str(root),
            "seed": int(seed),
            "seed_rows": [row for row in audit["rows"] if row["seed"] == seed],
            "condition": audit["condition"],
            "data_root": args.data_root,
            "artifact_root": args.artifact_root,
            "timesteps": timesteps,
            "threshold_scales": threshold_scales,
        }
        for seed in config["seeds"]
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        future_map = {
            executor.submit(_run_seed, task): task["seed"] for task in tasks
        }
        for future in as_completed(future_map):
            seed = future_map[future]
            output_rows.extend(future.result())
            print(f"Sobol validation seed={seed} complete", flush=True)
    output_rows.sort(
        key=lambda row: (row["seed"], row["audit_position"])
    )

    certified_rows = [row for row in output_rows if row["certificate_result"]]
    violations = [row for row in certified_rows if not row["sobol_identity"]]
    report = {
        "schema_version": "SHDHybridAuditSobolValidation/v1",
        "status": (
            "scrambled Sobol falsification diagnostic; passing sampled points is "
            "not a proof, while any certified-row mismatch is a soundness violation"
        ),
        "audit_report": str(audit_path.relative_to(root)).replace("\\", "/"),
        "audit_report_description": file_reference(audit_path),
        "audit_code_revision": audit["code_revision"],
        "condition": audit["condition"],
        "sobol_power": args.sobol_power,
        "sobol_seed": args.sobol_seed,
        "points_per_input": len(normalized_points),
        "normalized_points_reference": array_description(normalized_points),
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
        "executor": "VectorizedEmulator canonical operational semantics",
        "device": "cpu",
        "train_store_reference": PackedSHD(
            root / args.data_root / "train.npz"
        ).data_description,
        "reference_semantics": reference.semantics_description,
        "target_semantics": target.semantics_description,
        "code_revision": code_revision(root),
        "interpretation": (
            "This diagnostic complements the structured grid with a deterministic "
            "low-discrepancy interior design. It can reveal a false certificate or "
            "additional model counterexamples but cannot certify unsampled points."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)


if __name__ == "__main__":
    main()
