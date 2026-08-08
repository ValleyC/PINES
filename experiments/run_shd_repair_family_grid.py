from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

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


METHODS = (
    "no_repair",
    "certificate_directed",
    "guard_margin",
    "logit_only",
    "global_threshold",
    "per_platform_qat",
)
GRID_RESOLUTION = 9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--radius", type=float, default=1e-2)
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--repair-root", default="artifacts/shd_v3_repairs_matched")
    parser.add_argument("--guard-root", default="artifacts/shd_v15_guard_margin")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output-root", default="artifacts/shd_v12_repair_family_grid")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1:
        raise ValueError("sample count must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    repair_seed_dir = root / args.repair_root / f"seed_{args.seed}" / args.condition
    guard_seed_dir = root / args.guard_root / f"seed_{args.seed}" / args.condition
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.condition}_family_grid.json"
    if report_path.exists():
        raise FileExistsError(f"repair-family grid output exists: {report_path}")

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

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = TorchEmulator(device=device, dtype=torch.float32)
    source_reference_predictions = np.asarray(
        emulator.run(source_model, frames, reference).numpy().predictions,
        dtype=np.int16,
    )
    factors = np.linspace(
        1.0 - args.radius, 1.0 + args.radius, GRID_RESOLUTION, dtype=np.float64
    )
    center_index = (GRID_RESOLUTION // 2) * GRID_RESOLUTION + GRID_RESOLUTION // 2
    rows = []
    for method in args.methods:
        if method == "no_repair":
            deployed_model = source_model
            model_path = source_model_path
            repair_report_path = None
            repair_metadata = {
                "labels_used": "none",
                "optimization_steps": 0,
                "trainable_parameters": 0,
            }
        else:
            method_dir = (
                guard_seed_dir / method
                if method == "guard_margin"
                else repair_seed_dir / method
            )
            model_path = method_dir / "repaired_model.npz"
            repair_report_path = method_dir / "repair_report.json"
            deployed_model = DenseRecurrentSNN.load(model_path)
            repair_report = json.loads(repair_report_path.read_text(encoding="utf-8"))
            repair_metadata = {
                "labels_used": repair_report["labels_used"],
                "optimization_steps": int(repair_report["optimization_steps"]),
                "trainable_parameters": int(repair_report["trainable_parameters"]),
            }

        predictions = np.empty(
            (GRID_RESOLUTION * GRID_RESOLUTION, len(selected_indices)),
            dtype=np.int16,
        )
        point = 0
        for timestep_factor in factors:
            target_semantics = replace(
                target, timestep=reference.timestep * float(timestep_factor)
            )
            for threshold_factor in factors:
                candidate_model = deployed_model.with_parameters(
                    threshold=deployed_model.threshold * float(threshold_factor)
                )
                predictions[point] = np.asarray(
                    emulator.run(candidate_model, frames, target_semantics)
                    .numpy()
                    .predictions,
                    dtype=np.int16,
                )
                point += 1
        matches = predictions == source_reference_predictions[None, :]
        grid_identity = np.all(matches, axis=0)
        unique_predictions = np.asarray(
            [len(np.unique(predictions[:, index])) for index in range(len(selected_indices))]
        )
        rows.append(
            {
                "method": method,
                **repair_metadata,
                "deployed_model_hash": deployed_model.model_hash,
                "deployed_model_artifact_hash": sha256_file(model_path),
                "repair_report_hash": (
                    None
                    if repair_report_path is None
                    else sha256_file(repair_report_path)
                ),
                "center_target_identity_inputs": int(np.count_nonzero(matches[center_index])),
                "center_target_identity_fraction": float(np.mean(matches[center_index])),
                "grid_family_identity_inputs": int(np.count_nonzero(grid_identity)),
                "grid_family_identity_fraction": float(np.mean(grid_identity)),
                "continuous_grid_identity_cost": float(
                    np.mean(matches[center_index]) - np.mean(grid_identity)
                ),
                "grid_pair_disagreement_fraction": float(1.0 - np.mean(matches)),
                "mean_unique_predictions": float(np.mean(unique_predictions)),
                "max_unique_predictions": int(np.max(unique_predictions)),
            }
        )
        print(
            f"repair family grid seed={args.seed} method={method} "
            f"identity={np.mean(grid_identity):.4f}",
            flush=True,
        )

    report = {
        "schema_version": "SHDRepairFamilyGridDiagnostic/v2",
        "status": (
            "finite-grid post-repair diagnostic; identity is measured against the original "
            "source/reference predictions and is not a continuous-family certificate"
        ),
        "seed": args.seed,
        "condition": args.condition,
        "relative_radius": args.radius,
        "grid_resolution": GRID_RESOLUTION,
        "grid_points": GRID_RESOLUTION * GRID_RESOLUTION,
        "sample_count": len(selected_indices),
        "sample_selection": "first entries of the frozen certificate-audit order",
        "selected_indices_hash": array_hash(selected_indices),
        "source_model_hash": source_model.model_hash,
        "source_model_artifact_hash": sha256_file(source_model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "reference_semantics_hash": reference.semantics_hash,
        "target_semantics_hash": target.semantics_hash,
        "methods": list(args.methods),
        "rows": rows,
        "device": device,
        "torch_version": torch.__version__,
        "code_revision": code_revision(root),
        "interpretation": (
            "A repair that raises grid-family identity increases the empirical ceiling for "
            "a subsequent unchanged-argmax certificate. The grid cannot establish coverage "
            "between sampled points."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
