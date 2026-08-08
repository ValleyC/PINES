from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.models import DenseRecurrentSNN
from transportcert.torch_emulator import TorchEmulator


RADII = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def _variants(radius: float) -> dict[str, tuple[tuple[float, float], ...]]:
    lower = 1.0 - radius
    upper = 1.0 + radius
    return {
        "timestep": ((lower, 1.0), (upper, 1.0)),
        "threshold": ((1.0, lower), (1.0, upper)),
        "joint_corners": (
            (lower, lower),
            (lower, upper),
            (upper, lower),
            (upper, upper),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root", default="artifacts/shd_v3_branch_stability"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "branch_stability.json"
    if report_path.exists():
        raise FileExistsError(f"branch-stability output exists: {report_path}")

    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(split_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    reference = primary_semantic_conditions()["reference"]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = TorchEmulator(device=device, dtype=torch.float32)
    counts = {
        (radius, axis): {
            "trace_stable": 0,
            "prediction_stable": 0,
            "prediction_stable_trace_changed": 0,
            "max_spike_hamming_sum": 0.0,
        }
        for radius in RADII
        for axis in ("timestep", "threshold", "joint_corners")
    }

    for start in range(0, len(audit_indices), args.batch_size):
        batch_indices = audit_indices[start : start + args.batch_size]
        frames = store.frames(batch_indices)
        center = emulator.run(model, frames, reference).numpy()
        center_spikes = np.asarray(center.spikes)
        center_predictions = np.asarray(center.predictions)
        for radius in RADII:
            for axis, variants in _variants(radius).items():
                trace_stable = np.ones(len(batch_indices), dtype=bool)
                prediction_stable = np.ones(len(batch_indices), dtype=bool)
                max_hamming = np.zeros(len(batch_indices), dtype=np.float64)
                for timestep_factor, threshold_factor in variants:
                    target_semantics = replace(
                        reference,
                        timestep=reference.timestep * timestep_factor,
                    )
                    target_model = model.with_parameters(
                        threshold=model.threshold * threshold_factor
                    )
                    target = emulator.run(
                        target_model, frames, target_semantics
                    ).numpy()
                    spike_difference = np.asarray(target.spikes) != center_spikes
                    per_input_changed = np.any(spike_difference, axis=(1, 2))
                    trace_stable &= ~per_input_changed
                    prediction_stable &= (
                        np.asarray(target.predictions) == center_predictions
                    )
                    hamming = np.mean(spike_difference, axis=(1, 2))
                    max_hamming = np.maximum(max_hamming, hamming)
                if np.any(trace_stable & ~prediction_stable):
                    raise AssertionError(
                        "identical spike traces produced different predictions"
                    )
                values = counts[(radius, axis)]
                values["trace_stable"] += int(np.count_nonzero(trace_stable))
                values["prediction_stable"] += int(
                    np.count_nonzero(prediction_stable)
                )
                values["prediction_stable_trace_changed"] += int(
                    np.count_nonzero(prediction_stable & ~trace_stable)
                )
                values["max_spike_hamming_sum"] += float(np.sum(max_hamming))
        print(
            f"branch stability seed={args.seed} samples={min(start + args.batch_size, len(audit_indices))}/{len(audit_indices)}",
            flush=True,
        )

    rows = []
    for radius in RADII:
        for axis in ("timestep", "threshold", "joint_corners"):
            values = counts[(radius, axis)]
            rows.append(
                {
                    "relative_radius": radius,
                    "axis": axis,
                    "variant_count": len(_variants(radius)[axis]),
                    "audit_samples": len(audit_indices),
                    "corner_trace_identity_fraction": values["trace_stable"]
                    / len(audit_indices),
                    "corner_prediction_identity_fraction": values[
                        "prediction_stable"
                    ]
                    / len(audit_indices),
                    "prediction_identity_with_trace_change_fraction": values[
                        "prediction_stable_trace_changed"
                    ]
                    / len(audit_indices),
                    "mean_per_input_max_spike_hamming": values[
                        "max_spike_hamming_sum"
                    ]
                    / len(audit_indices),
                }
            )
    report = {
        "schema_version": "SHDBranchStabilityDiagnostic/v1",
        "status": (
            "endpoint diagnostic only; trace identity at corners is necessary but not "
            "sufficient for a sound fixed-branch box certificate"
        ),
        "seed": args.seed,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "audit_samples": len(audit_indices),
        "radii": list(RADII),
        "reference_semantics_hash": reference.semantics_hash,
        "rows": rows,
        "device": device,
        "torch_version": torch.__version__,
        "code_revision": code_revision(root),
        "interpretation": (
            "Any corner trace change rules out a certificate that assumes the center spike "
            "trace throughout the box. Corner trace identity does not prove interior identity."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
