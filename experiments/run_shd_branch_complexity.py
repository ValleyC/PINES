from __future__ import annotations

import argparse
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
from transportcert.branch_analysis import packed_trace_hashes, summarize_branch_grid
from transportcert.models import DenseRecurrentSNN
from transportcert.torch_emulator import TorchEmulator


GRID_RESOLUTIONS = (3, 5, 9, 17)
MAX_RESOLUTION = max(GRID_RESOLUTIONS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--radius", type=float, default=1e-2)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v4_branch_complexity")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1:
        raise ValueError("sample count must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "branch_complexity.json"
    if report_path.exists():
        raise FileExistsError(f"branch-complexity output exists: {report_path}")

    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(split_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    selected_indices = audit_indices[: min(args.sample_count, len(audit_indices))]
    frames = store.frames(selected_indices)
    reference = primary_semantic_conditions()["reference"]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = TorchEmulator(device=device, dtype=torch.float64)
    factors = np.linspace(
        1.0 - args.radius, 1.0 + args.radius, MAX_RESOLUTION, dtype=np.float64
    )
    grid_points = MAX_RESOLUTION * MAX_RESOLUTION
    trace_hashes = np.empty((grid_points, len(selected_indices)), dtype="|S32")
    predictions = np.empty((grid_points, len(selected_indices)), dtype=np.int16)

    point = 0
    for timestep_factor in factors:
        target_semantics = replace(
            reference, timestep=reference.timestep * float(timestep_factor)
        )
        for threshold_factor in factors:
            target_model = model.with_parameters(
                threshold=model.threshold * float(threshold_factor)
            )
            trace = emulator.run(target_model, frames, target_semantics).numpy()
            trace_hashes[point] = packed_trace_hashes(trace.spikes)
            predictions[point] = np.asarray(trace.predictions, dtype=np.int16)
            point += 1
        print(
            f"branch complexity seed={args.seed} rows={point // MAX_RESOLUTION}/{MAX_RESOLUTION}",
            flush=True,
        )

    rows = [
        summarize_branch_grid(
            trace_hashes,
            predictions,
            max_resolution=MAX_RESOLUTION,
            resolution=resolution,
        )
        for resolution in GRID_RESOLUTIONS
    ]
    report = {
        "schema_version": "SHDBranchComplexityDiagnostic/v1",
        "status": (
            "finite-grid diagnostic only; sampled branch counts are lower bounds on "
            "continuous-family branch counts and are not certificates"
        ),
        "seed": args.seed,
        "relative_radius": args.radius,
        "grid_resolutions": list(GRID_RESOLUTIONS),
        "maximum_grid_resolution": MAX_RESOLUTION,
        "sample_selection": "first entries of the frozen certificate-audit order",
        "sample_count": len(selected_indices),
        "selected_indices_hash": array_hash(selected_indices),
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(split_path),
        "reference_semantics_hash": reference.semantics_hash,
        "rows": rows,
        "device": device,
        "torch_version": torch.__version__,
        "code_revision": code_revision(root),
        "interpretation": (
            "Growth in distinct sampled spike traces measures the minimum branch burden "
            "faced by guarded exact propagation. Stable predictions despite many traces "
            "motivate a correlated output abstraction rather than trace enumeration."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
