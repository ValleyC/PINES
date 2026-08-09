from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np

from pines.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.branch_analysis import summarize_family_prediction_grid
from pines.models import DenseRecurrentSNN
from pines.semantics import (
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)
from pines.torch_emulator import TorchEmulator


GRID_RESOLUTIONS = (3, 5, 9)
MAX_RESOLUTION = max(GRID_RESOLUTIONS)


def _ordering(timing: ThresholdTiming) -> UpdateOrdering:
    if timing is ThresholdTiming.PRE_INTEGRATION:
        return UpdateOrdering.THRESHOLD_RESET_INTEGRATE
    return UpdateOrdering.INTEGRATE_THRESHOLD_RESET


def _members(reference):
    return tuple(
        replace(
            reference,
            integration_rule=integration,
            threshold_timing=timing,
            update_ordering=_ordering(timing),
            reset_rule=reset,
            synaptic_delay_steps=delay,
        )
        for integration, timing, reset, delay in itertools.product(
            (IntegrationRule.FORWARD_EULER, IntegrationRule.EXPONENTIAL_EULER),
            (ThresholdTiming.POST_INTEGRATION, ThresholdTiming.PRE_INTEGRATION),
            (ResetRule.SUBTRACTIVE, ResetRule.TO_VALUE),
            (0, 1),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--radius", type=float, default=1e-2)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v5_full_family_grid")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1:
        raise ValueError("sample count must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_family_grid.json"
    if report_path.exists():
        raise FileExistsError(f"full-family grid output exists: {report_path}")

    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(split_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    selected_indices = audit_indices[: min(args.sample_count, len(audit_indices))]
    frames = store.frames(selected_indices)
    reference = primary_semantic_conditions()["reference"]
    members = _members(reference)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = TorchEmulator(device=device, dtype=torch.float64)
    reference_predictions = np.asarray(
        emulator.run(model, frames, reference).numpy().predictions, dtype=np.int16
    )
    factors = np.linspace(
        1.0 - args.radius, 1.0 + args.radius, MAX_RESOLUTION, dtype=np.float64
    )
    predictions = np.empty(
        (
            len(members),
            MAX_RESOLUTION * MAX_RESOLUTION,
            len(selected_indices),
        ),
        dtype=np.int16,
    )
    for member_index, member in enumerate(members):
        point = 0
        for timestep_factor in factors:
            target_semantics = replace(
                member, timestep=reference.timestep * float(timestep_factor)
            )
            for threshold_factor in factors:
                target_model = model.with_parameters(
                    threshold=model.threshold * float(threshold_factor)
                )
                trace = emulator.run(target_model, frames, target_semantics).numpy()
                predictions[member_index, point] = np.asarray(
                    trace.predictions, dtype=np.int16
                )
                point += 1
        print(
            f"full family grid seed={args.seed} members={member_index + 1}/{len(members)}",
            flush=True,
        )

    rows = [
        summarize_family_prediction_grid(
            predictions,
            reference_predictions,
            max_resolution=MAX_RESOLUTION,
            resolution=resolution,
        )
        for resolution in GRID_RESOLUTIONS
    ]
    report = {
        "schema_version": "SHDFullFamilyGridDiagnostic/v1",
        "status": (
            "finite-grid falsification diagnostic only; sampled family identity is an "
            "upper bound on achievable certificate coverage, not a continuous proof"
        ),
        "seed": args.seed,
        "relative_radius": args.radius,
        "grid_resolutions": list(GRID_RESOLUTIONS),
        "maximum_grid_resolution": MAX_RESOLUTION,
        "member_count": len(members),
        "member_semantics_hashes": [member.semantics_hash for member in members],
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
            "Any sampled semantic setting that changes the prediction is an exact "
            "counterexample to an unchanged-argmax certificate for that input. Therefore "
            "sampled full-family identity upper-bounds possible certificate coverage."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
