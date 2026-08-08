from __future__ import annotations

import argparse
import itertools
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
from transportcert.semantics import (
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)
from transportcert.torch_emulator import TorchEmulator


GRID_RESOLUTION = 9
SUBFAMILIES = {
    "reference_member": (0,),
    "reset": (0, 2),
    "integration": (0, 8),
    "timing": (0, 4),
    "delay": (0, 1),
    "reset_delay": (0, 1, 2, 3),
    "integration_timing": (0, 4, 8, 12),
    "full": tuple(range(16)),
}


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


def _identity_fraction(
    predictions: np.ndarray,
    reference_predictions: np.ndarray,
    member_indices: tuple[int, ...],
    point_indices: np.ndarray,
) -> tuple[int, float]:
    selected = predictions[np.asarray(member_indices)][:, point_indices]
    stable = np.all(selected == reference_predictions[None, None, :], axis=(0, 1))
    return int(np.count_nonzero(stable)), float(np.mean(stable))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--radius", type=float, default=1e-2)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v11_subfamily_grid")
    args = parser.parse_args()
    if not (0 < args.radius < 1):
        raise ValueError("radius must be between zero and one")
    if args.sample_count < 1:
        raise ValueError("sample count must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "subfamily_grid.json"
    if report_path.exists():
        raise FileExistsError(f"subfamily grid output exists: {report_path}")

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
        1.0 - args.radius, 1.0 + args.radius, GRID_RESOLUTION, dtype=np.float64
    )
    predictions = np.empty(
        (len(members), GRID_RESOLUTION * GRID_RESOLUTION, len(selected_indices)),
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
                predictions[member_index, point] = np.asarray(
                    emulator.run(target_model, frames, target_semantics)
                    .numpy()
                    .predictions,
                    dtype=np.int16,
                )
                point += 1
        print(
            f"subfamily grid seed={args.seed} members={member_index + 1}/{len(members)}",
            flush=True,
        )

    center = GRID_RESOLUTION // 2
    center_index = center * GRID_RESOLUTION + center
    full_grid = np.arange(GRID_RESOLUTION * GRID_RESOLUTION, dtype=np.int64)
    rows = []
    for name, member_indices in SUBFAMILIES.items():
        center_inputs, center_fraction = _identity_fraction(
            predictions,
            reference_predictions,
            member_indices,
            np.asarray([center_index]),
        )
        grid_inputs, grid_fraction = _identity_fraction(
            predictions,
            reference_predictions,
            member_indices,
            full_grid,
        )
        rows.append(
            {
                "subfamily": name,
                "member_indices": list(member_indices),
                "member_count": len(member_indices),
                "center_only_prediction_identity_inputs": center_inputs,
                "center_only_prediction_identity_fraction": center_fraction,
                "grid_prediction_identity_inputs": grid_inputs,
                "grid_prediction_identity_fraction": grid_fraction,
                "continuous_grid_identity_cost": center_fraction - grid_fraction,
            }
        )

    report = {
        "schema_version": "SHDSubfamilyGridDiagnostic/v1",
        "status": (
            "finite-grid subfamily falsification diagnostic; grid identity upper-bounds "
            "continuous-family unchanged-argmax certificate coverage"
        ),
        "seed": args.seed,
        "relative_radius": args.radius,
        "grid_resolution": GRID_RESOLUTION,
        "grid_points_per_member": GRID_RESOLUTION * GRID_RESOLUTION,
        "sample_count": len(selected_indices),
        "sample_selection": "first entries of the frozen certificate-audit order",
        "selected_indices_hash": array_hash(selected_indices),
        "member_semantics_hashes": [member.semantics_hash for member in members],
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
            "Comparing the center-only finite subfamily with the full grid isolates the "
            "additional empirical identity loss caused by continuous timestep and threshold "
            "variation on the same inputs and discrete members."
        ),
    }
    write_json_immutable(report_path, report)


if __name__ == "__main__":
    main()
