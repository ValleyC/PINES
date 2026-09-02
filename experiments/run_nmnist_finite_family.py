from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from pines.artifacts import (
    array_description,
    code_revision,
    file_reference,
    write_json,
)
from pines.benchmarks.nmnist import PackedNMNIST
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.models import DenseRecurrentSNN
from pines.semantics import (
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)
from pines.torch_emulator import TorchEmulator


def _family():
    reference = primary_semantic_conditions()["reference"]
    members = []
    for integration, timing, reset, delay in itertools.product(
        (IntegrationRule.FORWARD_EULER, IntegrationRule.EXPONENTIAL_EULER),
        (ThresholdTiming.POST_INTEGRATION, ThresholdTiming.PRE_INTEGRATION),
        (ResetRule.SUBTRACTIVE, ResetRule.TO_VALUE),
        (0, 1),
    ):
        ordering = (
            UpdateOrdering.THRESHOLD_RESET_INTEGRATE
            if timing is ThresholdTiming.PRE_INTEGRATION
            else UpdateOrdering.INTEGRATE_THRESHOLD_RESET
        )
        members.append(
            replace(
                reference,
                integration_rule=integration,
                threshold_timing=timing,
                update_ordering=ordering,
                reset_rule=reset,
                synaptic_delay_steps=delay,
            )
        )
    return reference, tuple(members)


def _execute(emulator, model, store, indices, semantics, batch_size):
    predictions = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        trace = emulator.run(model, store.frames(batch_indices), semantics).numpy()
        predictions.append(trace.predictions)
    return np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data/processed/nmnist_v1")
    parser.add_argument("--artifact-root", default="artifacts/nmnist_v1")
    parser.add_argument(
        "--output-root", default="artifacts/nmnist_v3_full_audit_finite_family"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    import torch

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "finite_family_report.json"
    prediction_path = output_dir / "finite_family_predictions.npz"
    if report_path.exists() or prediction_path.exists():
        raise FileExistsError(f"finite-family output exists: {output_dir}")

    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    store = PackedNMNIST(root / args.data_root / "train.npz")
    model = DenseRecurrentSNN.load(model_path)
    if np.count_nonzero(model.recurrent_weights) != 0:
        raise ValueError("N-MNIST negative control must be feedforward")
    with np.load(split_path, allow_pickle=False) as split_data:
        audit_indices = np.asarray(
            split_data["certificate_audit"], dtype=np.int64
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = TorchEmulator(device=device, dtype=torch.float32)
    reference, members = _family()
    started = time.perf_counter()
    reference_predictions = _execute(
        emulator, model, store, audit_indices, reference, args.batch_size
    )
    family_agreement = np.ones(len(audit_indices), dtype=bool)
    member_predictions = []
    for member_index, member in enumerate(members):
        predictions = _execute(
            emulator, model, store, audit_indices, member, args.batch_size
        )
        member_predictions.append(predictions)
        family_agreement &= predictions == reference_predictions
        print(
            f"N-MNIST finite family seed={args.seed} member={member_index + 1}/16 "
            f"agreement={np.mean(predictions == reference_predictions):.4f}",
            flush=True,
        )
    elapsed = time.perf_counter() - started

    with prediction_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            audit_indices=audit_indices,
            reference_predictions=reference_predictions,
            member_predictions=np.stack(member_predictions),
            family_agreement=family_agreement,
        )
    report = {
        "schema_version": "NMNISTFiniteFamily/v1",
        "status": (
            "exact 16-member finite-family emulator certificate on the complete "
            "frozen unlabeled audit split; not physical evidence"
        ),
        "seed": args.seed,
        "member_count": len(members),
        "member_semantics": [member.to_dict() for member in members],
        "member_semantics_descriptions": [member.semantics_description for member in members],
        "reference_semantics": reference.semantics_description,
        "audit_samples": len(audit_indices),
        "audit_indices_reference": array_description(audit_indices),
        "certified_inputs": int(np.count_nonzero(family_agreement)),
        "certified_fraction": float(np.mean(family_agreement)),
        "falsified_inputs": int(np.count_nonzero(~family_agreement)),
        "unknown_inputs": 0,
        "elapsed_seconds": elapsed,
        "model_description": model.model_description,
        "model_file": file_reference(model_path),
        "split_indices_file": file_reference(split_path),
        "train_store_reference": store.data_description,
        "prediction_file": file_reference(prediction_path),
        "device": device,
        "torch_version": torch.__version__,
        "code_revision": code_revision(root),
    }
    write_json(report_path, report)
    print(json.dumps({
        "seed": args.seed,
        "certified_fraction": report["certified_fraction"],
        "elapsed_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
