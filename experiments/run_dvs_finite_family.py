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
from pines.benchmarks.dvs_gesture import (
    DVSGestureTrainConfig,
    PackedDVSGesture,
    build_dvs_conv_srnn,
    evaluate_dvs_model,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.semantics import (
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)


def _member_semantics():
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data/processed/dvs_gesture_v2")
    parser.add_argument("--artifact-root", default="artifacts/dvs_gesture_v3")
    parser.add_argument(
        "--output-root", default="artifacts/dvs_gesture_v6_finite_family"
    )
    parser.add_argument("--batch-size", type=int, default=64)
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

    checkpoint_path = seed_dir / "checkpoint.pt"
    split_path = seed_dir / "split_indices.npz"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DVSGestureTrainConfig(**checkpoint["config"])
    model = build_dvs_conv_srnn(
        int(checkpoint["sensor_width"]), int(checkpoint["sensor_height"]), config
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    store = PackedDVSGesture(root / args.data_root / "train.npz")
    with np.load(split_path, allow_pickle=False) as split_data:
        audit_indices = np.asarray(
            split_data["certificate_audit"], dtype=np.int64
        )
    reference, members = _member_semantics()

    started = time.perf_counter()
    _, reference_predictions, reference_logits = evaluate_dvs_model(
        model, store, audit_indices, args.batch_size, device, reference
    )
    family_agreement = np.ones(len(audit_indices), dtype=bool)
    member_predictions = []
    member_logits = []
    for member_index, member in enumerate(members):
        _, predictions, logits = evaluate_dvs_model(
            model, store, audit_indices, args.batch_size, device, member
        )
        member_predictions.append(predictions)
        member_logits.append(logits)
        family_agreement &= predictions == reference_predictions
        print(
            f"DVS finite family seed={args.seed} member={member_index + 1}/16 "
            f"agreement={np.mean(predictions == reference_predictions):.4f}",
            flush=True,
        )
    elapsed = time.perf_counter() - started

    prediction_payload = {
        "audit_indices": audit_indices,
        "reference_predictions": reference_predictions,
        "reference_logits": reference_logits,
        "member_predictions": np.stack(member_predictions),
        "member_logits": np.stack(member_logits),
        "family_agreement": family_agreement,
    }
    with prediction_path.open("xb") as handle:
        np.savez_compressed(handle, **prediction_payload)

    report = {
        "schema_version": "DVSGestureFiniteFamily/v1",
        "status": (
            "exact finite-family emulator certificate on the frozen unlabeled "
            "audit split; development pipeline, not physical evidence"
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
        "model_description": checkpoint["model_description"],
        "checkpoint_file": file_reference(checkpoint_path),
        "split_indices_file": file_reference(split_path),
        "train_store_reference": store.data_description,
        "prediction_file": file_reference(prediction_path),
        "device": str(device),
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
