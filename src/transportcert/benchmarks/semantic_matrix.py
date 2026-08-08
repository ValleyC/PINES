from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import array_hash, code_revision, sha256_file, write_json_immutable
from ..models import DenseRecurrentSNN
from ..semantics import (
    ExecutionSemantics,
    IntegrationRule,
    NumericFormat,
    OverflowMode,
    ResetRule,
    RoundingMode,
    ThresholdTiming,
    UpdateOrdering,
)
from ..statistics import bonferroni_alpha, clopper_pearson_upper
from ..torch_emulator import TorchEmulator
from .shd import PackedSHD


def primary_semantic_conditions() -> dict[str, ExecutionSemantics]:
    float32 = NumericFormat("float32")
    reference = ExecutionSemantics(state_format=float32, weight_format=float32)
    fixed_state = NumericFormat(
        "fixed", 16, 8, RoundingMode.NEAREST_EVEN, OverflowMode.SATURATE
    )
    fixed_weight = NumericFormat(
        "fixed", 8, 6, RoundingMode.NEAREST_EVEN, OverflowMode.SATURATE
    )
    floor_state = NumericFormat(
        "fixed", 16, 8, RoundingMode.FLOOR, OverflowMode.SATURATE
    )
    floor_weight = NumericFormat(
        "fixed", 8, 6, RoundingMode.FLOOR, OverflowMode.SATURATE
    )
    return {
        "reference": reference,
        "reset_to_value": replace(reference, reset_rule=ResetRule.TO_VALUE),
        "exponential_euler": replace(
            reference, integration_rule=IntegrationRule.EXPONENTIAL_EULER
        ),
        "pre_integration_threshold": replace(
            reference,
            threshold_timing=ThresholdTiming.PRE_INTEGRATION,
            update_ordering=UpdateOrdering.THRESHOLD_RESET_INTEGRATE,
        ),
        "fixed_q8_weights_q16_state": replace(
            reference, state_format=fixed_state, weight_format=fixed_weight
        ),
        "floor_rounding_saturation": replace(
            reference, state_format=floor_state, weight_format=floor_weight
        ),
        "synaptic_delay_1": replace(reference, synaptic_delay_steps=1),
        "reset_to_value__delay_1": replace(
            reference, reset_rule=ResetRule.TO_VALUE, synaptic_delay_steps=1
        ),
        "exponential__fixed": replace(
            reference,
            integration_rule=IntegrationRule.EXPONENTIAL_EULER,
            state_format=fixed_state,
            weight_format=fixed_weight,
        ),
        "pre_threshold__floor": replace(
            reference,
            threshold_timing=ThresholdTiming.PRE_INTEGRATION,
            update_ordering=UpdateOrdering.THRESHOLD_RESET_INTEGRATE,
            state_format=floor_state,
            weight_format=floor_weight,
        ),
        "reset__fixed__delay_1": replace(
            reference,
            reset_rule=ResetRule.TO_VALUE,
            state_format=fixed_state,
            weight_format=fixed_weight,
            synaptic_delay_steps=1,
        ),
    }


def _batched_execute(
    model: DenseRecurrentSNN,
    store: PackedSHD,
    indices: np.ndarray,
    semantics: ExecutionSemantics,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    emulator = TorchEmulator(device=device, dtype=torch.float64)
    predictions: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        trace = emulator.run(model, store.frames(batch_indices), semantics).numpy()
        predictions.append(trace.predictions)
        logits.append(trace.final_logits)
    return np.concatenate(predictions), np.concatenate(logits)


def run_shd_semantic_matrix(
    model_path: str | Path,
    train_store: PackedSHD,
    test_store: PackedSHD,
    split_indices_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
    *,
    confidence: float = 0.95,
    batch_size: int = 256,
    benchmark: str = "SHD",
) -> dict[str, Any]:
    import torch

    model_path = Path(model_path)
    split_indices_path = Path(split_indices_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "semantic_matrix.json"
    predictions_path = output_dir / "semantic_predictions.npz"
    if summary_path.exists() or predictions_path.exists():
        raise FileExistsError(f"semantic matrix output already exists: {output_dir}")
    with np.load(split_indices_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
        calibration_indices = np.asarray(splits["repair_calibration"], dtype=np.int64)
        test_indices = np.asarray(splits["test"], dtype=np.int64)
    model = DenseRecurrentSNN.load(model_path)
    conditions = primary_semantic_conditions()
    targets = [name for name in conditions if name != "reference"]
    alpha_each = bonferroni_alpha(1.0 - confidence, len(targets))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions: dict[str, np.ndarray] = {}
    logits: dict[str, np.ndarray] = {}
    for split_name, store, indices in (
        ("audit", train_store, audit_indices),
        ("calibration", train_store, calibration_indices),
        ("test", test_store, test_indices),
    ):
        for condition_name, semantics in conditions.items():
            key = f"{split_name}__{condition_name}"
            predictions[key], logits[key] = _batched_execute(
                model, store, indices, semantics, batch_size, device
            )
            print(
                f"executed {split_name} {condition_name} "
                f"samples={len(indices)}",
                flush=True,
            )
    reference_audit = predictions["audit__reference"]
    reference_test = predictions["test__reference"]
    reference_test_accuracy = float(
        np.mean(reference_test == test_store.labels[test_indices])
    )
    rows: list[dict[str, Any]] = []
    family_agreement = np.ones(len(audit_indices), dtype=bool)
    for name in targets:
        audit_predictions = predictions[f"audit__{name}"]
        test_predictions = predictions[f"test__{name}"]
        disagreements = int(np.count_nonzero(audit_predictions != reference_audit))
        disagreement_rate = disagreements / len(audit_indices)
        upper_bound = clopper_pearson_upper(
            disagreements, len(audit_indices), alpha_each
        )
        target_accuracy = float(
            np.mean(test_predictions == test_store.labels[test_indices])
        )
        absolute_change = abs(target_accuracy - reference_test_accuracy)
        loss = reference_test_accuracy - target_accuracy
        family_agreement &= audit_predictions == reference_audit
        rows.append(
            {
                "condition": name,
                "semantics_hash": conditions[name].semantics_hash,
                "audit_disagreements": disagreements,
                "audit_samples": len(audit_indices),
                "audit_disagreement_rate": disagreement_rate,
                "simultaneous_upper_bound": upper_bound,
                "reference_test_accuracy": reference_test_accuracy,
                "target_test_accuracy": target_accuracy,
                "accuracy_loss": loss,
                "absolute_accuracy_change": absolute_change,
                "bound_slack_vs_test_change": upper_bound - absolute_change,
                "bound_not_violated": absolute_change <= upper_bound,
                "budget_verdicts": {
                    str(budget): "accept" if upper_bound <= budget else "reject"
                    for budget in (0.01, 0.02, 0.05)
                },
            }
        )
    raw_payload: dict[str, Any] = {
        "audit_indices": audit_indices,
        "calibration_indices": calibration_indices,
        "test_indices": test_indices,
    }
    raw_payload.update({f"pred__{key}": value for key, value in predictions.items()})
    raw_payload.update({f"logits__{key}": value for key, value in logits.items()})
    with predictions_path.open("xb") as handle:
        np.savez_compressed(handle, **raw_payload)
    summary = {
        "schema_version": f"{benchmark}SemanticMatrix/v1",
        "benchmark": benchmark,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": train_store.data_hash,
        "test_store_hash": test_store.data_hash,
        "split_indices_hash": sha256_file(split_indices_path),
        "prediction_artifact_hash": sha256_file(predictions_path),
        "reference_semantics_hash": conditions["reference"].semantics_hash,
        "condition_semantics": {
            name: semantics.to_dict() for name, semantics in conditions.items()
        },
        "confidence": confidence,
        "simultaneous_method": "Bonferroni-corrected one-sided Clopper-Pearson",
        "per_condition_alpha": alpha_each,
        "audit_samples": len(audit_indices),
        "calibration_samples": len(calibration_indices),
        "test_samples": len(test_indices),
        "reference_test_accuracy": reference_test_accuracy,
        "family_observed_agreement_fraction": float(np.mean(family_agreement)),
        "conditions_over_five_point_loss": int(
            sum(row["accuracy_loss"] > 0.05 for row in rows)
        ),
        "simultaneous_bound_violations": int(
            sum(not row["bound_not_violated"] for row in rows)
        ),
        "rows": rows,
        "device": device,
        "torch_version": torch.__version__,
        "code_revision": code_revision(repository_root),
        "interpretation": (
            "Software-only prospective audit; conditional on emulator and not a physical certificate."
        ),
    }
    write_json_immutable(summary_path, summary)
    return summary
