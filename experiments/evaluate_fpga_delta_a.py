"""Compute post-capture accuracy change against the original frozen source.

Requires only NumPy and the published hardware/evaluation files.
Predictions must be recording-level class indices, not DVS window predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1701, 2718, 3141, 5772, 8119)


def read_predictions(path: Path, expected_ids: np.ndarray):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            ids = data["sample_ids"].astype(str)
            predictions = data["predictions"].copy()
    else:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("No predictions in capture")
        if "window_index" in rows[0]:
            raise ValueError("Aggregate DVS window logits with analyze_dvs_fpga_capture.py first")
        prediction_key = next((key for key in ("prediction", "hardware_prediction", "device_prediction", "rtl_prediction")
                               if key in rows[0]), None)
        if prediction_key is None:
            raise ValueError("CSV needs a prediction, hardware_prediction, device_prediction or rtl_prediction column")
        if "sample_id" in rows[0]:
            ids = np.asarray([row["sample_id"] for row in rows])
        else:
            indices = np.asarray([int(row["sample_index"]) for row in rows])
            if np.any(indices < 0) or np.any(indices >= len(expected_ids)):
                raise ValueError("Sample index outside this canary package")
            ids = expected_ids[indices]
        predictions = np.asarray([int(row[prediction_key]) for row in rows])
    return ids, predictions


def calculate(expected_ids, labels, source, ids, predictions, classes):
    if len(ids) == 0 or len(set(ids)) != len(ids):
        raise ValueError("Provide one primary prediction per sample, without duplicates")
    if predictions.shape != (len(ids),) or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Predictions must be a one-dimensional integer array")
    if np.any(predictions < 0) or np.any(predictions >= classes):
        raise ValueError(f"Class predictions must be zero-based, in [0, {classes - 1}]")
    positions = {sid: i for i, sid in enumerate(expected_ids)}
    if set(ids) - positions.keys():
        raise ValueError("Sample IDs do not match this canary package")
    selected = np.asarray([positions[sid] for sid in ids])
    truth, reference = labels[selected], source[selected]
    source_correct = int(np.count_nonzero(reference == truth))
    hardware_correct = int(np.count_nonzero(predictions == truth))
    change = 100.0 * (source_correct - hardware_correct) / len(ids)
    return dict(samples=len(ids), expected_samples=len(expected_ids),
                complete_canary=len(ids) == len(expected_ids),
                source_correct=source_correct, hardware_correct=hardware_correct,
                source_accuracy_percent=100.0 * source_correct / len(ids),
                hardware_accuracy_percent=100.0 * hardware_correct / len(ids),
                delta_a_pp=change, absolute_delta_a_pp=abs(change),
                source_hardware_disagreement_percent=100.0 * float(np.mean(reference != predictions)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("shd", "dvs"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--variant", choices=("unrepaired", "repaired"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(ROOT / "hardware/evaluation" / f"{args.task}_canary_labels.npz",
                 allow_pickle=False) as data:
        expected_ids, labels = data["sample_ids"].astype(str), data["labels"]
        source = data[f"source_prediction_seed{args.seed}"]
    ids, predictions = read_predictions(args.predictions, expected_ids)
    report = dict(task=args.task, seed=args.seed, variant=args.variant,
                  evaluation="post-capture accuracy comparison",
                  **calculate(expected_ids, labels, source, ids, predictions,
                              20 if args.task == "shd" else 11))
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
