"""Export post-capture ground truth and source predictions for FPGA canaries."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1701, 2718, 3141, 5772, 8119)
BUNDLES = {
    "shd": "shd_virtex7_canary_v2",
    "dvs": "dvs_floor_q8q16_v1",
}


def export_task(task: str, output: Path) -> None:
    bundle = ROOT / "hardware/bundles" / BUNDLES[task]
    dataset = "shd_v1" if task == "shd" else "dvs_gesture_v2"
    with np.load(bundle / "common/canary_inputs.npz", allow_pickle=False) as inputs:
        ids = inputs["sample_ids"].astype(str)
        indices = inputs["dataset_indices"] if task == "dvs" else None
    with np.load(ROOT / "data/processed" / dataset / "test.npz", allow_pickle=False) as data:
        if task == "shd":
            positions = {sid: i for i, sid in enumerate(data["sample_ids"].astype(str))}
            indices = np.asarray([positions[sid] for sid in ids], dtype=np.int64)
        else:
            np.testing.assert_array_equal(ids, [f"dvs-test-{i:05d}" for i in indices])
        labels = data["labels"][indices].astype(np.int64)

    arrays = dict(sample_ids=ids, sample_indices=np.arange(len(ids)),
                  dataset_indices=indices, labels=labels)
    for seed in SEEDS:
        with np.load(bundle / "seeds" / str(seed) / "golden_canary.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"].astype(str), ids)
            arrays[f"source_prediction_seed{seed}"] = data["reference_predictions"]

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{task}_canary_labels.npz", **arrays)
    fields = ["sample_index", "sample_id", "dataset_index", "label"]
    fields += [f"source_prediction_seed{seed}" for seed in SEEDS]
    with (output / f"{task}_canary_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for i, sid in enumerate(ids):
            writer.writerow([i, sid, int(indices[i]), int(labels[i])] +
                            [int(arrays[f"source_prediction_seed{seed}"][i]) for seed in SEEDS])
    print(f"{task.upper()}: {len(ids)} aligned labels and source predictions for {len(SEEDS)} seeds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "hardware/evaluation")
    args = parser.parse_args()
    for task in BUNDLES:
        export_task(task, args.output)
