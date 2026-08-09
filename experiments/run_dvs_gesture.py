from __future__ import annotations

import argparse
from pathlib import Path

from pines.benchmarks.dvs_gesture import (
    PackedDVSGesture,
    load_dvs_gesture_config,
    preprocess_dvs_gesture,
    train_dvs_gesture_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/dvs_gesture_v1.json"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="artifacts/dvs_gesture_v1")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    preprocess, training, raw = load_dvs_gesture_config(root / args.config)
    data_root = root / args.data_root
    raw_root = data_root / "raw" / "DVSGesture"
    processed = data_root / "processed" / raw.get(
        "artifact_name", "dvs_gesture_v1"
    )
    train_path = preprocess_dvs_gesture(
        raw_root, processed / "train.npz", "train", preprocess
    )
    test_path = preprocess_dvs_gesture(
        raw_root, processed / "test.npz", "test", preprocess
    )
    if args.preprocess_only:
        print(train_path)
        print(test_path)
        return
    train_store = PackedDVSGesture(train_path)
    test_store = PackedDVSGesture(test_path)
    for seed in args.seeds or raw["seeds"]:
        manifest = train_dvs_gesture_seed(
            train_store,
            test_store,
            root / args.output_root / f"seed_{seed}",
            seed,
            training,
            raw["partition_salt"],
            root,
        )
        print(
            f"completed DVS Gesture seed={seed} "
            f"test_accuracy={manifest['final_accuracy']['test']:.4f}"
        )


if __name__ == "__main__":
    main()
