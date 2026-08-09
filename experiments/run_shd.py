from __future__ import annotations

import argparse
from pathlib import Path

from pines.benchmarks.shd import (
    PackedSHD,
    load_experiment_config,
    preprocess_shd,
    train_shd_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/shd_v1.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="artifacts/shd_v1")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    preprocess, training, raw = load_experiment_config(root / args.config)
    data_root = root / args.data_root
    processed = data_root / "processed" / "shd_v1"
    train_path = preprocess_shd(
        data_root / "raw" / "SHD" / "shd_train.h5",
        processed / "train.npz",
        "train",
        preprocess,
    )
    test_path = preprocess_shd(
        data_root / "raw" / "SHD" / "shd_test.h5",
        processed / "test.npz",
        "test",
        preprocess,
    )
    if args.preprocess_only:
        print(train_path)
        print(test_path)
        return
    train_store = PackedSHD(train_path)
    test_store = PackedSHD(test_path)
    seeds = args.seeds or raw["seeds"]
    for seed in seeds:
        manifest = train_shd_seed(
            train_store,
            test_store,
            root / args.output_root / f"seed_{seed}",
            seed,
            training,
            raw["partition_salt"],
            root,
        )
        print(
            f"completed seed={seed} test_accuracy="
            f"{manifest['final_accuracy']['sequestered_test']:.4f}"
        )


if __name__ == "__main__":
    main()

