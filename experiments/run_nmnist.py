from __future__ import annotations

import argparse
from pathlib import Path

from pines.benchmarks.nmnist import (
    PackedNMNIST,
    load_nmnist_config,
    preprocess_nmnist,
    train_nmnist_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/nmnist_v1.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="artifacts/nmnist_v1")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    preprocess, training, raw = load_nmnist_config(root / args.config)
    data_root = root / args.data_root
    processed = data_root / "processed" / "nmnist_v1"
    train_path = preprocess_nmnist(
        data_root / "raw", processed / "train.npz", "train", preprocess
    )
    test_path = preprocess_nmnist(
        data_root / "raw", processed / "test.npz", "test", preprocess
    )
    if args.preprocess_only:
        print(train_path)
        print(test_path)
        return
    train_store = PackedNMNIST(train_path)
    test_store = PackedNMNIST(test_path)
    for seed in args.seeds or raw["seeds"]:
        manifest = train_nmnist_seed(
            train_store,
            test_store,
            root / args.output_root / f"seed_{seed}",
            seed,
            training,
            raw["partition_salt"],
            root,
        )
        print(
            f"completed N-MNIST seed={seed} "
            f"test_accuracy={manifest['final_accuracy']['test']:.4f}"
        )


if __name__ == "__main__":
    main()

