from __future__ import annotations

import argparse
from pathlib import Path

from transportcert.benchmarks.semantic_matrix import run_shd_semantic_matrix
from transportcert.benchmarks.shd import PackedSHD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1")
    parser.add_argument("--output-root", default="artifacts/shd_v1_semantics")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    summary = run_shd_semantic_matrix(
        seed_dir / "model.npz",
        PackedSHD(root / args.data_root / "train.npz"),
        PackedSHD(root / args.data_root / "test.npz"),
        seed_dir / "split_indices.npz",
        root / args.output_root / f"seed_{args.seed}",
        root,
        batch_size=args.batch_size,
    )
    print(f"reference_test_accuracy={summary['reference_test_accuracy']:.4f}")
    print(f"conditions_over_five_point_loss={summary['conditions_over_five_point_loss']}")
    print(f"simultaneous_bound_violations={summary['simultaneous_bound_violations']}")


if __name__ == "__main__":
    main()

