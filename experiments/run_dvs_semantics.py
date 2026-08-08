from __future__ import annotations

import argparse
from pathlib import Path

from transportcert.benchmarks.dvs_gesture import (
    PackedDVSGesture,
    run_dvs_semantic_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data/processed/dvs_gesture_v1")
    parser.add_argument("--artifact-root", default="artifacts/dvs_gesture_v1")
    parser.add_argument(
        "--output-root", default="artifacts/dvs_gesture_v1_semantics"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    report = run_dvs_semantic_matrix(
        seed_dir / "checkpoint.pt",
        PackedDVSGesture(root / args.data_root / "train.npz"),
        PackedDVSGesture(root / args.data_root / "test.npz"),
        seed_dir / "split_indices.npz",
        root / args.output_root / f"seed_{args.seed}",
        root,
    )
    print(
        f"completed DVS semantics seed={args.seed} "
        f"reference={report['reference_test_accuracy']:.4f} "
        f"conditions_over_5pp={report['conditions_over_five_point_loss']}"
    )


if __name__ == "__main__":
    main()
