from __future__ import annotations

import argparse
from pathlib import Path

from pines.benchmarks.dvs_gesture import PackedDVSGesture
from pines.benchmarks.dvs_repair import (
    DVSGestureRepairConfig,
    run_dvs_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", default="floor_rounding_saturation")
    parser.add_argument(
        "--method",
        choices=(
            "certificate_directed",
            "logit_only",
            "global_threshold",
            "per_platform_qat",
            "supervised_target_retraining",
        ),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--data-root", default="data/processed/dvs_gesture_v2")
    parser.add_argument("--artifact-root", default="artifacts/dvs_gesture_v3")
    parser.add_argument(
        "--semantic-root", default="artifacts/dvs_gesture_v3_semantics"
    )
    parser.add_argument(
        "--output-root", default="artifacts/dvs_gesture_v4_repairs"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = (
        root
        / args.output_root
        / f"seed_{args.seed}"
        / args.condition
        / args.method
    )
    report = run_dvs_repair(
        seed_dir / "checkpoint.pt",
        PackedDVSGesture(root / args.data_root / "train.npz"),
        PackedDVSGesture(root / args.data_root / "test.npz"),
        seed_dir / "split_indices.npz",
        root
        / args.semantic_root
        / f"seed_{args.seed}"
        / "semantic_predictions.npz",
        args.condition,
        args.method,
        output_dir,
        root,
        DVSGestureRepairConfig(epochs=args.epochs),
        seed=args.seed,
    )
    print(f"before_loss={report['before']['accuracy_loss']:.4f}")
    print(f"after_loss={report['after']['accuracy_loss']:.4f}")
    print(f"recovery={report['accuracy_recovery_fraction']:.4f}")
    print(f"post_bound={report['after']['certificate_upper_bound']:.4f}")


if __name__ == "__main__":
    main()
