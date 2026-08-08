from __future__ import annotations

import argparse
from pathlib import Path

from transportcert.benchmarks.shd import PackedSHD
from transportcert.benchmarks.shd_repair import SHDRepairConfig, run_shd_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--method",
        choices=(
            "certificate_directed",
            "guard_margin",
            "family_margin",
            "logit_only",
            "global_threshold",
            "per_platform_qat",
            "supervised_target_retraining",
        ),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--guard-weight", type=float, default=0.1)
    parser.add_argument("--guard-target", type=float, default=0.05)
    parser.add_argument("--family-radius", type=float, default=0.01)
    parser.add_argument("--family-grid-resolution", type=int, default=2)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1")
    parser.add_argument(
        "--semantic-root", default="artifacts/shd_v1_semantics_pilot"
    )
    parser.add_argument("--output-root", default="artifacts/shd_v1_repairs")
    args = parser.parse_args()
    if args.guard_weight < 0 or args.guard_target <= 0:
        raise ValueError("guard weight must be nonnegative and target must be positive")
    if not (0 < args.family_radius < 1):
        raise ValueError("family radius must be between zero and one")
    if args.family_grid_resolution < 2:
        raise ValueError("family grid resolution must be at least two")
    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    semantic_dir = root / args.semantic_root / f"seed_{args.seed}"
    output = (
        root
        / args.output_root
        / f"seed_{args.seed}"
        / args.condition
        / args.method
    )
    report = run_shd_repair(
        seed_dir / "model.npz",
        PackedSHD(root / args.data_root / "train.npz"),
        PackedSHD(root / args.data_root / "test.npz"),
        seed_dir / "split_indices.npz",
        semantic_dir / "semantic_predictions.npz",
        args.condition,
        args.method,
        output,
        root,
        SHDRepairConfig(
            epochs=args.epochs,
            guard_weight=args.guard_weight,
            guard_target=args.guard_target,
            family_radius=args.family_radius,
            family_grid_resolution=args.family_grid_resolution,
        ),
        seed=args.seed,
    )
    print(f"before_loss={report['before']['accuracy_loss']:.4f}")
    print(f"after_loss={report['after']['accuracy_loss']:.4f}")
    print(f"recovery={report['accuracy_recovery_fraction']:.4f}")
    print(f"post_bound={report['after']['certificate_upper_bound']:.4f}")


if __name__ == "__main__":
    main()
