from __future__ import annotations

import argparse
import json
from pathlib import Path

from pines.benchmarks.shd import PackedSHD
from pines.benchmarks.shd_repair import SHDRepairConfig, run_shd_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1701, 2718, 3141, 5772, 8119]
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["reset_to_value", "floor_rounding_saturation"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "certificate_directed",
            "logit_only",
            "global_threshold",
            "per_platform_qat",
            "supervised_target_retraining",
        ],
    )
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--semantic-root", default="artifacts/shd_v74_semantics_cast_faithful_v1"
    )
    parser.add_argument(
        "--output-root", default="artifacts/shd_v75_repairs_cast_faithful_v1"
    )
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_store = PackedSHD(root / args.data_root / "train.npz")
    test_store = PackedSHD(root / args.data_root / "test.npz")
    config = SHDRepairConfig(epochs=args.epochs)
    total = len(args.seeds) * len(args.conditions) * len(args.methods)
    completed = 0
    for seed in args.seeds:
        seed_dir = root / args.artifact_root / f"seed_{seed}"
        semantic_dir = root / args.semantic_root / f"seed_{seed}"
        for condition in args.conditions:
            for method in args.methods:
                output = (
                    root
                    / args.output_root
                    / f"seed_{seed}"
                    / condition
                    / method
                )
                report_path = output / "repair_report.json"
                if report_path.exists():
                    with report_path.open("r", encoding="utf-8") as handle:
                        existing = json.load(handle)
                    if (
                        int(existing.get("seed", seed)) != seed
                        or existing.get("condition") != condition
                        or existing.get("method") != method
                    ):
                        raise ValueError(f"existing repair cell mismatch: {report_path}")
                    status = "resumed"
                else:
                    report = run_shd_repair(
                        seed_dir / "model.npz",
                        train_store,
                        test_store,
                        seed_dir / "split_indices.npz",
                        semantic_dir / "semantic_predictions.npz",
                        condition,
                        method,
                        output,
                        root,
                        config,
                        seed=seed,
                    )
                    status = (
                        f"recovery={report['accuracy_recovery_fraction']:.4f} "
                        f"post_bound={report['after']['certificate_upper_bound']:.4f}"
                    )
                completed += 1
                print(
                    f"repair cell {completed}/{total} seed={seed} "
                    f"condition={condition} method={method} {status}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
