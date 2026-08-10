from __future__ import annotations

import argparse
import json
from pathlib import Path

from pines.benchmarks.dvs_gesture import PackedDVSGesture
from pines.benchmarks.dvs_repair import DVSGestureRepairConfig, run_dvs_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1701, 2718, 3141, 5772, 8119]
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["floor_rounding_saturation"]
    )
    parser.add_argument(
        "--methods", nargs="+", default=["certificate_directed", "logit_only"]
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--logit-weight", type=float, default=1.0)
    parser.add_argument("--data-root", default="data/processed/dvs_gesture_v2")
    parser.add_argument("--artifact-root", default="artifacts/dvs_gesture_v3")
    parser.add_argument(
        "--semantic-root", default="artifacts/dvs_gesture_v3_semantics"
    )
    parser.add_argument(
        "--output-root", default="artifacts/dvs_gesture_repair_matrix"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_store = PackedDVSGesture(root / args.data_root / "train.npz")
    test_store = PackedDVSGesture(root / args.data_root / "test.npz")
    config = DVSGestureRepairConfig(
        epochs=args.epochs,
        margin_weight=args.margin_weight,
        logit_weight=args.logit_weight,
    )
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
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if (
                        int(report["random_seed"]) != seed
                        or report["condition"] != condition
                        or report["method"] != method
                    ):
                        raise ValueError(f"existing repair cell mismatch: {report_path}")
                    status = "resumed"
                else:
                    report = run_dvs_repair(
                        seed_dir / "checkpoint.pt",
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
                    f"DVS repair cell {completed}/{total} seed={seed} "
                    f"condition={condition} method={method} {status}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
