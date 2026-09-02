from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_shd(summary_path: Path) -> int:
    summary = _load(summary_path)
    gates = summary["gate_assessment"]
    required_true = (
        "certificate_directed_all_ten_cells_recover_at_least_70_percent",
        "certificate_directed_beats_global_threshold_every_seed",
        "certificate_directed_beats_unrepaired_every_seed",
        "certificate_directed_beats_logit_only_in_both_conditions",
        "no_observed_bound_violation_in_all_fifty_cells",
    )
    if not all(gates[item] for item in required_true):
        raise ValueError("one or more SHD repair gates failed")
    if gates["certificate_directed_certifies_five_point_budget_any_cell"]:
        raise ValueError("SHD summary unexpectedly claims a five-point certificate")
    return len(summary["seeds"]) * len(summary["conditions"])


def _verify_dvs(summary_path: Path) -> int:
    summary = _load(summary_path)
    gates = summary["gate_assessment"]
    required_true = (
        "certificate_directed_all_seeds_recover_at_least_70_percent",
        "certificate_directed_beats_global_threshold_every_seed",
        "certificate_directed_beats_logit_only_mean_recovery",
        "no_observed_bound_violation",
    )
    if not all(gates[item] for item in required_true):
        raise ValueError("one or more DVS repair gates failed")
    if gates["certificate_directed_certifies_five_point_budget_any_seed"]:
        raise ValueError("DVS summary unexpectedly claims a five-point certificate")
    return len(summary["seeds"])


def _verify_family_grid(summary_path: Path) -> int:
    summary = _load(summary_path)
    if not summary.get("status"):
        raise ValueError("family-grid study has no status")
    expected = len(summary["conditions"])
    if len(summary["condition_results"]) != expected:
        raise ValueError("family-grid result count is incomplete")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=None)
    parser.add_argument(
        "--shd-summary",
        default="results/shd_v1/repair_task_tuned_clean_v5_summary.json",
    )
    parser.add_argument(
        "--dvs-summary",
        default=(
            "results/dvs_gesture_v3/"
            "repair_floor_task_tuned_clean_v5_summary.json"
        ),
    )
    parser.add_argument(
        "--family-summary",
        default="results/shd_v1/repair_task_tuned_family_grid_v5.json",
    )
    args = parser.parse_args()
    repository_root = (
        Path(args.repository_root).resolve()
        if args.repository_root
        else Path(__file__).resolve().parents[1]
    )
    counts = {
        "shd_conditions": _verify_shd(repository_root / args.shd_summary),
        "dvs_conditions": _verify_dvs(repository_root / args.dvs_summary),
        "family_grid_conditions": _verify_family_grid(
            repository_root / args.family_summary
        ),
    }
    print(json.dumps({"status": "PASS", **counts}, indent=2))


if __name__ == "__main__":
    main()
