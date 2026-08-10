from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pines.artifacts import sha256_file


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_clean_revision(value: object, path: Path) -> None:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"unclean or invalid code revision in {path}: {value!r}")


def _verify_repair_report(path: Path, expected_hash: str) -> None:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"repair-report hash mismatch: {path}")
    report = _load(path)
    _require_clean_revision(report.get("code_revision"), path)
    if not report.get("calibration_audit_disjoint"):
        raise ValueError(f"calibration/audit overlap: {path}")
    if report.get("test_labels_used_for_selection"):
        raise ValueError(f"test-label selection leakage: {path}")


def _verify_shd(summary_path: Path, repository_root: Path) -> int:
    summary = _load(summary_path)
    _require_clean_revision(summary.get("code_revision"), summary_path)
    rows_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_rows.csv"))
    if sha256_file(rows_path) != summary["rows_csv_hash"]:
        raise ValueError(f"SHD rows hash mismatch: {rows_path}")
    roots = summary["protocol"]["method_artifact_roots"]
    for key, expected_hash in summary["input_report_hashes"].items():
        condition, seed_tag, method = key.split("__")
        seed = int(seed_tag.removeprefix("seed_"))
        report_path = (
            repository_root
            / roots[method]
            / f"seed_{seed}"
            / condition
            / method
            / "repair_report.json"
        )
        _verify_repair_report(report_path, expected_hash)
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
    return len(summary["input_report_hashes"])


def _verify_dvs(summary_path: Path, repository_root: Path) -> int:
    summary = _load(summary_path)
    _require_clean_revision(summary.get("code_revision"), summary_path)
    rows_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_rows.csv"))
    if sha256_file(rows_path) != summary["rows_csv_hash"]:
        raise ValueError(f"DVS rows hash mismatch: {rows_path}")
    roots = summary["protocol"]["method_artifact_roots"]
    for key, expected_hash in summary["input_report_hashes"].items():
        seed_tag, method = key.split("__")
        seed = int(seed_tag.removeprefix("seed_"))
        report_path = (
            repository_root
            / roots[method]
            / f"seed_{seed}"
            / "floor_rounding_saturation"
            / method
            / "repair_report.json"
        )
        _verify_repair_report(report_path, expected_hash)
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
    return len(summary["input_report_hashes"])


def _verify_family_grid(summary_path: Path, repository_root: Path) -> int:
    summary = _load(summary_path)
    _require_clean_revision(summary.get("code_revision"), summary_path)
    for key, expected_hash in summary["input_report_hashes"].items():
        condition, seed_tag, alias = key.split("__")
        seed = int(seed_tag.removeprefix("seed_"))
        report_path = (
            repository_root
            / summary["methods"][alias]["path"]
            / f"seed_{seed}"
            / f"{condition}_family_grid.json"
        )
        if sha256_file(report_path) != expected_hash:
            raise ValueError(f"family-grid hash mismatch: {report_path}")
        _require_clean_revision(_load(report_path).get("code_revision"), report_path)
    return len(summary["input_report_hashes"])


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
        "shd_repair_reports": _verify_shd(
            repository_root / args.shd_summary, repository_root
        ),
        "dvs_repair_reports": _verify_dvs(
            repository_root / args.dvs_summary, repository_root
        ),
        "family_grid_reports": _verify_family_grid(
            repository_root / args.family_summary, repository_root
        ),
    }
    print(json.dumps({"status": "PASS", **counts}, indent=2))


if __name__ == "__main__":
    main()
