from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


SUMMARY_ROWS = {
    "results/shd_v1/software_matrix_summary_float32.json":
        "results/shd_v1/software_matrix_rows_float32.csv",
    "results/dvs_gesture_v3/software_matrix_summary.json":
        "results/dvs_gesture_v3/software_matrix_rows.csv",
    "results/nmnist_v1/software_matrix_summary_float32.json":
        "results/nmnist_v1/software_matrix_rows_float32.csv",
    "results/shd_v1/static_family_memberwise_summary.json":
        "results/shd_v1/static_family_memberwise_rows.csv",
    "results/dvs_gesture_v3/finite_family_summary.json":
        "results/dvs_gesture_v3/finite_family_rows.csv",
    "results/nmnist_v1/finite_family_full_audit_summary.json":
        "results/nmnist_v1/finite_family_full_audit_rows.csv",
    "results/shd_v1/hybrid_radius_sweep_summary.json":
        "results/shd_v1/hybrid_radius_sweep_rows.csv",
    "results/shd_v1/analysis_progression_summary.json":
        "results/shd_v1/analysis_progression_rows.csv",
    "results/family_certificates/software_family_summary.json":
        "results/family_certificates/software_family_rows.csv",
    "results/shd_v1/repair_task_tuned_clean_v5_summary.json":
        "results/shd_v1/repair_task_tuned_clean_v5_rows.csv",
    "results/dvs_gesture_v3/repair_floor_task_tuned_clean_v5_summary.json":
        "results/dvs_gesture_v3/repair_floor_task_tuned_clean_v5_rows.csv",
    "results/sample_complexity/zero_disagreement_summary.json":
        "results/sample_complexity/zero_disagreement_rows.csv",
}

STANDALONE_RESULTS = {
    "results/shd_v1/hybrid_family_audit_summary.json",
    "results/shd_v1/hybrid_family_full_audit_soundness_corrected_v1_summary.json",
    "results/shd_v1/repair_task_tuned_family_grid_v5.json",
    "results/shd_v1/residual_counterexample_search_summary.json",
    "results/shd_v1/shd_hybrid_radius_sweep.pdf",
}


TABLE_I = {
    "results/shd_v1/software_matrix_summary_float32.json": {
        "exponential_euler": ((3.7, 2.0), (15.2, 2.4), (18.6, 2.6)),
        "pre_integration_threshold": ((2.3, 2.8), (17.0, 2.9), (20.5, 3.0)),
        "reset_to_value": ((9.9, 2.5), (23.8, 2.9), (27.7, 3.0)),
        "synaptic_delay_1": ((2.3, 2.8), (17.0, 2.9), (20.5, 3.0)),
        "fixed_q8_weights_q16_state": ((0.3, 0.7), (5.0, 0.6), (7.2, 0.7)),
        "floor_rounding_saturation": ((53.6, 2.5), (80.0, 4.1), (83.4, 3.8)),
        "exponential__fixed": ((4.0, 2.6), (15.6, 1.6), (19.0, 1.7)),
        "pre_threshold__floor": ((53.5, 2.5), (80.3, 3.6), (83.6, 3.3)),
        "reset_to_value__delay_1": ((16.1, 4.6), (35.7, 5.4), (40.0, 5.5)),
        "reset__fixed__delay_1": ((16.2, 4.4), (36.3, 4.9), (40.6, 5.0)),
    },
    "results/dvs_gesture_v3/software_matrix_summary.json": {
        "exponential_euler": ((3.7, 3.9), (14.0, 4.2), (24.8, 5.3)),
        "pre_integration_threshold": ((-0.2, 0.8), (0.2, 0.4), (5.4, 0.9)),
        "reset_to_value": ((1.5, 3.8), (10.6, 4.4), (20.5, 5.8)),
        "synaptic_delay_1": ((-0.2, 0.8), (0.2, 0.4), (5.4, 0.9)),
        "fixed_q8_weights_q16_state": ((-0.2, 0.7), (2.5, 2.4), (9.2, 4.1)),
        "floor_rounding_saturation": ((36.6, 10.9), (53.5, 8.2), (66.0, 7.8)),
        "exponential__fixed": ((3.7, 3.6), (12.9, 5.2), (23.3, 6.7)),
        "pre_threshold__floor": ((36.2, 11.1), (54.0, 8.7), (66.5, 8.3)),
        "reset_to_value__delay_1": ((1.8, 4.0), (10.0, 4.2), (19.8, 5.6)),
        "reset__fixed__delay_1": ((1.8, 3.7), (11.0, 4.5), (20.9, 6.1)),
    },
    "results/nmnist_v1/software_matrix_summary_float32.json": {
        "exponential_euler": ((0.1, 0.1), (1.1, 0.1), (1.5, 0.2)),
        "pre_integration_threshold": ((0.0, 0.0), (0.0, 0.0), (0.1, 0.0)),
        "reset_to_value": ((0.3, 0.1), (1.0, 0.2), (1.4, 0.2)),
        "synaptic_delay_1": ((0.0, 0.0), (0.0, 0.0), (0.1, 0.0)),
        "fixed_q8_weights_q16_state": ((0.0, 0.1), (0.6, 0.1), (0.9, 0.1)),
        "floor_rounding_saturation": ((4.3, 0.9), (6.5, 0.8), (7.3, 0.8)),
        "exponential__fixed": ((0.1, 0.1), (1.1, 0.2), (1.5, 0.2)),
        "pre_threshold__floor": ((4.3, 0.9), (6.5, 0.8), (7.3, 0.8)),
        "reset_to_value__delay_1": ((0.3, 0.1), (1.0, 0.2), (1.4, 0.2)),
        "reset__fixed__delay_1": ((0.3, 0.1), (1.1, 0.2), (1.5, 0.2)),
    },
}


def load(path: str) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point(value: float) -> float:
    return round(100.0 * float(value), 1)


def close(actual: float, expected: float, label: str, tolerance: float = 0.051) -> None:
    if abs(float(actual) - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, observed {actual}")


def verify_hashes() -> None:
    for summary_path, rows_path in SUMMARY_ROWS.items():
        summary = load(summary_path)
        observed = file_hash(rows_path)
        expected = summary.get("rows_csv_hash")
        if observed != expected:
            raise AssertionError(
                f"row hash mismatch for {rows_path}: {observed} != {expected}"
            )


def verify_result_inventory() -> None:
    expected = set(SUMMARY_ROWS) | set(SUMMARY_ROWS.values()) | STANDALONE_RESULTS
    observed = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "results").rglob("*")
        if path.is_file() and path.suffix.lower() != ".md"
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise AssertionError(f"result inventory mismatch; missing={missing}, extra={extra}")


def verify_table_i() -> None:
    for path, expected_conditions in TABLE_I.items():
        grouped: dict[str, list[dict[str, str]]] = {}
        with (ROOT / SUMMARY_ROWS[path]).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                grouped.setdefault(row["condition"], []).append(row)
        for condition, expected_triplet in expected_conditions.items():
            rows = grouped[condition]
            for field, expected_pair in zip(
                ("accuracy_loss", "audit_disagreement_rate", "simultaneous_upper_bound"),
                expected_triplet,
            ):
                values = [float(row[field]) for row in rows]
                actual = (point(statistics.mean(values)), point(statistics.stdev(values)))
                if actual != expected_pair:
                    raise AssertionError(
                        f"Table I {path} {condition} {field}: "
                        f"expected {expected_pair}, observed {actual}"
                    )


def verify_figure_2() -> None:
    rows = load("results/shd_v1/hybrid_radius_sweep_summary.json")["rows"]
    expected = [
        (0.25, 73.8, 26.2, 0.0),
        (0.50, 68.8, 28.7, 2.5),
        (1.00, 37.5, 32.5, 30.0),
        (2.00, 0.0, 42.5, 57.5),
    ]
    for row, target in zip(rows, expected):
        actual = (
            float(row["radius_percent"]),
            point(row["certified_fraction"]),
            point(row["grid_falsified_fraction"]),
            point(row["stable_unresolved_fraction"]),
        )
        if actual != target:
            raise AssertionError(f"Figure 2 mismatch: expected {target}, observed {actual}")


def verify_table_ii() -> None:
    rows = load("results/family_certificates/software_family_summary.json")["rows"]
    expected = [
        ("SHD", "finite 16", 38.7, 61.3, 0.0, 65.9),
        ("SHD", "reset continuous", 33.9, 36.8, 29.3, 70.6),
        ("DVS Gesture", "finite 16", 57.7, 42.3, 0.0, 56.2),
        ("N-MNIST", "finite 16", 97.7, 2.3, 0.0, 2.9),
    ]
    for row, target in zip(rows, expected):
        actual = (
            row["task"],
            row["family"],
            point(row["certified_fraction"]["mean"]),
            point(row["falsified_fraction"]["mean"]),
            point(row["unknown_fraction"]["mean"]),
            point(row["semantic_upper_bound"]["mean"]),
        )
        if actual != target:
            raise AssertionError(f"Table II mismatch: expected {target}, observed {actual}")


def verify_table_iii() -> None:
    rows = load("results/shd_v1/analysis_progression_summary.json")["rows"]
    expected = [
        ("shared affine boxes", False, 86.35, 99.0),
        ("polygonal guard cuts", False, 99.76, 221.2),
        ("polygonal cuts and local branch closure", True, 100.00, 324.5),
    ]
    for row, target in zip(rows, expected):
        actual = (
            row["analysis"],
            bool(row["full_certificate"]),
            round(100.0 * row["certified_parameter_fraction"], 2),
            round(row["seconds"], 1),
        )
        if actual != target:
            raise AssertionError(f"Table III mismatch: expected {target}, observed {actual}")
    if int(rows[-1]["p99_active_branches"]) != 3:
        raise AssertionError("Table III full-analysis P99 branch count is not 3")


def repair_rows(summary: dict[str, Any], condition: str | None = None) -> dict[str, Any]:
    if condition is None:
        rows = summary["per_method"]
    else:
        entry = next(row for row in summary["per_condition"] if row["condition"] == condition)
        rows = entry["per_method"]
    return {row["method"]: row for row in rows}


def verify_table_iv() -> None:
    shd = load("results/shd_v1/repair_task_tuned_clean_v5_summary.json")
    dvs = load("results/dvs_gesture_v3/repair_floor_task_tuned_clean_v5_summary.json")
    expected = {
        "reset_to_value": {
            "global_threshold": (40.1, 17.4),
            "logit_only": (70.5, 13.5),
            "per_platform_qat": (80.0, 19.2),
            "supervised_target_retraining": (-134.0, 48.6),
            "certificate_directed": (80.6, 12.7),
        },
        "floor_rounding_saturation": {
            "global_threshold": (9.4, 76.6),
            "logit_only": (89.7, 17.7),
            "per_platform_qat": (102.2, 16.5),
            "supervised_target_retraining": (57.8, 47.0),
            "certificate_directed": (91.0, 16.8),
        },
    }
    for condition, methods in expected.items():
        observed = repair_rows(shd, condition)
        for method, target in methods.items():
            row = observed[method]
            actual = (
                point(row["accuracy_recovery"]["mean"]),
                point(row["after_certificate_upper_bound"]["mean"]),
            )
            if actual != target:
                raise AssertionError(
                    f"Table IV SHD {condition} {method}: expected {target}, observed {actual}"
                )
    dvs_expected = {
        "global_threshold": (74.8, 33.6),
        "logit_only": (89.3, 15.1),
        "per_platform_qat": (99.6, 22.3),
        "supervised_target_retraining": (6.0, 59.9),
        "certificate_directed": (94.9, 17.4),
    }
    observed = repair_rows(dvs)
    for method, target in dvs_expected.items():
        row = observed[method]
        actual = (
            point(row["accuracy_recovery"]["mean"]),
            point(row["after_certificate_upper_bound"]["mean"]),
        )
        if actual != target:
            raise AssertionError(
                f"Table IV DVS {method}: expected {target}, observed {actual}"
            )


def verify_sample_complexity() -> None:
    observed = load("results/sample_complexity/zero_disagreement_summary.json")
    expected = {"0.01": 528, "0.02": 263, "0.05": 104}
    if observed["minimum_zero_disagreement_samples"] != expected:
        raise AssertionError("sample-complexity values do not match the manuscript")


def main() -> None:
    verify_result_inventory()
    verify_hashes()
    verify_table_i()
    verify_figure_2()
    verify_table_ii()
    verify_table_iii()
    verify_table_iv()
    verify_sample_complexity()
    print("Verified final PINES artifacts for Tables I-IV and Figure 2.")
    print("Table V remains pending physical-backend evidence.")


if __name__ == "__main__":
    main()
