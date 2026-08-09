from __future__ import annotations

from experiments.verify_shd_hybrid_audit_artifact import _check_row


def _budgets() -> dict[str, int]:
    return {
        "maximum_polygon_leaves": 64,
        "maximum_guard_band_splits": 8,
        "maximum_local_branches": 64,
    }


def test_full_audit_artifact_row_accepts_exact_complete_accounting() -> None:
    row = {
        "seed": 1701,
        "audit_position": 3,
        "certified_parameter_fraction": 1.0,
        "unresolved_parameter_fraction": 0.0,
        "certified": True,
        "branch_certified_leaves": 3,
        "affine_certified_leaves": 1,
        "unresolved_leaves": 0,
        "final_leaves": 4,
        "guard_band_splits": 2,
        "maximum_completed_branches": 16,
        "branch_attempts": 7,
        "analyzed_polygons": 7,
    }
    assert _check_row(row, _budgets()) == []


def test_full_audit_artifact_row_rejects_sliver_and_leaf_mismatch() -> None:
    row = {
        "seed": 1701,
        "audit_position": 3,
        "certified_parameter_fraction": 0.999999999999,
        "unresolved_parameter_fraction": 0.0,
        "certified": True,
        "branch_certified_leaves": 3,
        "affine_certified_leaves": 1,
        "unresolved_leaves": 0,
        "final_leaves": 3,
        "guard_band_splits": 2,
        "maximum_completed_branches": 16,
        "branch_attempts": 7,
        "analyzed_polygons": 6,
    }
    failures = _check_row(row, _budgets())
    assert any("fractions do not close exactly" in failure for failure in failures)
    assert any("complete certificate does not cover full area" in failure for failure in failures)
    assert any("terminal leaf accounting mismatch" in failure for failure in failures)
    assert any("analyzed/branch-attempt count mismatch" in failure for failure in failures)
