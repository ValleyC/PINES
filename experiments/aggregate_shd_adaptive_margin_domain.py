from __future__ import annotations

import json
from pathlib import Path

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results" / "shd_v1" / "adaptive_margin_domain_summary.json"
    if output_path.exists():
        raise FileExistsError("adaptive-margin summary already exists")

    relative_paths = (
        "artifacts/shd_v25_adaptive_margin/seed_1701/"
        "reset_to_value_adaptive_margin.json",
        "artifacts/shd_v26_adaptive_margin_fine/seed_1701/"
        "reset_to_value_adaptive_margin.json",
    )
    reports = [_read(root / relative) for relative in relative_paths]
    rows = []
    for report in reports:
        rows.extend(report["rows"])
    summary = {
        "schema_version": "SHDAdaptiveMarginDomainSummary/v1",
        "status": (
            "sound single-seed grid-stable-input method-development kill test; finite "
            "grid selection is a tractability diagnostic, not a certificate"
        ),
        "seed": 1701,
        "condition": "reset_to_value",
        "family_radius": 0.01,
        "selection_pool": int(reports[0]["selection_pool"]),
        "selection_grid_resolution": int(
            reports[0]["selection_grid_resolution"]
        ),
        "grid_stable_pool_inputs": int(reports[0]["grid_stable_pool_inputs"]),
        "selected_indices": reports[0]["selected_indices"],
        "rows": sorted(
            rows,
            key=lambda row: (int(row["max_leaves"]), int(row["dataset_index"])),
        ),
        "route_assessment": {
            "maximum_leaf_budget": max(int(row["max_leaves"]) for row in rows),
            "maximum_analyzed_boxes": max(
                int(row["analyzed_boxes"]) for row in rows
            ),
            "any_retired_certified_leaf": any(
                int(row["certified_leaves"]) > 0 for row in rows
            ),
            "any_positive_certified_parameter_fraction": any(
                float(row["certified_parameter_fraction"]) > 0 for row in rows
            ),
            "any_full_input_certificate": any(bool(row["certified"]) for row in rows),
            "advance_to_more_inputs_or_five_seeds": False,
        },
        "interpretation": (
            "Eighty of 128 frozen audit-pool inputs are prediction-stable on the finite "
            "nine-by-nine grid. For the first two such inputs, adaptive decision-margin "
            "branch-and-bound retires no sub-box through 256 leaves; for the first input "
            "it still retires zero volume after 8,191 analyzed boxes and 4,096 leaves. "
            "Adaptive scheduling cannot rescue the non-relational recurrent state domain. "
            "The next sound method must represent correlated hidden states and guards."
        ),
        "input_report_hashes": {
            relative: sha256_file(root / relative) for relative in relative_paths
        },
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, summary)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
