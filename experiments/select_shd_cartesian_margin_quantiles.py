from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pines.artifacts import (
    code_revision,
    sha256_file,
    write_json_immutable,
)


def _quantile_field(quantile: float) -> str:
    percent = int(round(100.0 * quantile))
    return f"q{percent:02d}_input"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=(
            "artifacts/shd_v71_cartesian_development_selection_canonical_v1/"
            "cartesian_development_selection.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/shd_v83_cartesian_margin_quantile_selection_v1",
    )
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=(0.10, 0.50, 0.90)
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source_path = root / args.source
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != "SHDCartesianDevelopmentSelectionResult/v1":
        raise ValueError("unsupported source selection artifact")
    if source.get("split") != "repair_calibration":
        raise ValueError("margin-quantile selection must remain on repair calibration")
    if source.get("config", {}).get("selection", {}).get("uses_labels"):
        raise ValueError("source selection must be label-free")

    quantiles = tuple(float(value) for value in args.quantiles)
    if not quantiles or len(set(quantiles)) != len(quantiles):
        raise ValueError("quantiles must be nonempty and unique")
    if any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("quantiles must lie in [0, 1]")

    stable_rows = sorted(
        (row for row in source["rows"] if row["full_family_grid_identity"]),
        key=lambda row: (
            float(row["minimum_reference_margin"]),
            int(row["dataset_index"]),
        ),
    )
    if not stable_rows:
        raise ValueError("source selection contains no sampled-stable inputs")

    selections: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": "SHDCartesianMarginQuantileSelection/v1",
        "status": (
            "label-free development-only margin-stratified selection; not a "
            "population sample or certificate"
        ),
        "seed": int(source["seed"]),
        "split": source["split"],
        "uses_labels": False,
        "source_report": args.source,
        "source_report_hash": sha256_file(source_path),
        "source_config_hash": source["config_hash"],
        "source_executor": source["executor"],
        "source_stable_input_count": len(stable_rows),
        "selection_rule": (
            "Sort all full-family sampled-stable repair-calibration inputs by "
            "minimum reference-class margin, break ties by dataset index, and "
            "select nearest ranks floor(q*(n-1)+0.5)."
        ),
        "quantiles": list(quantiles),
        "code_revision": code_revision(root),
        "selector_source_hash": sha256_file(Path(__file__).resolve()),
        "selections": selections,
    }
    for quantile in quantiles:
        rank = int(math.floor(quantile * (len(stable_rows) - 1) + 0.5))
        source_row = stable_rows[rank]
        selected = {
            "quantile": quantile,
            "rank_zero_based": rank,
            "stable_input_count": len(stable_rows),
            "dataset_index": int(source_row["dataset_index"]),
            "development_position": int(source_row["development_position"]),
            "minimum_reference_margin": float(
                source_row["minimum_reference_margin"]
            ),
            "reference_prediction": int(source_row["reference_prediction"]),
            "full_family_grid_identity": True,
        }
        field = _quantile_field(quantile)
        if field in report:
            raise ValueError(f"duplicate quantile field {field}")
        report[field] = selected
        selections.append({"field": field, **selected})

    output_path = root / args.output_root / "cartesian_margin_quantile_selection.json"
    write_json_immutable(output_path, report)
    print(json.dumps({"output": str(output_path), "selections": selections}, indent=2))


if __name__ == "__main__":
    main()
