from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pines.artifacts import sha256_file, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default=(
            "artifacts/shd_v83_cartesian_margin_quantile_selection_v1/"
            "cartesian_margin_quantile_selection.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/shd_v83_cartesian_margin_quantile_selection_v1/"
            "selection_verification.json"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    selection_path = root / args.selection
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "SHDCartesianMarginQuantileSelection/v1":
        raise ValueError("unsupported margin-quantile selection artifact")
    if selection.get("uses_labels") is not False:
        raise ValueError("selection is not declared label-free")
    if selection.get("split") != "repair_calibration":
        raise ValueError("selection is not confined to repair calibration")

    source_path = root / selection["source_report"]
    if sha256_file(source_path) != selection["source_report_hash"]:
        raise ValueError("source selection hash mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != "SHDCartesianDevelopmentSelectionResult/v1":
        raise ValueError("unsupported canonical selection source")
    if int(source["seed"]) != int(selection["seed"]):
        raise ValueError("seed mismatch")
    if source["split"] != selection["split"]:
        raise ValueError("split mismatch")
    if source.get("config", {}).get("selection", {}).get("uses_labels"):
        raise ValueError("canonical source used labels")

    selector_path = root / "experiments" / "select_shd_cartesian_margin_quantiles.py"
    if sha256_file(selector_path) != selection["selector_source_hash"]:
        raise ValueError("selector source hash mismatch")

    stable_rows = sorted(
        (row for row in source["rows"] if row["full_family_grid_identity"]),
        key=lambda row: (
            float(row["minimum_reference_margin"]),
            int(row["dataset_index"]),
        ),
    )
    if len(stable_rows) != int(selection["source_stable_input_count"]):
        raise ValueError("stable-input count mismatch")

    expected_rows: list[dict[str, object]] = []
    for quantile in selection["quantiles"]:
        quantile = float(quantile)
        rank = int(math.floor(quantile * (len(stable_rows) - 1) + 0.5))
        row = stable_rows[rank]
        field = f"q{int(round(100.0 * quantile)):02d}_input"
        expected = {
            "quantile": quantile,
            "rank_zero_based": rank,
            "stable_input_count": len(stable_rows),
            "dataset_index": int(row["dataset_index"]),
            "development_position": int(row["development_position"]),
            "minimum_reference_margin": float(row["minimum_reference_margin"]),
            "reference_prediction": int(row["reference_prediction"]),
            "full_family_grid_identity": True,
        }
        if selection.get(field) != expected:
            raise ValueError(f"selection mismatch for {field}")
        expected_rows.append({"field": field, **expected})
    if selection["selections"] != expected_rows:
        raise ValueError("selection row list mismatch")
    if len({row["dataset_index"] for row in expected_rows}) != len(expected_rows):
        raise ValueError("quantiles selected duplicate inputs")

    report = {
        "schema_version": "SHDCartesianMarginQuantileSelectionVerification/v1",
        "status": "passed",
        "selection_report": args.selection,
        "selection_report_hash": sha256_file(selection_path),
        "source_report": selection["source_report"],
        "source_report_hash": selection["source_report_hash"],
        "selector_source_hash": selection["selector_source_hash"],
        "stable_input_count": len(stable_rows),
        "verified_selections": expected_rows,
        "checks": {
            "label_free": True,
            "repair_calibration_split": True,
            "source_hash": True,
            "selector_source_hash": True,
            "quantile_ranks": True,
            "row_values": True,
            "unique_dataset_indices": True,
        },
    }
    output_path = root / args.output
    write_json_immutable(output_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
