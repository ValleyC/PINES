from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pines.artifacts import code_revision, file_reference, write_json
from pines.benchmarks.semantic_matrix import primary_semantic_conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--benchmark", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result_root = root / args.result_root
    source_summary_path = result_root / "software_matrix_summary.json"
    source_rows_path = result_root / "software_matrix_rows.csv"
    corrected_summary_path = result_root / "software_matrix_summary_float32.json"
    corrected_rows_path = result_root / "software_matrix_rows_float32.csv"
    if corrected_summary_path.exists() or corrected_rows_path.exists():
        raise FileExistsError("corrected aggregate already exists")

    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    conditions = primary_semantic_conditions()
    with source_rows_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("source rows have no header")
        rows = list(reader)
    old_references: dict[str, set[str]] = {}
    for row in rows:
        condition = row["condition"]
        if condition not in conditions or condition == "reference":
            raise ValueError(f"unexpected condition: {condition}")
        old_references.setdefault(condition, set()).add(row["semantics_description"])
        row["semantics_description"] = conditions[condition].semantics_description
    with corrected_rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    seeds = tuple(int(seed) for seed in source_summary["seeds"])
    raw_references: dict[str, str] = {}
    prediction_references: dict[str, str] = {}
    source_revisions: dict[str, str] = {}
    for seed in seeds:
        semantic_path = (
            root / args.semantic_root / f"seed_{seed}" / "semantic_matrix.json"
        )
        raw = json.loads(semantic_path.read_text(encoding="utf-8"))
        reference = raw["condition_semantics"]["reference"]
        if reference["state_format"]["kind"] != "float64":
            raise ValueError("source report is not the known float64 metadata case")
        raw_references[f"seed_{seed}"] = file_reference(semantic_path)
        prediction_references[f"seed_{seed}"] = raw["prediction_file"]
        source_revisions[f"seed_{seed}"] = raw["code_revision"]

    corrected = dict(source_summary)
    corrected["schema_version"] = f"{source_summary['schema_version']}-float32-metadata-v2"
    corrected["status"] = (
        f"{source_summary['status']}; float32 semantics metadata corrected, metrics unchanged"
    )
    corrected["runtime_dtype"] = "float32"
    corrected["reference_semantics"] = conditions["reference"].semantics_description
    corrected["condition_semantics"] = {
        name: semantics.to_dict() for name, semantics in conditions.items()
    }
    corrected["superseded_aggregate_reference"] = file_reference(source_summary_path)
    corrected["superseded_rows_reference"] = file_reference(source_rows_path)
    corrected["source_raw_semantic_summary_references"] = raw_references
    corrected["source_prediction_files"] = prediction_references
    corrected["source_code_revisions"] = source_revisions
    corrected["superseded_condition_references"] = {
        name: sorted(values) for name, values in old_references.items()
    }
    corrected["rows_csv_reference"] = file_reference(corrected_rows_path)
    corrected["code_revision"] = code_revision(root)
    corrected["metadata_correction"] = {
        "benchmark": args.benchmark,
        "numerical_results_changed": False,
        "reason": (
            "The vectorized matrix runner explicitly instantiated TorchEmulator with "
            "torch.float32, but the source ExecutionSemantics object retained its float64 "
            "default for unquantized conditions. Non-fixed quantization is an identity in "
            "that runner, so correcting the declared runtime precision changes only semantics "
            "metadata and references, not predictions, bounds, or accuracy."
        ),
    }
    write_json(corrected_summary_path, corrected)
    print(corrected_summary_path)


if __name__ == "__main__":
    main()
