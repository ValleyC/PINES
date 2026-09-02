from __future__ import annotations

import csv
import json
from pathlib import Path

from pines.artifacts import code_revision, file_reference, write_json
from pines.statistics import (
    bonferroni_alpha,
    clopper_pearson_upper,
    zero_error_sample_size,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "sample_complexity"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "zero_disagreement_summary.json"
    rows_path = output / "zero_disagreement_rows.csv"
    table_path = root / "paper" / "generated" / "sample_complexity_table.tex"
    if any(path.exists() for path in (summary_path, rows_path, table_path)):
        raise FileExistsError("sample-complexity output exists; evidence is saved")
    confidence = 0.95
    comparisons = 10
    alpha = bonferroni_alpha(1.0 - confidence, comparisons)
    budgets = (0.01, 0.02, 0.05)
    required = {str(budget): zero_error_sample_size(budget, alpha) for budget in budgets}
    sources = {
        "SHD": root
        / "artifacts"
        / "shd_v1_semantics_final"
        / "seed_1701"
        / "semantic_matrix.json",
        "DVS Gesture": root
        / "artifacts"
        / "dvs_gesture_v3_semantics"
        / "seed_1701"
        / "semantic_matrix.json",
        "N-MNIST": root
        / "artifacts"
        / "nmnist_v1_semantics"
        / "seed_1701"
        / "semantic_matrix.json",
    }
    rows = []
    source_references: dict[str, str] = {}
    for dataset, path in sources.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        samples = int(report["audit_samples"])
        source_references[dataset] = file_reference(path)
        row: dict[str, object] = {
            "dataset": dataset,
            "audit_samples": samples,
            "zero_disagreement_upper_bound": clopper_pearson_upper(
                0, samples, alpha
            ),
        }
        for budget in budgets:
            row[f"enough_for_{int(budget * 100)}_point_budget"] = (
                samples >= required[str(budget)]
            )
        rows.append(row)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "SampleComplexityAnalysis/v1",
        "confidence": confidence,
        "simultaneous_comparisons": comparisons,
        "per_comparison_alpha": alpha,
        "minimum_zero_disagreement_samples": required,
        "datasets": rows,
        "interpretation": (
            "DVS Gesture cannot accept one- or two-point budgets from 104 audit recordings "
            "even with zero observed disagreement, while SHD and N-MNIST exceed the zero-error "
            "sample requirement for all three budgets. SHD looseness is therefore dominated by "
            "empirical decision disagreement, not finite-sample confidence inflation."
        ),
        "physical_note": (
            "These requirements apply to the conditional reference-emulator term. A physical "
            "certificate splits confidence between semantic and conformance terms and must plan "
            "canary sizes against the allocated component budgets."
        ),
        "input_report_references": source_references,
        "rows_csv_reference": file_reference(rows_path),
        "code_revision": code_revision(root),
    }
    write_json(summary_path, summary)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrccc}",
        "\\toprule",
        "Dataset & $n$ & Zero-error bound & 1 pt & 2 pt & 5 pt \\\\",
        "\\midrule",
    ]
    for row in rows:
        marks = [
            "\\checkmark"
            if row[f"enough_for_{int(budget * 100)}_point_budget"]
            else "--"
            for budget in budgets
        ]
        lines.append(
            f"{row['dataset']} & {row['audit_samples']} & "
            f"{float(row['zero_disagreement_upper_bound']) * 100:.2f} & "
            f"{marks[0]} & {marks[1]} & {marks[2]} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(required, indent=2))


if __name__ == "__main__":
    main()
