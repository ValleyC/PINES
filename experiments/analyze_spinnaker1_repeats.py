"""Describe repeated physical predictions without enlarging the primary audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from analyze_spinnaker1_matrix import collect as collect_shd
from analyze_spinnaker1_dvs_matrix import collect as collect_dvs


def summarize(groups):
    rows = []
    for (seed, variant, sample), predictions in sorted(groups.items()):
        if len(predictions) < 2:
            continue
        changes = sum(prediction != predictions[0] for prediction in predictions[1:])
        rows.append(dict(seed=seed, variant=variant, sample_id=sample,
            predictions=predictions, executions=len(predictions), changes_from_first=changes,
            any_prediction_change=changes > 0))
    comparisons = sum(row["executions"]-1 for row in rows)
    changes = sum(row["changes_from_first"] for row in rows)
    return dict(status="repeated_execution_diagnostic", physical_certificate=False,
        repeated_input_conditions=len(rows), comparisons_to_first_execution=comparisons,
        changed_predictions=changes,
        change_rate_from_first=changes/comparisons if comparisons else None,
        fraction_conditions_with_any_change=sum(row["any_prediction_change"] for row in rows)/len(rows) if rows else None,
        assumptions=["Each grouped condition uses the same checkpoint, input and execution configuration.",
                     "These repeats characterize variability and are not added to the primary binomial sample count."],
        rows=rows)


def analyze(captures, task):
    groups = defaultdict(list)
    collect = collect_shd if task == "shd" else collect_dvs
    for capture in captures:
        observations, _ = collect([capture])
        for key, value in observations.items():
            groups[key].append(int(value["prediction"] if task == "shd" else value))
    return dict(task=task, **summarize(groups))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["shd", "dvs"], required=True)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.captures, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
