"""Compare SHD RTL simulation predictions with the frozen handoff emulator."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEEDS = (1701, 2718, 3141, 5772, 8119)


def compare(csv_path, golden, variant):
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    indices = [int(row["sample_index"]) for row in rows]
    expected = np.asarray(golden[f"{variant}_emulator_predictions"])
    if indices != list(range(len(expected))):
        raise ValueError(f"{csv_path.name}: sample indices must cover the frozen audit once")
    rtl = np.asarray([int(row["rtl_prediction"]) for row in rows])
    declared_golden = np.asarray([int(row["golden_prediction"]) for row in rows])
    return dict(samples=len(rows), rtl_emulator_disagreements=int((rtl != expected).sum()),
                supplied_golden_disagreements=int((declared_golden != expected).sum()),
                match_column_errors=sum(int(row["match"]) != int(a == b)
                                        for row, a, b in zip(rows, rtl, declared_golden)))


def summarize(results, bundle):
    rows = []
    for seed in SEEDS:
        with np.load(bundle / "seeds" / str(seed) / "golden_audit.npz") as golden:
            for variant in ("unrepaired", "repaired"):
                path = results / f"seed{seed}_{variant}_batch_results.csv"
                rows.append(dict(seed=seed, variant=variant, file=path.name,
                                 **compare(path, golden, variant)))
    return dict(evidence_type="RTL simulation", physical_board_measurement=False,
                task="SHD", target="floor-rounded signed 8-bit weights (6 fractional bits) and 16-bit states (8 fractional bits)",
                source_bundle="hardware/bundles/shd_floor_q8q16_v1",
                total_prediction_comparisons=sum(row["samples"] for row in rows),
                total_disagreements=sum(row["rtl_emulator_disagreements"] for row in rows),
                scope="Class predictions on the supplied audit inputs. No physical conformance or per-transition equivalence claim.",
                reproduction_status="Batch-result CSVs supplied. The corresponding full-model testbench and simulation project are still required for independent rerunning.",
                rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/shd_rtl"))
    parser.add_argument("--bundle", type=Path, default=Path("hardware/bundles/shd_floor_q8q16_v1"))
    args = parser.parse_args()
    report = summarize(args.results, args.bundle)
    (args.results / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
