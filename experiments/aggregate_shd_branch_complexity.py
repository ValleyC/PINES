from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable


SEEDS = (1701, 2718, 3141, 5772, 8119)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / "artifacts" / "shd_v4_branch_complexity"
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / "branch_complexity_summary.json"
    rows_path = output_root / "branch_complexity_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_branch_complexity.pdf"
    figure_png = root / "paper" / "figures" / "shd_branch_complexity.png"
    destinations = (summary_path, rows_path, figure_pdf, figure_png)
    if any(path.exists() for path in destinations):
        raise FileExistsError("branch-complexity aggregate destination already exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    radii: set[float] = set()
    sample_counts: set[int] = set()
    for seed in SEEDS:
        path = artifact_root / f"seed_{seed}" / "branch_complexity.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["schema_version"] != "SHDBranchComplexityDiagnostic/v1":
            raise ValueError(f"unexpected report schema: {path}")
        radii.add(float(report["relative_radius"]))
        sample_counts.add(int(report["sample_count"]))
        for item in report["rows"]:
            rows.append({"seed": seed, **item})
        report_hashes[f"seed_{seed}"] = sha256_file(path)
    if len(radii) != 1 or len(sample_counts) != 1:
        raise ValueError("branch-complexity reports do not share radius and sample count")

    output_root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = (
        "mean_unique_spike_traces",
        "median_unique_spike_traces",
        "p90_unique_spike_traces",
        "max_unique_spike_traces",
        "single_spike_trace_fraction",
        "all_points_unique_trace_fraction",
        "mean_unique_predictions",
        "max_unique_predictions",
        "single_prediction_fraction",
        "all_grid_predictions_match_center_fraction",
        "trace_grid_pair_disagreement_fraction",
        "prediction_grid_pair_disagreement_fraction",
    )
    resolutions = sorted({int(row["resolution"]) for row in rows})
    aggregate_rows = []
    for resolution in resolutions:
        selected = [row for row in rows if int(row["resolution"]) == resolution]
        aggregate_rows.append(
            {
                "resolution": resolution,
                "grid_points": resolution * resolution,
                **{
                    metric: _stats([float(row[metric]) for row in selected])
                    for metric in metric_names
                },
            }
        )

    grid_points = np.asarray(
        [row["grid_points"] for row in aggregate_rows], dtype=float
    )
    trace_means = np.asarray(
        [row["mean_unique_spike_traces"]["mean"] for row in aggregate_rows],
        dtype=float,
    )
    branch_growth_exponent = float(
        np.polyfit(np.log(grid_points), np.log(trace_means), 1)[0]
    )
    finest = aggregate_rows[-1]
    summary = {
        "schema_version": "SHDBranchComplexityAggregate/v1",
        "status": "five-seed finite-grid diagnostic; not a continuous-family proof",
        "seeds": list(SEEDS),
        "relative_radius": next(iter(radii)),
        "sample_count_per_seed": next(iter(sample_counts)),
        "resolutions": resolutions,
        "aggregate_rows": aggregate_rows,
        "route_assessment": {
            "finest_grid_points": finest["grid_points"],
            "mean_unique_spike_traces": finest["mean_unique_spike_traces"]["mean"],
            "mean_seed_p90_unique_spike_traces": finest[
                "p90_unique_spike_traces"
            ]["mean"],
            "mean_unique_predictions": finest["mean_unique_predictions"]["mean"],
            "single_prediction_fraction": finest["single_prediction_fraction"][
                "mean"
            ],
            "single_spike_trace_fraction": finest[
                "single_spike_trace_fraction"
            ]["mean"],
            "prediction_grid_pair_disagreement_fraction": finest[
                "prediction_grid_pair_disagreement_fraction"
            ]["mean"],
            "sampled_branch_growth_exponent": branch_growth_exponent,
            "mean_sampled_traces_below_128": bool(
                finest["mean_unique_spike_traces"]["mean"] < 128
            ),
        },
        "interpretation": (
            "Every sampled input changes spike trace within the one-percent joint box, "
            "so a single-branch proof has no sampled coverage. Trace multiplicity grows "
            "sublinearly with grid density and remains tens rather than hundreds on average, "
            "while most inputs retain one prediction. This does not prove that complete "
            "continuous branch enumeration is tractable: sampled trace counts are lower "
            "bounds and must still be multiplied across discrete semantics."
        ),
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    trace_std = np.asarray(
        [row["mean_unique_spike_traces"]["std"] for row in aggregate_rows]
    )
    axes[0].plot(grid_points, trace_means, marker="o", color="#7570b3")
    axes[0].fill_between(
        grid_points,
        np.maximum(1, trace_means - trace_std),
        trace_means + trace_std,
        color="#7570b3",
        alpha=0.18,
    )
    axes[0].plot(grid_points, grid_points, linestyle="--", color="0.45", label="one trace per point")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Sampled semantics points")
    axes[0].set_ylabel("Mean distinct spike traces per input")
    axes[0].set_title("Sampled trajectory branches")
    axes[0].legend(frameon=False, fontsize=8)

    single_prediction = np.asarray(
        [row["single_prediction_fraction"]["mean"] for row in aggregate_rows]
    )
    pair_agreement = np.asarray(
        [
            1.0 - row["prediction_grid_pair_disagreement_fraction"]["mean"]
            for row in aggregate_rows
        ]
    )
    axes[1].plot(
        grid_points,
        single_prediction * 100,
        marker="o",
        color="#1b9e77",
        label="one prediction over grid",
    )
    axes[1].plot(
        grid_points,
        pair_agreement * 100,
        marker="s",
        color="#d95f02",
        label="grid--center pair agreement",
    )
    axes[1].set_xscale("log")
    axes[1].set_ylim(75, 101)
    axes[1].set_xlabel("Sampled semantics points")
    axes[1].set_ylabel(r"Inputs or pairs (\%)")
    axes[1].set_title("Decisions remain comparatively stable")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
