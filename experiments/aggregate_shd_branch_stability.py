from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


SEEDS = (1701, 2718, 3141, 5772, 8119)
AXES = ("timestep", "threshold", "joint_corners")


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
    artifact_root = root / "artifacts" / "shd_v3_branch_stability"
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / "branch_stability_summary.json"
    rows_path = output_root / "branch_stability_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_branch_stability.pdf"
    figure_png = root / "paper" / "figures" / "shd_branch_stability.png"
    if any(
        path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)
    ):
        raise FileExistsError("branch-stability aggregate destination already exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    for seed in SEEDS:
        path = artifact_root / f"seed_{seed}" / "branch_stability.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["schema_version"] != "SHDBranchStabilityDiagnostic/v1":
            raise ValueError(f"unexpected report schema: {path}")
        for item in report["rows"]:
            rows.append({"seed": seed, **item})
        report_hashes[f"seed_{seed}"] = sha256_file(path)

    output_root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    radii = sorted({float(row["relative_radius"]) for row in rows})
    for radius in radii:
        for axis in AXES:
            selected = [
                row
                for row in rows
                if float(row["relative_radius"]) == radius and row["axis"] == axis
            ]
            aggregate_rows.append(
                {
                    "relative_radius": radius,
                    "axis": axis,
                    "corner_trace_identity_fraction": _stats(
                        [
                            float(row["corner_trace_identity_fraction"])
                            for row in selected
                        ]
                    ),
                    "corner_prediction_identity_fraction": _stats(
                        [
                            float(row["corner_prediction_identity_fraction"])
                            for row in selected
                        ]
                    ),
                    "prediction_identity_with_trace_change_fraction": _stats(
                        [
                            float(
                                row[
                                    "prediction_identity_with_trace_change_fraction"
                                ]
                            )
                            for row in selected
                        ]
                    ),
                    "mean_per_input_max_spike_hamming": _stats(
                        [
                            float(row["mean_per_input_max_spike_hamming"])
                            for row in selected
                        ]
                    ),
                }
            )

    joint_one_percent = next(
        row
        for row in aggregate_rows
        if row["relative_radius"] == 1e-2 and row["axis"] == "joint_corners"
    )
    summary = {
        "schema_version": "SHDBranchStabilityAggregate/v1",
        "status": "five-seed endpoint diagnostic; not a continuous-box proof",
        "seeds": list(SEEDS),
        "radii": radii,
        "axes": list(AXES),
        "aggregate_rows": aggregate_rows,
        "route_assessment": {
            "joint_one_percent_trace_identity_mean": joint_one_percent[
                "corner_trace_identity_fraction"
            ]["mean"],
            "joint_one_percent_prediction_identity_mean": joint_one_percent[
                "corner_prediction_identity_fraction"
            ]["mean"],
            "fixed_center_trace_certificate_has_twenty_percent_endpoint_headroom": bool(
                joint_one_percent["corner_trace_identity_fraction"]["mean"] >= 0.20
            ),
        },
        "interpretation": (
            "Corner trace identity is an optimistic upper bound on coverage for a method that "
            "must preserve the center spike branch throughout the continuous box: observed "
            "corner changes rule out that branch, while observed corner identity is not a proof "
            "about the interior. Prediction identity can remain high despite trace changes."
        ),
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    labels = {
        "timestep": "Timestep endpoints",
        "threshold": "Threshold endpoints",
        "joint_corners": "Joint corners",
    }
    colors = {
        "timestep": "#1b9e77",
        "threshold": "#d95f02",
        "joint_corners": "#7570b3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for axis in AXES:
        selected = [row for row in aggregate_rows if row["axis"] == axis]
        x = np.asarray([float(row["relative_radius"]) for row in selected])
        trace_mean = np.asarray(
            [row["corner_trace_identity_fraction"]["mean"] for row in selected]
        )
        trace_std = np.asarray(
            [row["corner_trace_identity_fraction"]["std"] for row in selected]
        )
        prediction_mean = np.asarray(
            [row["corner_prediction_identity_fraction"]["mean"] for row in selected]
        )
        prediction_std = np.asarray(
            [row["corner_prediction_identity_fraction"]["std"] for row in selected]
        )
        axes[0].plot(x, trace_mean * 100, marker="o", color=colors[axis], label=labels[axis])
        axes[0].fill_between(
            x,
            np.maximum(0, trace_mean - trace_std) * 100,
            np.minimum(1, trace_mean + trace_std) * 100,
            color=colors[axis],
            alpha=0.15,
        )
        axes[1].plot(
            x, prediction_mean * 100, marker="o", color=colors[axis], label=labels[axis]
        )
        axes[1].fill_between(
            x,
            np.maximum(0, prediction_mean - prediction_std) * 100,
            np.minimum(1, prediction_mean + prediction_std) * 100,
            color=colors[axis],
            alpha=0.15,
        )
    axes[0].set_title("Center spike trace retained at tested endpoints")
    axes[0].set_ylabel("Audit inputs (%)")
    axes[1].set_title("Prediction retained at tested endpoints")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Relative radius")
        axis.set_ylim(-2, 102)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
