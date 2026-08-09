from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


SEEDS = (1701, 2718, 3141, 5772, 8119)
FAMILIES = (
    "reference_member",
    "reset",
    "integration",
    "timing",
    "delay",
    "reset_delay",
    "integration_timing",
    "full",
)


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
    artifact_root = root / "artifacts" / "shd_v11_subfamily_grid"
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / "subfamily_grid_summary.json"
    rows_path = output_root / "subfamily_grid_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_subfamily_grid.pdf"
    figure_png = root / "paper" / "figures" / "shd_subfamily_grid.png"
    if any(
        path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)
    ):
        raise FileExistsError("subfamily-grid aggregate destination already exists")

    rows = []
    input_hashes = {}
    for seed in SEEDS:
        path = artifact_root / f"seed_{seed}" / "subfamily_grid.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["schema_version"] != "SHDSubfamilyGridDiagnostic/v1":
            raise ValueError(f"unexpected report schema: {path}")
        for row in report["rows"]:
            rows.append({"seed": seed, **row})
        input_hashes[f"seed_{seed}"] = sha256_file(path)

    output_root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for family in FAMILIES:
        selected = [row for row in rows if row["subfamily"] == family]
        aggregate_rows.append(
            {
                "subfamily": family,
                "member_count": int(selected[0]["member_count"]),
                "center_only_prediction_identity_fraction": _stats(
                    [
                        float(row["center_only_prediction_identity_fraction"])
                        for row in selected
                    ]
                ),
                "grid_prediction_identity_fraction": _stats(
                    [float(row["grid_prediction_identity_fraction"]) for row in selected]
                ),
                "continuous_grid_identity_cost": _stats(
                    [float(row["continuous_grid_identity_cost"]) for row in selected]
                ),
            }
        )

    primary_axes = [
        row
        for row in aggregate_rows
        if row["subfamily"] in ("reset", "integration", "timing", "delay")
    ]
    full = next(row for row in aggregate_rows if row["subfamily"] == "full")
    summary = {
        "schema_version": "SHDSubfamilyGridAggregate/v1",
        "status": "five-seed finite-grid falsification diagnostic; not a proof",
        "seeds": list(SEEDS),
        "sample_count_per_seed": 128,
        "relative_radius": 0.01,
        "grid_resolution": 9,
        "aggregate_rows": aggregate_rows,
        "route_assessment": {
            "minimum_primary_axis_grid_identity_mean": min(
                row["grid_prediction_identity_fraction"]["mean"]
                for row in primary_axes
            ),
            "minimum_primary_axis_grid_identity_min_seed": min(
                row["grid_prediction_identity_fraction"]["min"]
                for row in primary_axes
            ),
            "every_primary_axis_sampled_ceiling_exceeds_twenty_percent": all(
                row["grid_prediction_identity_fraction"]["min"] >= 0.20
                for row in primary_axes
            ),
            "full_family_grid_identity_mean": full[
                "grid_prediction_identity_fraction"
            ]["mean"],
            "full_family_continuous_cost_mean": full[
                "continuous_grid_identity_cost"
            ]["mean"],
        },
        "interpretation": (
            "Individual semantic-axis families retain substantially more sampled prediction "
            "identity than the complete Cartesian family. This supports a targeted bounded-"
            "axis fallback formulation if the sound full-family analyzer remains vacuous; "
            "finite-grid ceilings do not themselves certify either formulation."
        ),
        "input_report_hashes": input_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    labels = {
        "reference_member": "continuous only",
        "reset": "reset",
        "integration": "integration",
        "timing": "timing",
        "delay": "delay",
        "reset_delay": "reset + delay",
        "integration_timing": "integration + timing",
        "full": "all 16",
    }
    x = np.arange(len(aggregate_rows))
    center_means = np.asarray(
        [row["center_only_prediction_identity_fraction"]["mean"] for row in aggregate_rows]
    )
    center_stds = np.asarray(
        [row["center_only_prediction_identity_fraction"]["std"] for row in aggregate_rows]
    )
    grid_means = np.asarray(
        [row["grid_prediction_identity_fraction"]["mean"] for row in aggregate_rows]
    )
    grid_stds = np.asarray(
        [row["grid_prediction_identity_fraction"]["std"] for row in aggregate_rows]
    )
    fig, axis = plt.subplots(figsize=(9.2, 3.9), constrained_layout=True)
    width = 0.38
    axis.bar(
        x - width / 2,
        center_means * 100,
        width,
        yerr=center_stds * 100,
        capsize=2,
        color="#1b9e77",
        label="center-only discrete family",
    )
    axis.bar(
        x + width / 2,
        grid_means * 100,
        width,
        yerr=grid_stds * 100,
        capsize=2,
        color="#7570b3",
        label="same family over 9x9 grid",
    )
    axis.axhline(20, linestyle="--", color="0.35", label="coverage gate")
    axis.set_xticks(x, [labels[row["subfamily"]] for row in aggregate_rows], rotation=24)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Sampled prediction identity (%)")
    axis.set_title("Continuous uncertainty reduces every SHD subfamily ceiling")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
