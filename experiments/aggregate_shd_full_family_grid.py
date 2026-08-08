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
    artifact_root = root / "artifacts" / "shd_v5_full_family_grid"
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / "full_family_grid_summary.json"
    rows_path = output_root / "full_family_grid_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_full_family_grid.pdf"
    figure_png = root / "paper" / "figures" / "shd_full_family_grid.png"
    if any(
        path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)
    ):
        raise FileExistsError("full-family grid aggregate destination already exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    member_hashes: list[str] | None = None
    for seed in SEEDS:
        path = artifact_root / f"seed_{seed}" / "full_family_grid.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["schema_version"] != "SHDFullFamilyGridDiagnostic/v1":
            raise ValueError(f"unexpected report schema: {path}")
        current_hashes = list(report["member_semantics_hashes"])
        if member_hashes is None:
            member_hashes = current_hashes
        elif current_hashes != member_hashes:
            raise ValueError("semantic member order differs between reports")
        for item in report["rows"]:
            flat = {key: value for key, value in item.items() if not isinstance(value, list)}
            rows.append({"seed": seed, **flat})
        report_hashes[f"seed_{seed}"] = sha256_file(path)
    assert member_hashes is not None

    output_root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    resolutions = sorted({int(row["resolution"]) for row in rows})
    aggregate_rows = []
    member_rows = []
    for resolution in resolutions:
        selected = [row for row in rows if int(row["resolution"]) == resolution]
        source_items = []
        for seed in SEEDS:
            path = artifact_root / f"seed_{seed}" / "full_family_grid.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            source_items.append(
                next(item for item in report["rows"] if item["resolution"] == resolution)
            )
        aggregate_rows.append(
            {
                "resolution": resolution,
                "grid_points_per_member": resolution * resolution,
                "total_sampled_semantics": 16 * resolution * resolution,
                "full_family_prediction_identity_fraction": _stats(
                    [
                        float(item["full_family_prediction_identity_fraction"])
                        for item in source_items
                    ]
                ),
                "mean_unique_predictions": _stats(
                    [float(item["mean_unique_predictions"]) for item in source_items]
                ),
                "sample_input_pair_disagreement_fraction": _stats(
                    [
                        float(item["sample_input_pair_disagreement_fraction"])
                        for item in source_items
                    ]
                ),
                "mean_member_prediction_identity_fraction": _stats(
                    [
                        float(item["mean_member_prediction_identity_fraction"])
                        for item in source_items
                    ]
                ),
                "reference_member_prediction_identity_fraction": _stats(
                    [
                        float(item["per_member_prediction_identity_fractions"][0])
                        for item in source_items
                    ]
                ),
            }
        )
        for member_index, member_hash in enumerate(member_hashes):
            member_rows.append(
                {
                    "resolution": resolution,
                    "member_index": member_index,
                    "member_semantics_hash": member_hash,
                    "prediction_identity_fraction": _stats(
                        [
                            float(
                                item["per_member_prediction_identity_fractions"][
                                    member_index
                                ]
                            )
                            for item in source_items
                        ]
                    ),
                }
            )

    finest = aggregate_rows[-1]
    summary = {
        "schema_version": "SHDFullFamilyGridAggregate/v1",
        "status": "five-seed finite-grid falsification diagnostic; not a proof",
        "seeds": list(SEEDS),
        "sample_count_per_seed": 128,
        "relative_radius": 0.01,
        "member_count": len(member_hashes),
        "member_semantics_hashes": member_hashes,
        "aggregate_rows": aggregate_rows,
        "member_rows": member_rows,
        "route_assessment": {
            "finest_full_family_identity_mean": finest[
                "full_family_prediction_identity_fraction"
            ]["mean"],
            "finest_full_family_identity_min_seed": finest[
                "full_family_prediction_identity_fraction"
            ]["min"],
            "finest_reference_member_identity_mean": finest[
                "reference_member_prediction_identity_fraction"
            ]["mean"],
            "sampled_ceiling_exceeds_twenty_percent_every_seed": bool(
                finest["full_family_prediction_identity_fraction"]["min"] >= 0.20
            ),
            "headroom_above_twenty_percent_mean": finest[
                "full_family_prediction_identity_fraction"
            ]["mean"]
            - 0.20,
        },
        "interpretation": (
            "The sampled full-family ceiling remains above the 20-percent coverage gate "
            "for every seed, so improved sound analysis is not precluded by true prediction "
            "instability. The margin is narrow and finite-grid agreement can only overstate "
            "continuous-family certificate coverage."
        ),
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    x = np.asarray([row["grid_points_per_member"] for row in aggregate_rows])
    for key, color, label in (
        (
            "reference_member_prediction_identity_fraction",
            "#1b9e77",
            "reference discrete member",
        ),
        (
            "full_family_prediction_identity_fraction",
            "#7570b3",
            "all 16 discrete members",
        ),
    ):
        means = np.asarray([row[key]["mean"] for row in aggregate_rows])
        stds = np.asarray([row[key]["std"] for row in aggregate_rows])
        axes[0].plot(x, means * 100, marker="o", color=color, label=label)
        axes[0].fill_between(
            x,
            np.maximum(0, means - stds) * 100,
            np.minimum(1, means + stds) * 100,
            color=color,
            alpha=0.18,
        )
    axes[0].axhline(20, linestyle="--", color="0.35", label="coverage gate")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("Grid points per discrete member")
    axes[0].set_ylabel("Sampled unchanged-prediction inputs (%)")
    axes[0].set_title("Empirical coverage ceiling")
    axes[0].legend(frameon=False, fontsize=8)

    finest_members = [
        row for row in member_rows if row["resolution"] == resolutions[-1]
    ]
    matrix = np.asarray(
        [row["prediction_identity_fraction"]["mean"] for row in finest_members]
    ).reshape(4, 4)
    image = axes[1].imshow(matrix * 100, vmin=0, vmax=100, cmap="viridis")
    axes[1].set_xticks(range(4), ("sub, d0", "sub, d1", "value, d0", "value, d1"))
    axes[1].set_yticks(range(4), ("FE, post", "FE, pre", "EXP, post", "EXP, pre"))
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].set_title("Member identity at 9x9 grid")
    for row in range(4):
        for column in range(4):
            color = "white" if matrix[row, column] < 0.55 else "black"
            axes[1].text(
                column,
                row,
                f"{matrix[row, column] * 100:.0f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(image, ax=axes[1], label="Prediction identity (%)", shrink=0.85)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
