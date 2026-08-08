from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable


SEEDS = (1701, 2718, 3141, 5772, 8119)
METHODS = (
    "no_repair",
    "certificate_directed",
    "logit_only",
    "global_threshold",
    "per_platform_qat",
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
    artifact_root = root / "artifacts" / "shd_v12_repair_family_grid"
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / "repair_family_grid_summary.json"
    rows_path = output_root / "repair_family_grid_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_repair_family_grid.pdf"
    figure_png = root / "paper" / "figures" / "shd_repair_family_grid.png"
    if any(
        path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)
    ):
        raise FileExistsError("repair-family grid aggregate destination already exists")

    rows = []
    input_hashes = {}
    for seed in SEEDS:
        path = artifact_root / f"seed_{seed}" / "reset_to_value_family_grid.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["schema_version"] != "SHDRepairFamilyGridDiagnostic/v1":
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
    no_repair_by_seed = {
        int(row["seed"]): float(row["grid_family_identity_fraction"])
        for row in rows
        if row["method"] == "no_repair"
    }
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate_rows.append(
            {
                "method": method,
                "labels_used": selected[0]["labels_used"],
                "optimization_steps": int(selected[0]["optimization_steps"]),
                "trainable_parameters": int(selected[0]["trainable_parameters"]),
                "center_target_identity_fraction": _stats(
                    [float(row["center_target_identity_fraction"]) for row in selected]
                ),
                "grid_family_identity_fraction": _stats(
                    [float(row["grid_family_identity_fraction"]) for row in selected]
                ),
                "continuous_grid_identity_cost": _stats(
                    [float(row["continuous_grid_identity_cost"]) for row in selected]
                ),
                "grid_pair_disagreement_fraction": _stats(
                    [float(row["grid_pair_disagreement_fraction"]) for row in selected]
                ),
                "paired_grid_identity_gain_over_no_repair": _stats(
                    [
                        float(row["grid_family_identity_fraction"])
                        - no_repair_by_seed[int(row["seed"])]
                        for row in selected
                    ]
                ),
            }
        )

    proposed = next(
        row for row in aggregate_rows if row["method"] == "certificate_directed"
    )
    logit = next(row for row in aggregate_rows if row["method"] == "logit_only")
    summary = {
        "schema_version": "SHDRepairFamilyGridAggregate/v1",
        "status": "five-seed finite-grid post-repair diagnostic; not a proof",
        "seeds": list(SEEDS),
        "condition": "reset_to_value",
        "sample_count_per_seed": 128,
        "relative_radius": 0.01,
        "grid_resolution": 9,
        "aggregate_rows": aggregate_rows,
        "route_assessment": {
            "certificate_directed_grid_identity_mean": proposed[
                "grid_family_identity_fraction"
            ]["mean"],
            "certificate_directed_gain_over_no_repair_mean": proposed[
                "paired_grid_identity_gain_over_no_repair"
            ]["mean"],
            "certificate_directed_gain_positive_every_seed": proposed[
                "paired_grid_identity_gain_over_no_repair"
            ]["min"]
            > 0,
            "certificate_directed_minus_logit_only_mean": proposed[
                "grid_family_identity_fraction"
            ]["mean"]
            - logit["grid_family_identity_fraction"]["mean"],
            "certificate_directed_beats_logit_only_mean": proposed[
                "grid_family_identity_fraction"
            ]["mean"]
            > logit["grid_family_identity_fraction"]["mean"],
        },
        "interpretation": (
            "Restricted label-free repair enlarges the sampled continuous-family identity "
            "ceiling in every seed, but the certificate-directed objective does not beat "
            "matched logit-only imitation on mean family identity. A sound post-repair "
            "certificate remains necessary."
        ),
        "input_report_hashes": input_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    labels = {
        "no_repair": "no repair",
        "certificate_directed": "certificate-directed",
        "logit_only": "logit-only",
        "global_threshold": "global threshold",
        "per_platform_qat": "labeled fine-tune",
    }
    x = np.arange(len(aggregate_rows))
    width = 0.38
    center_means = np.asarray(
        [row["center_target_identity_fraction"]["mean"] for row in aggregate_rows]
    )
    center_stds = np.asarray(
        [row["center_target_identity_fraction"]["std"] for row in aggregate_rows]
    )
    grid_means = np.asarray(
        [row["grid_family_identity_fraction"]["mean"] for row in aggregate_rows]
    )
    grid_stds = np.asarray(
        [row["grid_family_identity_fraction"]["std"] for row in aggregate_rows]
    )
    gains = np.asarray(
        [row["paired_grid_identity_gain_over_no_repair"]["mean"] for row in aggregate_rows]
    )
    gain_stds = np.asarray(
        [row["paired_grid_identity_gain_over_no_repair"]["std"] for row in aggregate_rows]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
    axes[0].bar(
        x - width / 2,
        center_means * 100,
        width,
        yerr=center_stds * 100,
        capsize=2,
        color="#1b9e77",
        label="center target",
    )
    axes[0].bar(
        x + width / 2,
        grid_means * 100,
        width,
        yerr=grid_stds * 100,
        capsize=2,
        color="#7570b3",
        label="9x9 target family",
    )
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Identity to source/reference (%)")
    axes[0].set_title("Post-repair family ceiling")
    axes[0].legend(frameon=False, fontsize=8)

    colors = ["0.6", "#d95f02", "#1b9e77", "#7570b3", "#e6ab02"]
    axes[1].bar(
        x,
        gains * 100,
        yerr=gain_stds * 100,
        capsize=2,
        color=colors,
    )
    axes[1].axhline(0, color="0.35", linewidth=1)
    axes[1].set_ylabel("Paired gain over no repair (points)")
    axes[1].set_title("Repair enlarges sampled headroom")
    for axis in axes:
        axis.set_xticks(x, [labels[row["method"]] for row in aggregate_rows], rotation=28)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["route_assessment"], indent=2))


if __name__ == "__main__":
    main()
