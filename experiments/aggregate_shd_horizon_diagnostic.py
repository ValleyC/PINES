from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", default="artifacts/shd_v1_horizon_diagnostic"
    )
    parser.add_argument("--output-root", default="results/shd_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "horizon_diagnostic_summary.json"
    rows_path = output / "horizon_diagnostic_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_horizon_diagnostic.pdf"
    figure_png = root / "paper" / "figures" / "shd_horizon_diagnostic.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("horizon diagnostic aggregate exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    audit_sizes: set[int] = set()
    complete_audit_sizes: set[int] = set()
    horizons: tuple[int, ...] | None = None
    for seed in seeds:
        path = root / args.input_root / f"seed_{seed}" / "horizon_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["seed"]) != seed:
            raise ValueError("seed mismatch in horizon report")
        current_horizons = tuple(int(value) for value in report["horizons"])
        if horizons is None:
            horizons = current_horizons
        elif current_horizons != horizons:
            raise ValueError("inconsistent horizons")
        audit_sizes.add(int(report["audit_samples"]))
        complete_audit_sizes.add(int(report["complete_audit_samples"]))
        report_hashes[f"seed_{seed}"] = sha256_file(path)
        rows.extend({"seed": seed, **row} for row in report["rows"])
    if horizons is None or len(audit_sizes) != 1 or len(complete_audit_sizes) != 1:
        raise ValueError("invalid horizon aggregate dimensions")

    fields = tuple(rows[0].keys())
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    model_conditions = ("trained_recurrent", "zero_recurrence")
    trajectories: dict[str, list[dict[str, object]]] = {}
    for model_condition in model_conditions:
        entries = []
        for horizon in horizons:
            selected = [
                row
                for row in rows
                if row["model_condition"] == model_condition
                and int(row["horizon"]) == horizon
            ]
            if len(selected) != len(seeds):
                raise ValueError("incomplete horizon matrix")
            static = np.asarray(
                [float(row["static_certified_fraction"]) for row in selected]
            )
            exact = np.asarray(
                [float(row["exact_family_agreement_fraction"]) for row in selected]
            )
            margins = np.asarray(
                [float(row["median_reference_margin"]) for row in selected]
            )
            entries.append(
                {
                    "horizon": horizon,
                    "static_certified_mean": float(np.mean(static)),
                    "static_certified_std": float(np.std(static, ddof=1)),
                    "exact_family_agreement_mean": float(np.mean(exact)),
                    "exact_family_agreement_std": float(np.std(exact, ddof=1)),
                    "relaxation_gap_mean": float(np.mean(exact - static)),
                    "median_reference_margin_across_seeds": float(np.median(margins)),
                    "observed_unsound_certificates": int(
                        sum(int(row["observed_unsound_certificates"]) for row in selected)
                    ),
                }
            )
        trajectories[model_condition] = entries

    trained_first = trajectories["trained_recurrent"][0]
    trained_last = trajectories["trained_recurrent"][-1]
    zero_first = trajectories["zero_recurrence"][0]
    zero_last = trajectories["zero_recurrence"][-1]
    summary = {
        "schema_version": "SHDHorizonDiagnosticAggregate/v1",
        "status": (
            "five-seed software mechanism diagnostic on deterministic 256-input "
            "audit subsets; not primary or physical evidence"
        ),
        "seeds": list(seeds),
        "audit_samples_per_seed": next(iter(audit_sizes)),
        "complete_audit_samples_per_seed": next(iter(complete_audit_sizes)),
        "subset_rule": "first 256 entries of each frozen certificate audit split",
        "horizons": list(horizons),
        "trajectories": trajectories,
        "endpoint_changes": {
            "trained_static": float(
                trained_last["static_certified_mean"]
                - trained_first["static_certified_mean"]
            ),
            "trained_exact": float(
                trained_last["exact_family_agreement_mean"]
                - trained_first["exact_family_agreement_mean"]
            ),
            "zero_recurrence_static": float(
                zero_last["static_certified_mean"]
                - zero_first["static_certified_mean"]
            ),
            "zero_recurrence_exact": float(
                zero_last["exact_family_agreement_mean"]
                - zero_first["exact_family_agreement_mean"]
            ),
        },
        "formulation_assessment": (
            "Coverage does not monotonically decay with unroll length. Margin "
            "accumulation raises both exact agreement and static coverage after the "
            "first ten bins, but recurrent models remain unstable at every measured "
            "horizon. The next abstraction should target recurrent transition and spike "
            "branch dependence rather than relying on shorter unrolls."
        ),
        "gate_assessment": {
            "no_observed_static_unsoundness": all(
                int(row["observed_unsound_certificates"]) == 0 for row in rows
            ),
            "trained_recurrent_reaches_20_percent_static": any(
                float(item["static_certified_mean"]) >= 0.20
                for item in trajectories["trained_recurrent"]
            ),
        },
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    colors = {"trained_recurrent": "#d95f02", "zero_recurrence": "#1b9e77"}
    labels = {"trained_recurrent": "Trained recurrent", "zero_recurrence": "Zero recurrence"}
    for model_condition in model_conditions:
        entries = trajectories[model_condition]
        x = np.asarray([int(item["horizon"]) for item in entries])
        for axis, mean_key, std_key, title in (
            (
                axes[0],
                "exact_family_agreement_mean",
                "exact_family_agreement_std",
                "Exact family agreement",
            ),
            (
                axes[1],
                "static_certified_mean",
                "static_certified_std",
                "Sound interval certificate",
            ),
        ):
            mean = np.asarray([float(item[mean_key]) for item in entries]) * 100
            std = np.asarray([float(item[std_key]) for item in entries]) * 100
            axis.plot(
                x,
                mean,
                marker="o",
                color=colors[model_condition],
                label=labels[model_condition],
            )
            axis.fill_between(x, mean - std, mean + std, color=colors[model_condition], alpha=0.15)
            axis.set_title(title)
            axis.set_xlabel("Unrolled time bins")
            axis.set_ylim(0, 100)
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Audit inputs (%)")
    axes[1].axhline(20, color="black", linestyle="--", linewidth=1, label="20% gate")
    model_handles, model_labels = axes[0].get_legend_handles_labels()
    gate_handles, gate_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        model_handles + gate_handles[-1:],
        model_labels + gate_labels[-1:],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Shortening the unroll does not rescue recurrent SHD certification",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
