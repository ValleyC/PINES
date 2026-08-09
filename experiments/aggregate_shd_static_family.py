from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="artifacts/shd_v1_static_family")
    parser.add_argument("--output-root", default="results/shd_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "static_family_summary.json"
    rows_path = output / "static_family_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_static_family.pdf"
    figure_png = root / "paper" / "figures" / "shd_static_family.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("static-family aggregate exists; evidence is immutable")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    for seed in seeds:
        path = root / args.input_root / f"seed_{seed}" / "static_family_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["seed"]) != seed:
            raise ValueError("seed mismatch in static-family report")
        report_hashes[f"seed_{seed}"] = sha256_file(path)
        rows.extend({"seed": seed, **row} for row in report["rows"])
    fields = tuple(rows[0].keys())
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    families = (
        "reset",
        "integration",
        "timing",
        "delay",
        "full",
        "fixed_nearest",
        "fixed_floor",
    )
    per_family: list[dict[str, object]] = []
    for family in families:
        selected = [row for row in rows if row["family"] == family]
        static = np.asarray(
            [float(row["static_certified_fraction"]) for row in selected]
        )
        exact = np.asarray(
            [float(row["exact_family_agreement_fraction"]) for row in selected]
        )
        per_family.append(
            {
                "family": family,
                "static_certified_mean": float(np.mean(static)),
                "static_certified_std": float(np.std(static, ddof=1)),
                "static_vacuous_mean": float(1.0 - np.mean(static)),
                "exact_family_agreement_mean": float(np.mean(exact)),
                "exact_family_agreement_std": float(np.std(exact, ddof=1)),
                "relaxation_gap_mean": float(np.mean(exact - static)),
                "observed_unsound_certificates": int(
                    sum(int(row["observed_unsound_certificates"]) for row in selected)
                ),
                "mean_static_seconds": float(
                    np.mean([float(row["static_seconds"]) for row in selected])
                ),
            }
        )
    full = next(item for item in per_family if item["family"] == "full")
    full_seed_rows = [row for row in rows if row["family"] == "full"]
    summary = {
        "schema_version": "SHDStaticFamilyAggregate/v1",
        "status": "five-seed software interval-family evidence; not physical evidence",
        "seeds": list(seeds),
        "audit_samples_per_seed": int(full_seed_rows[0]["samples"]),
        "per_family": per_family,
        "full_family_per_seed_certified_fraction": {
            str(row["seed"]): row["static_certified_fraction"]
            for row in full_seed_rows
        },
        "gate_assessment": {
            "full_family_vacuous_on_more_than_80_percent": bool(
                float(full["static_vacuous_mean"]) > 0.80
            ),
            "every_seed_vacuous_on_more_than_80_percent": all(
                1.0 - float(row["static_certified_fraction"]) > 0.80
                for row in full_seed_rows
            ),
            "no_observed_static_unsoundness": all(
                int(row["observed_unsound_certificates"]) == 0 for row in rows
            ),
        },
        "formulation_assessment": (
            "The sound interval engine is useful for single semantic axes and nearest fixed "
            "point, but the preregistered 16-member high-risk family is vacuous on more than "
            "80 percent of inputs for every seed. This triggers the planned stop/pivot rule "
            "unless a substantially tighter sound abstraction reverses the result."
        ),
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    static_values = [float(item["static_certified_mean"]) * 100 for item in per_family]
    exact_values = [
        float(item["exact_family_agreement_mean"]) * 100 for item in per_family
    ]
    positions = np.arange(len(families))
    width = 0.38
    fig, axis = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        exact_values,
        width=width,
        label="Exact family agreement",
        color="#1b9e77",
    )
    axis.bar(
        positions + width / 2,
        static_values,
        width=width,
        label="Sound interval certificate",
        color="#7570b3",
    )
    axis.axhline(20, color="black", linestyle="--", linewidth=1, label="20% gate")
    axis.set_xticks(positions, [name.replace("_", "\n") for name in families])
    axis.set_ylabel("Mean fraction of audit inputs (%)")
    axis.set_title("Broad family certification crosses the preregistered pivot threshold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
