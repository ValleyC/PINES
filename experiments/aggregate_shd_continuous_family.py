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
        "--input-root", default="artifacts/shd_v1_continuous_family"
    )
    parser.add_argument("--output-root", default="results/shd_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "continuous_family_summary.json"
    rows_path = output / "continuous_family_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_continuous_family.pdf"
    figure_png = root / "paper" / "figures" / "shd_continuous_family.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("continuous-family aggregate exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    for seed in seeds:
        path = root / args.input_root / f"seed_{seed}" / "continuous_family_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["seed"]) != seed:
            raise ValueError("seed mismatch in continuous-family report")
        report_hashes[f"seed_{seed}"] = sha256_file(path)
        rows.extend({"seed": seed, **row} for row in report["rows"])

    fields = tuple(rows[0].keys())
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    radii = tuple(sorted({float(row["relative_radius"]) for row in rows}))
    per_radius = []
    for radius in radii:
        selected = [
            row for row in rows if float(row["relative_radius"]) == radius
        ]
        static = np.asarray(
            [float(row["static_certified_fraction"]) for row in selected]
        )
        corners = np.asarray(
            [float(row["sampled_corner_agreement_fraction"]) for row in selected]
        )
        per_radius.append(
            {
                "relative_radius": radius,
                "static_certified_mean": float(np.mean(static)),
                "static_certified_std": float(np.std(static, ddof=1)),
                "static_vacuous_mean": float(1.0 - np.mean(static)),
                "sampled_corner_agreement_mean": float(np.mean(corners)),
                "sampled_corner_agreement_std": float(np.std(corners, ddof=1)),
                "sampled_corner_counterexamples_to_static": int(
                    sum(
                        int(row["sampled_corner_counterexamples_to_static"])
                        for row in selected
                    )
                ),
            }
        )

    summary = {
        "schema_version": "SHDContinuousFamilyAggregate/v1",
        "status": (
            "five-seed sound continuous-box software evidence; corner agreement is "
            "diagnostic only and is not a proof over the continuous interior"
        ),
        "seeds": list(seeds),
        "audit_samples_per_seed": int(rows[0]["audit_samples"]),
        "validation_samples_per_seed": int(rows[0]["validation_samples"]),
        "per_radius": per_radius,
        "gate_assessment": {
            "vacuous_on_more_than_80_percent_at_one_percent": bool(
                float(per_radius[0]["static_vacuous_mean"]) > 0.80
            ),
            "every_seed_zero_coverage_at_one_percent": all(
                float(row["static_certified_fraction"]) == 0.0
                for row in rows
                if float(row["relative_radius"]) == radii[0]
            ),
            "no_sampled_corner_counterexample": all(
                int(row["sampled_corner_counterexamples_to_static"]) == 0
                for row in rows
            ),
        },
        "formulation_assessment": (
            "Member-wise checking rescues the finite enumerated family, but the "
            "promised bounded-family claim is unsupported by the current interval "
            "domain: joint timestep and threshold uncertainty of only plus/minus one "
            "percent yields zero certified SHD inputs for every seed. Endpoint "
            "agreement remains nonzero, confirming substantial abstraction slack."
        ),
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    positions = np.arange(len(per_radius))
    width = 0.38
    static_values = [float(item["static_certified_mean"]) * 100 for item in per_radius]
    corner_values = [
        float(item["sampled_corner_agreement_mean"]) * 100 for item in per_radius
    ]
    corner_errors = [
        float(item["sampled_corner_agreement_std"]) * 100 for item in per_radius
    ]
    fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        corner_values,
        width=width,
        yerr=corner_errors,
        capsize=3,
        label="Endpoint-grid agreement (diagnostic)",
        color="#1b9e77",
    )
    axis.bar(
        positions + width / 2,
        static_values,
        width=width,
        label="Sound continuous-box certificate",
        color="#7570b3",
    )
    axis.axhline(20, color="black", linestyle="--", linewidth=1, label="20% gate")
    axis.set_xticks(
        positions,
        [f"±{100 * float(item['relative_radius']):.0f}%" for item in per_radius],
    )
    axis.set_ylim(0, 50)
    axis.set_xlabel("Joint timestep and threshold uncertainty")
    axis.set_ylabel("Audit inputs (%)")
    axis.set_title("Continuous uncertainty makes the current SHD abstraction vacuous")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
