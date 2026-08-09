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
    parser.add_argument(
        "--input-root", default="artifacts/nmnist_v1_static_family"
    )
    parser.add_argument("--output-root", default="results/nmnist_v1")
    parser.add_argument(
        "--shd-summary", default="results/shd_v1/static_family_summary.json"
    )
    parser.add_argument("--suffix", default="diagnostic")
    parser.add_argument("--figure-stem", default="architecture_static_family")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    families = ("reset", "integration", "timing", "delay", "full")
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"static_family_{args.suffix}_summary.json"
    rows_path = output / f"static_family_{args.suffix}_rows.csv"
    figure_pdf = root / "paper" / "figures" / f"{args.figure_stem}.pdf"
    figure_png = root / "paper" / "figures" / f"{args.figure_stem}.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("N-MNIST static-family aggregate exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    audit_samples: set[int] = set()
    complete_audit_samples: set[int] = set()
    for seed in seeds:
        path = root / args.input_root / f"seed_{seed}" / "static_family_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["seed"]) != seed:
            raise ValueError("seed mismatch in N-MNIST static-family report")
        report_hashes[f"seed_{seed}"] = sha256_file(path)
        audit_samples.add(int(report["audit_samples"]))
        complete_audit_samples.add(int(report["complete_audit_samples"]))
        rows.extend({"seed": seed, **row} for row in report["rows"])
    if len(audit_samples) != 1 or len(complete_audit_samples) != 1:
        raise ValueError("inconsistent N-MNIST audit sizes")

    fields = tuple(rows[0].keys())
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

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
                "exact_family_agreement_mean": float(np.mean(exact)),
                "exact_family_agreement_std": float(np.std(exact, ddof=1)),
                "relaxation_gap_mean": float(np.mean(exact - static)),
                "observed_unsound_certificates": int(
                    sum(int(row["observed_unsound_certificates"]) for row in selected)
                ),
            }
        )
    full = next(item for item in per_family if item["family"] == "full")
    summary = {
        "schema_version": "NMNISTStaticFamilyDiagnosticAggregate/v1",
        "status": (
            "five-seed software diagnostic on deterministic 128-input subsets; "
            "not primary or physical evidence"
        ),
        "seeds": list(seeds),
        "audit_samples_per_seed": next(iter(audit_samples)),
        "complete_audit_samples_per_seed": next(iter(complete_audit_samples)),
        "subset_rule": "first 128 entries of each frozen certificate audit split",
        "per_family": per_family,
        "gate_assessment": {
            "full_family_certifies_at_least_20_percent": bool(
                float(full["static_certified_mean"]) >= 0.20
            ),
            "full_family_certifies_at_least_90_percent": bool(
                float(full["static_certified_mean"]) >= 0.90
            ),
            "no_observed_static_unsoundness": all(
                int(row["observed_unsound_certificates"]) == 0 for row in rows
            ),
        },
        "formulation_assessment": (
            "The identical interval abstraction remains tight on the feedforward "
            "negative control, so the SHD pivot is specifically a recurrent-state "
            "and reset-branching problem rather than a universal failure of static "
            "semantic certification."
        ),
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    shd = json.loads((root / args.shd_summary).read_text(encoding="utf-8"))
    shd_by_family = {item["family"]: item for item in shd["per_family"]}
    nmnist_by_family = {item["family"]: item for item in per_family}

    def static_mean(item: dict[str, object]) -> float:
        if "memberwise_static_mean" in item:
            return float(item["memberwise_static_mean"])
        return float(item["static_certified_mean"])
    positions = np.arange(len(families))
    width = 0.38
    fig, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        [static_mean(shd_by_family[name]) * 100 for name in families],
        width=width,
        label="SHD recurrent (861/seed)",
        color="#d95f02",
    )
    axis.bar(
        positions + width / 2,
        [float(nmnist_by_family[name]["static_certified_mean"]) * 100 for name in families],
        width=width,
        label="N-MNIST feedforward (128/seed)",
        color="#1b9e77",
    )
    axis.axhline(20, color="black", linestyle="--", linewidth=1, label="20% gate")
    axis.set_xticks(positions, families)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Soundly certified inputs (%)")
    axis.set_title("Static-family certificate contrast across trained models")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
