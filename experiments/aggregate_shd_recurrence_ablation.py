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
        "--input-root", default="artifacts/shd_v1_static_family_zero_recurrence"
    )
    parser.add_argument("--output-root", default="results/shd_v1")
    parser.add_argument(
        "--trained-summary", default="results/shd_v1/static_family_summary.json"
    )
    parser.add_argument(
        "--feedforward-summary",
        default="results/nmnist_v1/static_family_diagnostic_summary.json",
    )
    parser.add_argument("--suffix", default="")
    parser.add_argument("--figure-stem", default="recurrence_ablation_static")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.suffix}" if args.suffix else ""
    summary_path = output / f"recurrence_ablation{suffix}_summary.json"
    rows_path = output / f"recurrence_ablation{suffix}_rows.csv"
    figure_pdf = root / "paper" / "figures" / f"{args.figure_stem}.pdf"
    figure_png = root / "paper" / "figures" / f"{args.figure_stem}.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("recurrence-ablation aggregate exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    for seed in seeds:
        path = root / args.input_root / f"seed_{seed}" / "static_family_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["seed"]) != seed or report["ablation"] != "zero_recurrence":
            raise ValueError("invalid recurrence-ablation report")
        if len(report["rows"]) != 1 or report["rows"][0]["family"] != "full":
            raise ValueError("recurrence ablation must contain only the full family")
        report_hashes[f"seed_{seed}"] = sha256_file(path)
        rows.append({"seed": seed, **report["rows"][0]})

    fields = tuple(rows[0].keys())
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    trained = json.loads((root / args.trained_summary).read_text(encoding="utf-8"))
    feedforward = json.loads(
        (root / args.feedforward_summary).read_text(encoding="utf-8")
    )
    trained_source = next(
        item for item in trained["per_family"] if item["family"] == "full"
    )
    feedforward_source = next(
        item for item in feedforward["per_family"] if item["family"] == "full"
    )

    def normalized(item: dict[str, object]) -> dict[str, object]:
        static_key = (
            "memberwise_static_mean"
            if "memberwise_static_mean" in item
            else "static_certified_mean"
        )
        static_std_key = (
            "memberwise_static_std"
            if "memberwise_static_std" in item
            else "static_certified_std"
        )
        gap_key = (
            "remaining_relaxation_gap_mean"
            if "remaining_relaxation_gap_mean" in item
            else "relaxation_gap_mean"
        )
        return {
            "family": item["family"],
            "static_certified_mean": float(item[static_key]),
            "static_certified_std": float(item[static_std_key]),
            "exact_family_agreement_mean": float(
                item["exact_family_agreement_mean"]
            ),
            "exact_family_agreement_std": float(
                item["exact_family_agreement_std"]
            ),
            "relaxation_gap_mean": float(item[gap_key]),
            "observed_unsound_certificates": int(
                item["observed_unsound_certificates"]
            ),
        }

    trained_full = normalized(trained_source)
    feedforward_full = normalized(feedforward_source)
    static = np.asarray([float(row["static_certified_fraction"]) for row in rows])
    exact = np.asarray(
        [float(row["exact_family_agreement_fraction"]) for row in rows]
    )
    zero_recurrence = {
        "static_certified_mean": float(np.mean(static)),
        "static_certified_std": float(np.std(static, ddof=1)),
        "exact_family_agreement_mean": float(np.mean(exact)),
        "exact_family_agreement_std": float(np.std(exact, ddof=1)),
        "relaxation_gap_mean": float(np.mean(exact - static)),
        "observed_unsound_certificates": int(
            sum(int(row["observed_unsound_certificates"]) for row in rows)
        ),
    }
    summary = {
        "schema_version": "SHDRecurrenceAblationAggregate/v1",
        "status": (
            "five-seed within-model software diagnostic; zeroing recurrence changes "
            "the reference model and is not an accuracy-matched architecture comparison"
        ),
        "seeds": list(seeds),
        "audit_samples_per_seed": int(rows[0]["samples"]),
        "trained_shd": trained_full,
        "zero_recurrence_shd": zero_recurrence,
        "feedforward_nmnist_subset": feedforward_full,
        "causal_diagnostic": {
            "static_coverage_increase": float(
                zero_recurrence["static_certified_mean"]
                - float(trained_full["static_certified_mean"])
            ),
            "exact_agreement_increase": float(
                zero_recurrence["exact_family_agreement_mean"]
                - float(trained_full["exact_family_agreement_mean"])
            ),
            "interpretation": (
                "Holding SHD inputs and all nonrecurrent parameters fixed, removing the "
                "feedback path reverses the 20-percent static-coverage failure and "
                "substantially increases exact semantic agreement. The intervention also "
                "changes model predictions and margins, so it is a mechanism diagnostic, "
                "not an accuracy-matched architecture comparison."
            ),
        },
        "gate_assessment": {
            "zero_recurrence_full_family_certifies_at_least_20_percent": bool(
                zero_recurrence["static_certified_mean"] >= 0.20
            ),
            "no_observed_static_unsoundness": bool(
                zero_recurrence["observed_unsound_certificates"] == 0
            ),
        },
        "input_report_hashes": report_hashes,
        "trained_summary_hash": sha256_file(root / args.trained_summary),
        "feedforward_summary_hash": sha256_file(root / args.feedforward_summary),
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    labels = ("SHD trained\nrecurrent", "SHD zero\nrecurrence", "N-MNIST\nfeedforward")
    static_means = np.asarray(
        [
            float(trained_full["static_certified_mean"]),
            zero_recurrence["static_certified_mean"],
            float(feedforward_full["static_certified_mean"]),
        ]
    ) * 100
    exact_means = np.asarray(
        [
            float(trained_full["exact_family_agreement_mean"]),
            zero_recurrence["exact_family_agreement_mean"],
            float(feedforward_full["exact_family_agreement_mean"]),
        ]
    ) * 100
    positions = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        exact_means,
        width=width,
        label="Exact family agreement",
        color="#1b9e77",
    )
    axis.bar(
        positions + width / 2,
        static_means,
        width=width,
        label="Sound interval certificate",
        color="#7570b3",
    )
    axis.axhline(20, color="black", linestyle="--", linewidth=1, label="20% gate")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Full-family audit inputs (%)")
    axis.set_title("Recurrent feedback drives SHD semantic instability and bound vacuity")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
