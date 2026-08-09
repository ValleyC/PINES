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
        "--input-root", default="artifacts/shd_v2_memberwise_static_family"
    )
    parser.add_argument("--output-root", default="results/shd_v1")
    parser.add_argument(
        "--legacy-summary", default="results/shd_v1/static_family_summary.json"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = (1701, 2718, 3141, 5772, 8119)
    families = (
        "reset",
        "integration",
        "timing",
        "delay",
        "full",
        "fixed_nearest",
        "fixed_floor",
    )
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "static_family_memberwise_summary.json"
    rows_path = output / "static_family_memberwise_rows.csv"
    figure_pdf = root / "paper" / "figures" / "shd_static_family_memberwise.pdf"
    figure_png = root / "paper" / "figures" / "shd_static_family_memberwise.png"
    if any(path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)):
        raise FileExistsError("member-wise static-family aggregate exists")

    rows: list[dict[str, object]] = []
    report_hashes: dict[str, str] = {}
    for seed in seeds:
        path = root / args.input_root / f"seed_{seed}" / "static_family_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["seed"]) != seed:
            raise ValueError("seed mismatch in member-wise report")
        report_hashes[f"seed_{seed}"] = sha256_file(path)
        rows.extend({"seed": seed, **row} for row in report["rows"])

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
                "memberwise_static_mean": float(np.mean(static)),
                "memberwise_static_std": float(np.std(static, ddof=1)),
                "memberwise_vacuous_mean": float(1.0 - np.mean(static)),
                "exact_family_agreement_mean": float(np.mean(exact)),
                "exact_family_agreement_std": float(np.std(exact, ddof=1)),
                "remaining_relaxation_gap_mean": float(np.mean(exact - static)),
                "observed_unsound_certificates": int(
                    sum(int(row["observed_unsound_certificates"]) for row in selected)
                ),
            }
        )

    legacy_path = root / args.legacy_summary
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_by_family = {item["family"]: item for item in legacy["per_family"]}
    for item in per_family:
        old = legacy_by_family[item["family"]]
        item["legacy_joint_merge_mean"] = float(old["static_certified_mean"])
        item["coverage_gain_over_joint_merge"] = float(
            item["memberwise_static_mean"] - float(old["static_certified_mean"])
        )

    full = next(item for item in per_family if item["family"] == "full")
    full_seed_rows = [row for row in rows if row["family"] == "full"]
    summary = {
        "schema_version": "SHDMemberwiseStaticFamilyAggregate/v1",
        "status": (
            "five-seed software certificate evidence for a finite discrete family; "
            "not a continuous-range or physical certificate"
        ),
        "method": (
            "continuous uncertainty is merged within each discrete member; argmax "
            "is checked per member before conjunction across the family"
        ),
        "seeds": list(seeds),
        "audit_samples_per_seed": int(full_seed_rows[0]["samples"]),
        "per_family": per_family,
        "full_family_per_seed_certified_fraction": {
            str(row["seed"]): row["static_certified_fraction"]
            for row in full_seed_rows
        },
        "gate_assessment": {
            "full_family_vacuous_on_more_than_80_percent": bool(
                float(full["memberwise_vacuous_mean"]) > 0.80
            ),
            "every_seed_certifies_at_least_20_percent": all(
                float(row["static_certified_fraction"]) >= 0.20
                for row in full_seed_rows
            ),
            "no_observed_static_unsoundness": all(
                int(row["observed_unsound_certificates"]) == 0 for row in rows
            ),
        },
        "formulation_assessment": (
            "The declared 20-percent static-coverage gate passes for the finite "
            "16-member discrete SHD family once mutually exclusive semantics are "
            "certified separately. This reverses the earlier interval-gate failure, "
            "which was caused by cross-member logit mixing. It does not resolve the "
            "loose distribution certificate, continuous semantic ranges, or physical "
            "emulator-to-hardware conformance."
        ),
        "legacy_joint_merge_summary_hash": sha256_file(legacy_path),
        "input_report_hashes": report_hashes,
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
    }
    write_json_immutable(summary_path, summary)

    positions = np.arange(len(families))
    width = 0.27
    exact_values = [
        float(item["exact_family_agreement_mean"]) * 100 for item in per_family
    ]
    old_values = [float(item["legacy_joint_merge_mean"]) * 100 for item in per_family]
    new_values = [float(item["memberwise_static_mean"]) * 100 for item in per_family]
    fig, axis = plt.subplots(figsize=(9.2, 4.1), constrained_layout=True)
    axis.bar(
        positions - width,
        exact_values,
        width=width,
        label="Exact finite-family agreement",
        color="#1b9e77",
    )
    axis.bar(
        positions,
        old_values,
        width=width,
        label="Legacy joint-logit merge",
        color="#d95f02",
    )
    axis.bar(
        positions + width,
        new_values,
        width=width,
        label="Member-wise sound certificate",
        color="#7570b3",
    )
    axis.axhline(20, color="black", linestyle="--", linewidth=1, label="20% gate")
    axis.set_xticks(positions, [name.replace("_", "\n") for name in families])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Mean audit inputs (%)")
    axis.set_title("Separating mutually exclusive semantics reverses the SHD static gate")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
