from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, spearmanr

from transportcert.artifacts import code_revision, sha256_file, write_json_immutable


def _exact_interval(successes: int, samples: int, confidence: float) -> list[float]:
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, samples - successes + 1)
    )
    upper = 1.0 if successes == samples else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, samples - successes)
    )
    return [lower, upper]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--grid-report", required=True)
    parser.add_argument("--sobol-report", required=True)
    parser.add_argument(
        "--screen-summary",
        default="results/shd_v1/hybrid_family_audit_summary.json",
    )
    parser.add_argument(
        "--output", default="results/shd_v1/hybrid_family_full_audit_summary.json"
    )
    parser.add_argument(
        "--figure", default="paper/figures/shd_hybrid_full_audit.pdf"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    audit_path = root / args.audit_report
    grid_path = root / args.grid_report
    sobol_path = root / args.sobol_report
    screen_path = root / args.screen_summary
    output_path = root / args.output
    figure_path = root / args.figure
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    with grid_path.open("r", encoding="utf-8") as handle:
        grid = json.load(handle)
    with sobol_path.open("r", encoding="utf-8") as handle:
        sobol = json.load(handle)
    with screen_path.open("r", encoding="utf-8") as handle:
        screen = json.load(handle)
    if audit.get("schema_version") != "SHDHybridFamilyFullAuditResult/v1":
        raise ValueError("unsupported full hybrid audit report")
    if grid.get("schema_version") != "SHDHybridAuditGridValidation/v1":
        raise ValueError("unsupported hybrid grid validation report")
    if sobol.get("schema_version") != "SHDHybridAuditSobolValidation/v1":
        raise ValueError("unsupported hybrid Sobol validation report")
    if grid["audit_report_hash"] != sha256_file(audit_path):
        raise ValueError("grid validation does not reference the supplied audit")
    if sobol["audit_report_hash"] != sha256_file(audit_path):
        raise ValueError("Sobol validation does not reference the supplied audit")
    shard_revisions = set()
    for shard_record in audit["shards"]:
        shard_path = root / shard_record["path"]
        if sha256_file(shard_path) != shard_record["hash"]:
            raise ValueError(f"audit shard hash mismatch: {shard_path}")
        with shard_path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        if shard.get("config_hash") != audit["config_hash"]:
            raise ValueError(f"audit shard config mismatch: {shard_path}")
        shard_revisions.add(shard.get("code_revision"))
    if shard_revisions != {audit["code_revision"]}:
        raise ValueError("audit shards do not share the report code revision")

    audit_rows = {
        (row["seed"], row["audit_position"], row["dataset_index"]): row
        for row in audit["rows"]
    }
    grid_rows = {
        (row["seed"], row["audit_position"], row["dataset_index"]): row
        for row in grid["rows"]
    }
    sobol_rows = {
        (row["seed"], row["audit_position"], row["dataset_index"]): row
        for row in sobol["rows"]
    }
    if len(audit_rows) != len(audit["rows"]):
        raise ValueError("duplicate input identities in audit report")
    if len(audit_rows) != int(audit["sample_count"]):
        raise ValueError("audit sample count does not match unique rows")
    if len(grid_rows) != len(grid["rows"]):
        raise ValueError("duplicate input identities in grid report")
    if len(sobol_rows) != len(sobol["rows"]):
        raise ValueError("duplicate input identities in Sobol report")
    if audit_rows.keys() != grid_rows.keys() or audit_rows.keys() != sobol_rows.keys():
        raise ValueError("audit and falsification rows do not identify the same inputs")
    audit_seeds = [int(seed) for seed in audit["config"]["seeds"]]
    if screen["sample_count"] % len(audit_seeds) != 0:
        raise ValueError("screen count is not balanced across audit seeds")

    samples = len(audit_rows)
    certified = sum(row["certified"] for row in audit_rows.values())
    stable = sum(row["grid_identity"] for row in grid_rows.values())
    violations = sum(
        audit_rows[key]["certified"] and not grid_rows[key]["grid_identity"]
        for key in audit_rows
    )
    stable_uncertified = sum(
        not audit_rows[key]["certified"] and grid_rows[key]["grid_identity"]
        for key in audit_rows
    )
    counterexample_uncertified = sum(
        not audit_rows[key]["certified"] and not grid_rows[key]["grid_identity"]
        for key in audit_rows
    )
    confidence = 0.95
    screen_inputs_per_seed = int(
        screen["sample_count"] // len(audit["config"]["seeds"])
    )
    confirmation_keys = [
        key for key, row in audit_rows.items()
        if int(row["audit_position"]) >= screen_inputs_per_seed
    ]
    confirmation_samples = len(confirmation_keys)
    confirmation_certified = sum(
        audit_rows[key]["certified"] for key in confirmation_keys
    )
    confirmation_stable = sum(
        grid_rows[key]["grid_identity"] for key in confirmation_keys
    )
    confirmation_violations = sum(
        audit_rows[key]["certified"] and not grid_rows[key]["grid_identity"]
        for key in confirmation_keys
    )
    confirmation_sobol_violations = sum(
        audit_rows[key]["certified"] and not sobol_rows[key]["sobol_identity"]
        for key in confirmation_keys
    )
    confirmation_joint_identity = sum(
        grid_rows[key]["grid_identity"] and sobol_rows[key]["sobol_identity"]
        for key in confirmation_keys
    )
    confirmation_joint_violations = sum(
        audit_rows[key]["certified"]
        and not (
            grid_rows[key]["grid_identity"] and sobol_rows[key]["sobol_identity"]
        )
        for key in confirmation_keys
    )
    coverage = np.asarray(
        [row["certified_parameter_fraction"] for row in audit_rows.values()],
        dtype=np.float64,
    )
    margins = np.asarray(
        [grid_rows[key]["minimum_reference_margin"] for key in audit_rows],
        dtype=np.float64,
    )
    coverage_margin = spearmanr(coverage, margins)

    per_seed = []
    labels = []
    certified_values = []
    grid_values = []
    for row in audit["per_seed"]:
        seed = int(row["seed"])
        keys = [key for key in audit_rows if key[0] == seed]
        seed_samples = len(keys)
        seed_certified = sum(audit_rows[key]["certified"] for key in keys)
        seed_stable = sum(grid_rows[key]["grid_identity"] for key in keys)
        labels.append(str(seed))
        seed_confirmation_keys = [
            key
            for key in keys
            if int(audit_rows[key]["audit_position"]) >= screen_inputs_per_seed
        ]
        seed_confirmation_certified = sum(
            audit_rows[key]["certified"] for key in seed_confirmation_keys
        )
        seed_confirmation_stable = sum(
            grid_rows[key]["grid_identity"] for key in seed_confirmation_keys
        )
        seed_confirmation_joint_identity = sum(
            grid_rows[key]["grid_identity"] and sobol_rows[key]["sobol_identity"]
            for key in seed_confirmation_keys
        )
        certified_values.append(
            seed_confirmation_certified / len(seed_confirmation_keys)
        )
        grid_values.append(
            seed_confirmation_joint_identity / len(seed_confirmation_keys)
        )
        per_seed.append(
            {
                **row,
                "certified_input_fraction_exact_95_percent_interval": (
                    _exact_interval(seed_certified, seed_samples, confidence)
                ),
                "grid_identity_count": seed_stable,
                "grid_identity_fraction": seed_stable / seed_samples,
                "certified_grid_violation_count": sum(
                    audit_rows[key]["certified"]
                    and not grid_rows[key]["grid_identity"]
                    for key in keys
                ),
                "stable_but_uncertified_count": sum(
                    not audit_rows[key]["certified"]
                    and grid_rows[key]["grid_identity"]
                    for key in keys
                ),
                "counterexample_uncertified_count": sum(
                    not grid_rows[key]["grid_identity"] for key in keys
                ),
                "confirmation_excluding_screen": {
                    "sample_count": len(seed_confirmation_keys),
                    "certified_input_count": seed_confirmation_certified,
                    "certified_input_fraction": (
                        seed_confirmation_certified / len(seed_confirmation_keys)
                    ),
                    "certified_input_fraction_exact_95_percent_interval": (
                        _exact_interval(
                            seed_confirmation_certified,
                            len(seed_confirmation_keys),
                            confidence,
                        )
                    ),
                    "grid_identity_count": seed_confirmation_stable,
                    "grid_identity_fraction": (
                        seed_confirmation_stable / len(seed_confirmation_keys)
                    ),
                    "joint_grid_sobol_identity_count": (
                        seed_confirmation_joint_identity
                    ),
                    "joint_grid_sobol_identity_fraction": (
                        seed_confirmation_joint_identity
                        / len(seed_confirmation_keys)
                    ),
                },
            }
        )

    class_rows = []
    predictions = sorted(
        {int(row["reference_prediction"]) for row in grid_rows.values()}
    )
    for prediction in predictions:
        keys = [
            key
            for key, row in grid_rows.items()
            if int(row["reference_prediction"]) == prediction
        ]
        class_rows.append(
            {
                "reference_prediction": prediction,
                "sample_count": len(keys),
                "certified_input_fraction": float(
                    np.mean([audit_rows[key]["certified"] for key in keys])
                ),
                "grid_identity_fraction": float(
                    np.mean([grid_rows[key]["grid_identity"] for key in keys])
                ),
                "mean_certified_parameter_fraction": float(
                    np.mean(
                        [
                            audit_rows[key]["certified_parameter_fraction"]
                            for key in keys
                        ]
                    )
                ),
            }
        )

    summary = {
        "schema_version": "SHDHybridFamilyFullAuditSummary/v1",
        "status": (
            "frozen five-seed all-input audit plus independent grid and Sobol "
            "falsification diagnostics"
        ),
        "scope": {
            "task": "SHD recurrent SNN",
            "target_reset": "reset-to-value",
            "integration": "forward Euler",
            "relative_timestep_bounds": [-0.01, 0.01],
            "relative_threshold_scale_bounds": [-0.01, 0.01],
            "selection": "all 861 frozen certificate-audit inputs per seed",
        },
        "sample_count": samples,
        "seed_count": len(per_seed),
        "certified_input_count": certified,
        "certified_input_fraction": certified / samples,
        "certified_input_fraction_exact_95_percent_interval": _exact_interval(
            certified, samples, confidence
        ),
        "primary_confirmation_excluding_screen": {
            "exclusion_rule": (
                f"audit positions 0 through {screen_inputs_per_seed - 1} "
                "per seed were used by the advancement screen"
            ),
            "sample_count": confirmation_samples,
            "certified_input_count": confirmation_certified,
            "certified_input_fraction": (
                confirmation_certified / confirmation_samples
            ),
            "certified_input_fraction_exact_95_percent_interval": (
                _exact_interval(
                    confirmation_certified, confirmation_samples, confidence
                )
            ),
            "grid_identity_count": confirmation_stable,
            "grid_identity_fraction": confirmation_stable / confirmation_samples,
            "certified_grid_violation_count": confirmation_violations,
            "certified_sobol_violation_count": confirmation_sobol_violations,
            "joint_grid_sobol_identity_count": confirmation_joint_identity,
            "joint_grid_sobol_identity_fraction": (
                confirmation_joint_identity / confirmation_samples
            ),
            "certified_joint_falsification_violation_count": (
                confirmation_joint_violations
            ),
            "passes_20_percent_gate": (
                confirmation_certified / confirmation_samples >= 0.2
                and confirmation_joint_violations == 0
            ),
        },
        "mean_certified_parameter_fraction": float(np.mean(coverage)),
        "median_certified_parameter_fraction": float(np.median(coverage)),
        "certified_parameter_fraction_quantiles": {
            "q10": float(np.quantile(coverage, 0.1)),
            "q25": float(np.quantile(coverage, 0.25)),
            "median": float(np.quantile(coverage, 0.5)),
            "q75": float(np.quantile(coverage, 0.75)),
            "q90": float(np.quantile(coverage, 0.9)),
        },
        "per_seed": per_seed,
        "per_reference_prediction": class_rows,
        "full_audit_gate_passed": bool(audit["full_audit_gate_passed"]),
        "area_cover_valid": bool(audit["area_cover_valid"]),
        "finite_grid_falsification": {
            "resolution_per_axis": int(grid["grid_resolution_per_axis"]),
            "points_per_input": int(grid["grid_point_count"]),
            "grid_identity_count": stable,
            "grid_identity_fraction": stable / samples,
            "grid_counterexample_input_count": samples - stable,
            "certified_grid_violation_count": violations,
            "stable_but_uncertified_count": stable_uncertified,
            "counterexample_uncertified_count": counterexample_uncertified,
            "certified_fraction_of_grid_stable_inputs": certified / stable,
            "coverage_minimum_grid_margin_spearman": {
                "correlation": float(coverage_margin.statistic),
                "p_value": float(coverage_margin.pvalue),
            },
        },
        "sobol_falsification": {
            "points_per_input": int(sobol["points_per_input"]),
            "sobol_seed": int(sobol["sobol_seed"]),
            "sobol_identity_count": int(sobol["sobol_identity_count"]),
            "sobol_identity_fraction": float(sobol["sobol_identity_fraction"]),
            "certified_sobol_violation_count": int(
                sobol["certified_sobol_violation_count"]
            ),
            "confirmation_certified_sobol_violation_count": (
                confirmation_sobol_violations
            ),
        },
        "screen_comparison": {
            "screen_sample_count": int(screen["sample_count"]),
            "screen_certified_input_fraction": float(
                screen["certified_input_fraction"]
            ),
            "full_minus_screen_certified_fraction": (
                certified / samples - float(screen["certified_input_fraction"])
            ),
        },
        "runtime": {
            "wall_seconds_this_invocation": float(
                audit["wall_seconds_this_invocation"]
            ),
            "sum_shard_seconds": float(audit["sum_shard_seconds"]),
            "median_seconds_per_input": float(
                np.median([row["seconds"] for row in audit_rows.values()])
            ),
            "grid_seconds": float(grid["seconds"]),
            "sobol_seconds": float(sobol["seconds"]),
        },
        "source_report_hashes": {
            str(audit_path.relative_to(root)).replace("\\", "/"): sha256_file(
                audit_path
            ),
            str(grid_path.relative_to(root)).replace("\\", "/"): sha256_file(
                grid_path
            ),
            str(sobol_path.relative_to(root)).replace("\\", "/"): sha256_file(
                sobol_path
            ),
            str(screen_path.relative_to(root)).replace("\\", "/"): sha256_file(
                screen_path
            ),
        },
        "source_code_revisions": {
            "certificate_audit": audit["code_revision"],
            "certificate_shards": sorted(shard_revisions),
            "grid_validation": grid["code_revision"],
            "sobol_validation": sobol["code_revision"],
        },
        "code_revision": code_revision(root),
        "route_assessment": {
            "claim_shd_population_certificate_fraction": (
                confirmation_certified / confirmation_samples >= 0.2
                and confirmation_joint_violations == 0
            ),
            "claim_all_execution_semantics_axes": False,
            "claim_second_event_task": False,
            "claim_physical_certificate": False,
        },
        "interpretation": (
            "Every certified input is proved invariant across the full joint "
            "continuous box. The grid and Sobol designs can falsify but cannot "
            "strengthen that proof. Stable-but-uncertified inputs measure analyzer "
            "or fixed-budget slack; sampled counterexamples measure genuine observed "
            "semantic instability."
        ),
    }
    write_json_immutable(output_path, summary)

    labels.append("all")
    certified_values.append(confirmation_certified / confirmation_samples)
    grid_values.append(confirmation_joint_identity / confirmation_samples)
    x = np.arange(len(labels))
    width = 0.36
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(7.2, 3.4))
    axis.bar(
        x - width / 2,
        np.asarray(certified_values) * 100.0,
        width,
        label="Sound full-box certificate",
        color="#2a9d8f",
    )
    axis.bar(
        x + width / 2,
        np.asarray(grid_values) * 100.0,
        width,
        label="Grid+Sobol sampled identity ceiling",
        color="#8e7dbe",
    )
    axis.axhline(20.0, color="#b23a48", linestyle="--", linewidth=1.2)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Audit inputs (%)")
    axis.set_ylim(0.0, 100.0)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    axis.set_title("Confirmatory SHD joint timestep/threshold family audit")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
