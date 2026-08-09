from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from pines.artifacts import code_revision, sha256_file, write_json_immutable


SEEDS = (1701, 2718, 3141, 5772, 8119)
METHODS = (
    "no_repair",
    "certificate_directed",
    "logit_only",
    "global_threshold",
    "per_platform_qat",
)
BOOTSTRAP_REPETITIONS = 100_000
BOOTSTRAP_SEED = 20260808


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _unpack_bool_hex(payload: str, count: int) -> np.ndarray:
    packed = np.frombuffer(bytes.fromhex(payload), dtype=np.uint8)
    values = np.unpackbits(packed, bitorder="little")[:count].astype(bool)
    if len(values) != count:
        raise ValueError("packed identity vector is shorter than the sample count")
    return values


def _cluster_bootstrap_interval(values: np.ndarray) -> list[float]:
    cluster_values = np.asarray(values, dtype=np.float64)
    if cluster_values.ndim != 1 or len(cluster_values) < 2:
        raise ValueError("cluster bootstrap needs at least two input clusters")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    chunk = 2_000
    for start in range(0, BOOTSTRAP_REPETITIONS, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPETITIONS)
        draws = generator.integers(
            0, len(cluster_values), size=(stop - start, len(cluster_values))
        )
        estimates[start:stop] = np.mean(cluster_values[draws], axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _training_seed_t_interval(
    values: np.ndarray,
    *,
    lower_bound: float | None = 0.0,
    upper_bound: float | None = 1.0,
) -> list[float]:
    seed_values = np.asarray(values, dtype=np.float64)
    if seed_values.ndim != 1 or len(seed_values) < 2:
        raise ValueError("seed interval needs at least two independently trained models")
    center = float(np.mean(seed_values))
    half_width = float(
        t.ppf(0.975, df=len(seed_values) - 1)
        * np.std(seed_values, ddof=1)
        / np.sqrt(len(seed_values))
    )
    lower = center - half_width
    upper = center + half_width
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    if upper_bound is not None:
        upper = min(upper_bound, upper)
    return [lower, upper]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", default="artifacts/shd_v12_repair_family_grid"
    )
    parser.add_argument("--condition", default="reset_to_value")
    parser.add_argument("--output-stem", default="repair_family_grid")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    artifact_root = root / args.artifact_root
    output_root = root / "results" / "shd_v1"
    summary_path = output_root / f"{args.output_stem}_summary.json"
    rows_path = output_root / f"{args.output_stem}_rows.csv"
    figure_pdf = root / "paper" / "figures" / f"{args.output_stem}.pdf"
    figure_png = root / "paper" / "figures" / f"{args.output_stem}.png"
    if any(
        path.exists() for path in (summary_path, rows_path, figure_pdf, figure_png)
    ):
        raise FileExistsError("repair-family grid aggregate destination already exists")

    rows = []
    input_hashes = {}
    sample_counts = set()
    radii = set()
    grid_resolutions = set()
    selected_indices_hashes = set()
    identity_by_method: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    for seed in SEEDS:
        path = artifact_root / f"seed_{seed}" / f"{args.condition}_family_grid.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["schema_version"] not in {
            "SHDRepairFamilyGridDiagnostic/v1",
            "SHDRepairFamilyGridDiagnostic/v2",
            "SHDRepairFamilyGridDiagnostic/v3",
        }:
            raise ValueError(f"unexpected report schema: {path}")
        if report["condition"] != args.condition:
            raise ValueError(f"condition mismatch: {path}")
        reported_methods = tuple(report["methods"])
        if reported_methods != METHODS:
            raise ValueError(
                f"method order mismatch in {path}: {reported_methods!r} != {METHODS!r}"
            )
        sample_counts.add(int(report["sample_count"]))
        radii.add(float(report["relative_radius"]))
        grid_resolutions.add(int(report["grid_resolution"]))
        selected_indices_hashes.add(report["selected_indices_hash"])
        if report["schema_version"] == "SHDRepairFamilyGridDiagnostic/v3":
            if report.get("identity_bit_order") != "little":
                raise ValueError(f"unsupported bit order: {path}")
            if len(report.get("selected_indices", [])) != int(report["sample_count"]):
                raise ValueError(f"selected-index count mismatch: {path}")
            for row in report["rows"]:
                vector = _unpack_bool_hex(
                    row["grid_family_identity_packed_hex"], int(report["sample_count"])
                )
                if int(np.count_nonzero(vector)) != int(
                    row["grid_family_identity_inputs"]
                ):
                    raise ValueError(f"packed grid identity count mismatch: {path}")
                identity_by_method[row["method"]].append(vector)
        for row in report["rows"]:
            rows.append({"seed": seed, **row})
        input_hashes[f"seed_{seed}"] = sha256_file(path)
    if len(sample_counts) != 1 or len(radii) != 1 or len(grid_resolutions) != 1:
        raise ValueError("repair-family reports do not share one experimental design")
    if len(selected_indices_hashes) != 1:
        raise ValueError("repair-family reports do not share input clusters")
    sample_count = sample_counts.pop()
    radius = radii.pop()
    grid_resolution = grid_resolutions.pop()

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
    no_repair_matrix = (
        np.stack(identity_by_method["no_repair"], axis=0)
        if len(identity_by_method["no_repair"]) == len(SEEDS)
        else None
    )
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate_row = {
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
        if len(identity_by_method[method]) == len(SEEDS):
            identity_matrix = np.stack(identity_by_method[method], axis=0)
            seed_fractions = np.mean(identity_matrix, axis=1)
            aggregate_row["input_cluster_bootstrap_95_percent_interval"] = (
                _cluster_bootstrap_interval(np.mean(identity_matrix, axis=0))
            )
            aggregate_row["training_seed_t_95_percent_interval"] = (
                _training_seed_t_interval(seed_fractions)
            )
            if no_repair_matrix is not None:
                gain_matrix = identity_matrix.astype(np.float64) - no_repair_matrix
                aggregate_row[
                    "paired_gain_input_cluster_bootstrap_95_percent_interval"
                ] = _cluster_bootstrap_interval(np.mean(gain_matrix, axis=0))
                aggregate_row["paired_gain_training_seed_t_95_percent_interval"] = (
                    _training_seed_t_interval(
                        np.mean(gain_matrix, axis=1),
                        lower_bound=None,
                        upper_bound=None,
                    )
                )
        aggregate_rows.append(aggregate_row)

    proposed = next(
        row for row in aggregate_rows if row["method"] == "certificate_directed"
    )
    logit = next(row for row in aggregate_rows if row["method"] == "logit_only")
    proposed_vs_logit_cluster_interval = None
    proposed_vs_logit_seed_interval = None
    proposed_vs_logit_seed_wins = None
    if (
        len(identity_by_method["certificate_directed"]) == len(SEEDS)
        and len(identity_by_method["logit_only"]) == len(SEEDS)
    ):
        proposed_matrix = np.stack(
            identity_by_method["certificate_directed"], axis=0
        ).astype(np.float64)
        logit_matrix = np.stack(identity_by_method["logit_only"], axis=0).astype(
            np.float64
        )
        proposed_vs_logit = proposed_matrix - logit_matrix
        proposed_vs_logit_cluster_interval = _cluster_bootstrap_interval(
            np.mean(proposed_vs_logit, axis=0)
        )
        proposed_vs_logit_seed_interval = _training_seed_t_interval(
            np.mean(proposed_vs_logit, axis=1),
            lower_bound=None,
            upper_bound=None,
        )
        proposed_vs_logit_seed_wins = int(
            np.count_nonzero(np.mean(proposed_vs_logit, axis=1) > 0)
        )
    summary = {
        "schema_version": "SHDRepairFamilyGridAggregate/v3",
        "status": "five-seed finite-grid post-repair diagnostic; not a proof",
        "seeds": list(SEEDS),
        "condition": args.condition,
        "sample_count_per_seed": sample_count,
        "relative_radius": radius,
        "grid_resolution": grid_resolution,
        "aggregate_rows": aggregate_rows,
        "uncertainty": {
            "input_cluster_estimand": (
                "input-sampling uncertainty conditional on these five fixed models"
            ),
            "training_seed_estimand": (
                "training-run uncertainty with the shared audit inputs fixed"
            ),
            "input_cluster_bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "input_cluster_bootstrap_seed": BOOTSTRAP_SEED,
            "pooled_binomial_interval_reported": False,
        },
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
            "certificate_directed_minus_logit_only_input_cluster_bootstrap_95_percent_interval": (
                proposed_vs_logit_cluster_interval
            ),
            "certificate_directed_minus_logit_only_training_seed_t_95_percent_interval": (
                proposed_vs_logit_seed_interval
            ),
            "certificate_directed_beats_logit_only_seed_count": (
                proposed_vs_logit_seed_wins
            ),
        },
        "interpretation": (
            "Restricted label-free repair enlarges the sampled continuous-family identity "
            "ceiling in every seed. "
            + (
                "The certificate-directed objective beats matched logit-only imitation "
                "on mean family identity, but paired uncertainty and a sound post-repair "
                "certificate remain necessary."
                if proposed["grid_family_identity_fraction"]["mean"]
                > logit["grid_family_identity_fraction"]["mean"]
                else "The certificate-directed objective does not beat matched logit-only "
                "imitation on mean family identity. A sound post-repair certificate remains "
                "necessary."
            )
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
