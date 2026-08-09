from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pines.artifacts import code_revision, sha256_file, write_json_immutable


AXES = (
    ("integration_rule", ("forward_euler", "exponential_euler")),
    ("threshold_timing", ("post_integration", "pre_integration")),
    ("reset_rule", ("subtractive", "to_value")),
    ("synaptic_delay_steps", (0, 1)),
)
AXIS_VALUE_LABELS = {
    "forward_euler": "forward Euler",
    "exponential_euler": "exponential Euler",
    "post_integration": "post",
    "pre_integration": "pre",
    "subtractive": "subtractive",
    "to_value": "reset-to-value",
    0: "0-step",
    1: "1-step",
}


def _member_label(semantics: dict) -> str:
    integration = "FE" if semantics["integration_rule"] == "forward_euler" else "EXP"
    timing = "post" if semantics["threshold_timing"] == "post_integration" else "pre"
    reset = "sub" if semantics["reset_rule"] == "subtractive" else "value"
    delay = f"d{int(semantics['synaptic_delay_steps'])}"
    return f"{integration}, {timing}, {reset}, {delay}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="artifacts/shd_v73_cartesian_member_scaling_v1/cartesian_member_scaling.json",
    )
    parser.add_argument(
        "--output",
        default="results/shd_v1/cartesian_member_scaling_summary.json",
    )
    parser.add_argument(
        "--rows", default="results/shd_v1/cartesian_member_scaling_rows.csv"
    )
    parser.add_argument(
        "--figure", default="paper/figures/shd_cartesian_member_scaling.pdf"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    report_path = root / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "SHDHybridCartesianMemberScalingResult/v1":
        raise ValueError("unsupported Cartesian member-scaling report")
    members = sorted(report["member_rows"], key=lambda row: int(row["member_index"]))
    if len(members) != int(report["member_count"]):
        raise ValueError("member count mismatch")
    if len({row["semantics_hash"] for row in members}) != len(members):
        raise ValueError("duplicate semantics member")
    budgets = tuple(
        int(row["maximum_polygon_leaves"]) for row in members[0]["budget_rows"]
    )
    if any(
        tuple(int(row["maximum_polygon_leaves"]) for row in member["budget_rows"])
        != budgets
        for member in members
    ):
        raise ValueError("members do not share one budget schedule")

    flat_rows = []
    coverage = np.empty((len(members), len(budgets)), dtype=np.float64)
    runtime = np.empty_like(coverage)
    for member_position, member in enumerate(members):
        semantics = member["semantics"]
        for budget_position, budget_row in enumerate(member["budget_rows"]):
            fraction = float(budget_row["certified_parameter_fraction"])
            unresolved = float(budget_row["unresolved_parameter_fraction"])
            if not np.isclose(fraction + unresolved, 1.0, atol=5e-12, rtol=0.0):
                raise ValueError("member parameter-area accounting does not close")
            coverage[member_position, budget_position] = fraction
            runtime[member_position, budget_position] = float(budget_row["seconds"])
            flat_rows.append(
                {
                    "member_index": int(member["member_index"]),
                    "member_label": _member_label(semantics),
                    "semantics_hash": member["semantics_hash"],
                    "integration_rule": semantics["integration_rule"],
                    "threshold_timing": semantics["threshold_timing"],
                    "reset_rule": semantics["reset_rule"],
                    "synaptic_delay_steps": int(semantics["synaptic_delay_steps"]),
                    "maximum_polygon_leaves": int(
                        budget_row["maximum_polygon_leaves"]
                    ),
                    "certified": bool(budget_row["certified"]),
                    "certified_parameter_fraction": fraction,
                    "unresolved_parameter_fraction": unresolved,
                    "branch_certified_leaves": int(
                        budget_row["branch_certified_leaves"]
                    ),
                    "affine_certified_leaves": int(
                        budget_row["affine_certified_leaves"]
                    ),
                    "unresolved_leaves": int(budget_row["unresolved_leaves"]),
                    "branch_cap_hits": int(budget_row["branch_cap_hits"]),
                    "branch_prediction_rejections": int(
                        budget_row["branch_prediction_rejections"]
                    ),
                    "seconds": float(budget_row["seconds"]),
                }
            )

    output_path = root / args.output
    rows_path = root / args.rows
    figure_path = root / args.figure
    for path in (output_path, rows_path, figure_path, figure_path.with_suffix(".png")):
        if path.exists():
            raise FileExistsError(f"Cartesian aggregate destination exists: {path}")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    budget_rows = []
    axis_effects = {}
    for budget_position, budget in enumerate(budgets):
        values = coverage[:, budget_position]
        budget_rows.append(
            {
                "maximum_polygon_leaves": budget,
                "mean_certified_parameter_fraction": float(np.mean(values)),
                "median_certified_parameter_fraction": float(np.median(values)),
                "minimum_certified_parameter_fraction": float(np.min(values)),
                "maximum_certified_parameter_fraction": float(np.max(values)),
                "fully_certified_member_count": int(
                    np.count_nonzero(np.asarray(
                        [member["budget_rows"][budget_position]["certified"] for member in members]
                    ))
                ),
                "mean_seconds": float(np.mean(runtime[:, budget_position])),
                "maximum_seconds": float(np.max(runtime[:, budget_position])),
                "full_family_certified": bool(
                    all(member["budget_rows"][budget_position]["certified"] for member in members)
                ),
            }
        )
        effects_for_budget = {}
        for field, declared_values in AXES:
            field_rows = {}
            for declared_value in declared_values:
                mask = np.asarray(
                    [member["semantics"][field] == declared_value for member in members]
                )
                if np.count_nonzero(mask) != len(members) // 2:
                    raise ValueError(f"Cartesian axis {field} is not balanced")
                field_rows[str(declared_value)] = {
                    "member_count": int(np.count_nonzero(mask)),
                    "mean_certified_parameter_fraction": float(np.mean(values[mask])),
                    "minimum_certified_parameter_fraction": float(np.min(values[mask])),
                }
            first, second = declared_values
            field_rows["second_minus_first_mean"] = float(
                field_rows[str(second)]["mean_certified_parameter_fraction"]
                - field_rows[str(first)]["mean_certified_parameter_fraction"]
            )
            effects_for_budget[field] = field_rows
        axis_effects[str(budget)] = effects_for_budget

    attribution_position = (
        len(budgets) - 2
        if len(budgets) > 1 and np.allclose(coverage[:, -1], 1.0)
        else len(budgets) - 1
    )
    attribution_budget = budgets[attribution_position]
    final_order = sorted(
        range(len(members)),
        key=lambda position: (
            coverage[position, attribution_position],
            -runtime[position, attribution_position],
            int(members[position]["member_index"]),
        ),
    )
    bottlenecks = [
        {
            "rank": rank + 1,
            "member_index": int(members[position]["member_index"]),
            "member_label": _member_label(members[position]["semantics"]),
            "certified_parameter_fraction": float(
                coverage[position, attribution_position]
            ),
            "seconds": float(runtime[position, attribution_position]),
        }
        for rank, position in enumerate(final_order)
    ]
    summary = {
        "schema_version": "SHDHybridCartesianMemberScalingAggregate/v2",
        "status": (
            "development-only sound member-wise resource attribution on one "
            "label-free selected input; not an audit-population result"
        ),
        "source_report": args.report.replace("\\", "/"),
        "source_report_hash": sha256_file(report_path),
        "seed": int(report["seed"]),
        "dataset_index": int(report["dataset_index"]),
        "member_count": len(members),
        "budgets": list(budgets),
        "budget_rows": budget_rows,
        "axis_main_effects": axis_effects,
        "attribution_budget": attribution_budget,
        "bottleneck_members_at_attribution_budget": bottlenecks,
        "affine_layer_certified_any_leaf": any(
            int(row["affine_certified_leaves"]) > 0 for row in flat_rows
        ),
        "branch_layer_certified_any_leaf": any(
            int(row["branch_certified_leaves"]) > 0 for row in flat_rows
        ),
        "rows_csv": args.rows.replace("\\", "/"),
        "rows_csv_hash": sha256_file(rows_path),
        "code_revision": code_revision(root),
        "interpretation": (
            "Member-wise coverage separates mutually exclusive execution semantics. "
            "Main effects identify which declared axes consume proof budget; only a "
            "certificate for every member establishes the complete Cartesian family."
        ),
    }
    write_json_immutable(output_path, summary)

    labels = [_member_label(member["semantics"]) for member in members]
    fig, (heat_axis, effect_axis) = plt.subplots(
        1, 2, figsize=(10.4, 6.2), gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )
    image = heat_axis.imshow(100 * coverage, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    heat_axis.set_xticks(np.arange(len(budgets)), [str(value) for value in budgets])
    heat_axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    heat_axis.set_xlabel("Maximum polygon leaves")
    heat_axis.set_title("Sound coverage by discrete member")
    for row_index in range(len(members)):
        for column_index in range(len(budgets)):
            value = 100 * coverage[row_index, column_index]
            heat_axis.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value < 55 else "black",
            )
    fig.colorbar(image, ax=heat_axis, label="Certified parameter area (%)", shrink=0.8)

    final_effects = axis_effects[str(attribution_budget)]
    y = np.arange(len(AXES), dtype=np.float64)
    for position, (field, declared_values) in enumerate(AXES):
        first, second = declared_values
        first_mean = 100 * final_effects[field][str(first)][
            "mean_certified_parameter_fraction"
        ]
        second_mean = 100 * final_effects[field][str(second)][
            "mean_certified_parameter_fraction"
        ]
        effect_axis.plot([first_mean, second_mean], [position, position], color="0.65")
        effect_axis.scatter(first_mean, position, color="#1b9e77", marker="o", s=45)
        effect_axis.scatter(second_mean, position, color="#d95f02", marker="s", s=45)
        effect_axis.text(
            first_mean,
            position + 0.2,
            AXIS_VALUE_LABELS[first],
            ha="center",
            fontsize=7,
        )
        effect_axis.text(
            second_mean,
            position - 0.2,
            AXIS_VALUE_LABELS[second],
            ha="center",
            fontsize=7,
        )
    effect_axis.set_yticks(y, [field.replace("_", " ") for field, _ in AXES])
    effect_axis.set_ylim(len(AXES) - 0.6, -0.6)
    effect_axis.set_xlim(-5, 110)
    effect_axis.set_xlabel(f"Mean coverage at {attribution_budget} leaves (%)")
    effect_axis.set_title(f"Axis attribution at {attribution_budget} leaves")
    effect_axis.grid(axis="x", alpha=0.2)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "largest_budget": budget_rows[-1],
        "hardest_member": bottlenecks[0],
        "affine_layer_certified_any_leaf": summary["affine_layer_certified_any_leaf"],
    }, indent=2))


if __name__ == "__main__":
    main()
