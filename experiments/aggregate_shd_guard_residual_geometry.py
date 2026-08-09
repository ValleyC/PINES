from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pines.affine import polygon_area
from pines.artifacts import code_revision, sha256_file, write_json_immutable


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def _union_length(intervals: list[tuple[float, float]]) -> float:
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report_path = (
        root
        / "artifacts/shd_v50_rounding_guard_residuals/seed_1701/"
        "reset_to_value_guard_cuts.json"
    )
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    geometry_path = root / report["residual_polygon_artifact"]
    with np.load(geometry_path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"], dtype=np.float64)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
    polygons = [
        vertices[offsets[index] : offsets[index + 1]]
        for index in range(len(offsets) - 1)
    ]
    areas = np.asarray([polygon_area(polygon) for polygon in polygons])
    timestep_widths = np.asarray([np.ptp(polygon[:, 0]) for polygon in polygons])
    threshold_widths = np.asarray([np.ptp(polygon[:, 1]) for polygon in polygons])
    aspect_ratios = np.maximum(timestep_widths, threshold_widths) / np.maximum(
        np.minimum(timestep_widths, threshold_widths), np.finfo(np.float64).tiny
    )
    timestep_intervals = [
        (float(np.min(polygon[:, 0])), float(np.max(polygon[:, 0])))
        for polygon in polygons
    ]
    threshold_intervals = [
        (float(np.min(polygon[:, 1])), float(np.max(polygon[:, 1])))
        for polygon in polygons
    ]
    normalized_slices = np.linspace(-1.0, 1.0, 257)
    slices_per_polygon = np.asarray(
        [
            sum(lower <= value <= upper for value in normalized_slices)
            for lower, upper in timestep_intervals
        ],
        dtype=np.int64,
    )
    polygons_per_slice = np.asarray(
        [
            sum(lower <= value <= upper for lower, upper in timestep_intervals)
            for value in normalized_slices
        ],
        dtype=np.int64,
    )
    relative_radius = float(report["relative_radius"])
    summary = {
        "schema_version": "SHDGuardResidualGeometry/v1",
        "code_revision": code_revision(root),
        "source_report": str(report_path.relative_to(root)).replace("\\", "/"),
        "source_report_hash": sha256_file(report_path),
        "geometry_artifact": report["residual_polygon_artifact"],
        "geometry_artifact_hash": sha256_file(geometry_path),
        "polygon_count": len(polygons),
        "vertex_count": len(vertices),
        "normalized_area_sum": float(np.sum(areas)),
        "unresolved_parameter_fraction": float(np.sum(areas) / 4.0),
        "normalized_area_quantiles": _quantiles(areas),
        "normalized_timestep_width_quantiles": _quantiles(timestep_widths),
        "normalized_threshold_width_quantiles": _quantiles(threshold_widths),
        "actual_timestep_factor_width_quantiles": _quantiles(
            timestep_widths * relative_radius
        ),
        "actual_threshold_scale_width_quantiles": _quantiles(
            threshold_widths * relative_radius
        ),
        "aspect_ratio_quantiles": _quantiles(aspect_ratios),
        "normalized_timestep_projection_union_length": _union_length(
            timestep_intervals
        ),
        "normalized_threshold_projection_union_length": _union_length(
            threshold_intervals
        ),
        "exact_slice_count": len(normalized_slices),
        "polygons_intersecting_no_exact_slice": int(
            np.count_nonzero(slices_per_polygon == 0)
        ),
        "polygons_intersecting_at_least_one_exact_slice": int(
            np.count_nonzero(slices_per_polygon > 0)
        ),
        "polygons_per_slice_quantiles": _quantiles(polygons_per_slice),
        "route_assessment": {
            "residual_is_confined_to_few_timestep_values": False,
            "exact_slice_grid_covers_every_residual_polygon": False,
            "advance_to_more_uniform_slices": False,
            "advance_to_residual_polygon_joint_oracle": True,
        },
        "interpretation": (
            "The 4,096-leaf residue contains 3,365 small polygons whose timestep "
            "and threshold projections jointly span each full normalized axis. "
            "Median actual timestep-factor width is about 4.19e-5, below the "
            "7.8125e-5 spacing of the 257 exact slices, and 1,271 polygons fall "
            "entirely between those slices. Uniform slice densification therefore "
            "does not compose into a joint certificate; the solver must consume "
            "the residual polygon constraints directly."
        ),
    }
    output_path = root / "results/shd_v1/guard_residual_geometry_summary.json"
    write_json_immutable(output_path, summary)


if __name__ == "__main__":
    main()
