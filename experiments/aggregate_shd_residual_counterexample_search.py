from __future__ import annotations

import json
from pathlib import Path

from pines.artifacts import code_revision, sha256_file, write_json_immutable


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = (
        root
        / "artifacts/shd_v52_residual_counterexample_search_dense/seed_1701/"
        "reset_to_value_residual_search.json"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    summary = {
        "schema_version": "SHDResidualCounterexampleSearchSummary/v1",
        "source_report": str(source_path.relative_to(root)).replace("\\", "/"),
        "source_report_hash": sha256_file(source_path),
        "source_code_revision": source["code_revision"],
        "code_revision": code_revision(root),
        "seed": source["seed"],
        "condition": source["condition"],
        "dataset_index": source["dataset_index"],
        "polygon_count": source["polygon_count"],
        "sampling_design": (
            "every polygon vertex, edge midpoint, vertex centroid, and 32 "
            "deterministic random convex combinations per polygon"
        ),
        "proposed_point_count": source["proposed_point_count"],
        "unique_point_count": source["unique_point_count"],
        "counterexample_count": source["counterexample_count"],
        "reference_prediction": source["reference_prediction"],
        "observed_predictions": source["observed_predictions"],
        "minimum_reference_margin": source["minimum_reference_margin"],
        "margin_quantiles": source["margin_quantiles"],
        "seconds": source["seconds"],
        "device": source["device"],
        "sample_artifact": source["sample_artifact"],
        "sample_artifact_hash": source["sample_artifact_hash"],
        "residual_geometry_hash": source["residual_geometry_hash"],
        "route_assessment": {
            "sampled_counterexample_found": source["counterexample_count"] > 0,
            "claim_certificate_from_search": False,
            "residue_supported_as_abstraction_slack": (
                source["counterexample_count"] == 0
                and source["minimum_reference_margin"] > 0.0
            ),
            "advance_to_more_random_search": False,
            "advance_to_sound_joint_oracle": True,
        },
        "interpretation": (
            "Across 128,460 unique points targeted specifically inside the sound "
            "analyzer's 3,365 unresolved polygons, every execution preserves class "
            "2 and the smallest reference margin is 1.033. This strongly supports "
            "the residue being abstraction slack, but sampled search is only a "
            "falsification attempt and cannot issue a certificate."
        ),
    }
    output_path = root / "results/shd_v1/residual_counterexample_search_summary.json"
    write_json_immutable(output_path, summary)


if __name__ == "__main__":
    main()
