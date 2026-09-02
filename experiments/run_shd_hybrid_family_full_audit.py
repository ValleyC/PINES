from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from pines.abstract import SemanticsBox
from pines.affine import AdaptiveHybridPolygonCertifier
from pines.artifacts import (
    array_description,
    code_revision,
    file_reference,
    write_json,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.models import DenseRecurrentSNN


def _box_and_certifier(config: dict[str, Any]):
    semantics = primary_semantic_conditions()
    reference = semantics["reference"]
    target = semantics[config["condition"]]
    box = SemanticsBox(
        base=target,
        timestep_bounds=(
            reference.timestep
            * (1.0 - float(config["relative_timestep_radius"])),
            reference.timestep
            * (1.0 + float(config["relative_timestep_radius"])),
        ),
        threshold_scale_bounds=(
            1.0 - float(config["relative_threshold_radius"]),
            1.0 + float(config["relative_threshold_radius"]),
        ),
        integration_rules=(target.integration_rule,),
        threshold_timings=(target.threshold_timing,),
        reset_rules=(target.reset_rule,),
        synaptic_delays=(target.synaptic_delay_steps,),
        output_delays=(target.output_delay_steps,),
        name=f"{config['condition']}-hybrid-full-audit-v1",
    )
    budgets = config["certificate_budgets"]
    certifier = AdaptiveHybridPolygonCertifier(
        max_branches=int(budgets["maximum_local_branches"]),
        max_guard_band_splits=int(budgets["maximum_guard_band_splits"]),
    )
    return reference, target, box, certifier


def _run_shard(task: dict[str, Any]) -> dict[str, Any]:
    from threadpoolctl import threadpool_limits

    root = Path(task["root"])
    config = task["config"]
    seed = int(task["seed"])
    positions = np.asarray(task["positions"], dtype=np.int64)
    seed_dir = root / task["artifact_root"] / f"seed_{seed}"
    model_path = seed_dir / "model.npz"
    split_path = seed_dir / "split_indices.npz"
    model = DenseRecurrentSNN.load(model_path)
    store = PackedSHD(root / task["data_root"] / "train.npz")
    with np.load(split_path, allow_pickle=False) as splits:
        audit_indices = np.asarray(
            splits[config["audit_selection"]["split"]], dtype=np.int64
        )
    selected_indices = audit_indices[positions]
    frames = store.frames(selected_indices)
    reference, target, box, certifier = _box_and_certifier(config)
    budgets = config["certificate_budgets"]
    rows = []
    started = time.perf_counter()
    with threadpool_limits(
        limits=int(config["execution"]["blas_threads_per_worker"]),
        user_api="blas",
    ):
        for position, dataset_index, frame in zip(
            positions, selected_indices, frames, strict=True
        ):
            input_started = time.perf_counter()
            result = certifier.certify(
                model,
                frame[None, ...],
                reference,
                box,
                max_leaves=int(budgets["maximum_polygon_leaves"]),
            )
            rows.append(
                {
                    "seed": seed,
                    "audit_position": int(position),
                    "dataset_index": int(dataset_index),
                    "certified": result.certified,
                    "certified_parameter_fraction": (
                        result.certified_parameter_fraction
                    ),
                    "unresolved_parameter_fraction": (
                        result.unresolved_parameter_fraction
                    ),
                    "analyzed_polygons": result.analyzed_polygons,
                    "final_leaves": result.final_leaves,
                    "branch_certified_leaves": result.branch_certified_leaves,
                    "affine_certified_leaves": result.affine_certified_leaves,
                    "unresolved_leaves": result.unresolved_leaves,
                    "branch_attempts": result.branch_attempts,
                    "branch_cap_hits": result.branch_cap_hits,
                    "branch_prediction_rejections": (
                        result.branch_prediction_rejections
                    ),
                    "maximum_completed_branches": (
                        result.maximum_completed_branches
                    ),
                    "guard_band_splits": result.guard_band_splits,
                    "axis_fallback_splits": result.axis_fallback_splits,
                    "seconds": time.perf_counter() - input_started,
                }
            )
    return {
        "schema_version": "SHDHybridFamilyFullAuditShard/v1",
        "config_reference": task["config_reference"],
        "seed": seed,
        "shard_index": int(task["shard_index"]),
        "shard_count": int(task["shard_count"]),
        "audit_positions": positions,
        "audit_positions_reference": array_description(positions),
        "selected_indices": selected_indices,
        "selected_indices_reference": array_description(selected_indices),
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "model_description": model.model_description,
        "model_file": file_reference(model_path),
        "split_indices_file": file_reference(split_path),
        "train_store_reference": store.data_description,
        "box_description": box.box_description,
        "reference_semantics": reference.semantics_description,
        "target_semantics": target.semantics_description,
        "code_revision": code_revision(root),
    }


def _validated_shard(path: Path, task: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        shard = json.load(handle)
    expected_positions = np.asarray(task["positions"], dtype=np.int64)
    if (
        shard.get("schema_version") != "SHDHybridFamilyFullAuditShard/v1"
        or shard.get("config_reference") != task["config_reference"]
        or int(shard.get("seed", -1)) != int(task["seed"])
        or int(shard.get("shard_index", -1)) != int(task["shard_index"])
        or shard.get("audit_positions_reference") != array_description(expected_positions)
        or len(shard.get("rows", [])) != len(expected_positions)
    ):
        raise ValueError(f"existing shard failed validation: {path}")
    return shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/shd_hybrid_full_audit_v1.json"
    )
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root", default="artifacts/shd_v59_hybrid_full_audit_v1"
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        help=(
            "Override process parallelism without changing the frozen scientific "
            "configuration. This affects scheduling only."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "SHDHybridFamilyFullAudit/v1":
        raise ValueError("unsupported full hybrid audit configuration")
    if not config["audit_selection"].get("use_all_available"):
        raise ValueError("full audit configuration must select all audit inputs")
    config_reference = file_reference(config_path)
    output_dir = root / args.output_root
    output_path = output_dir / "hybrid_family_full_audit.json"
    if output_path.exists():
        raise FileExistsError(f"full audit output exists: {output_path}")
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_count = int(config["execution"]["shards_per_seed"])
    tasks = []
    seed_provenance = {}
    for seed in config["seeds"]:
        seed_dir = root / args.artifact_root / f"seed_{seed}"
        split_path = seed_dir / "split_indices.npz"
        with np.load(split_path, allow_pickle=False) as splits:
            audit_indices = np.asarray(
                splits[config["audit_selection"]["split"]], dtype=np.int64
            )
        seed_provenance[str(seed)] = {
            "sample_count": len(audit_indices),
            "selected_indices_reference": array_description(audit_indices),
            "split_indices_file": file_reference(split_path),
        }
        for shard_index in range(shard_count):
            positions = np.arange(shard_index, len(audit_indices), shard_count)
            tasks.append(
                {
                    "root": str(root),
                    "config": config,
                    "config_reference": config_reference,
                    "data_root": args.data_root,
                    "artifact_root": args.artifact_root,
                    "seed": seed,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "positions": positions.tolist(),
                }
            )

    shards = []
    pending = []
    for task in tasks:
        path = shard_dir / (
            f"seed_{task['seed']}_shard_{task['shard_index']:02d}.json"
        )
        shard = _validated_shard(path, task)
        if shard is None:
            pending.append((task, path))
        else:
            shards.append(shard)
    print(
        f"full audit shards complete={len(shards)} pending={len(pending)}",
        flush=True,
    )

    run_started = time.perf_counter()
    worker_count_this_invocation = 0
    if pending:
        requested_worker_count = (
            int(args.worker_count)
            if args.worker_count is not None
            else int(config["execution"]["worker_count"])
        )
        if requested_worker_count < 1:
            raise ValueError("worker count must be positive")
        worker_count = min(requested_worker_count, len(pending))
        worker_count_this_invocation = worker_count
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_run_shard, task): (task, path)
                for task, path in pending
            }
            for future in as_completed(future_map):
                task, path = future_map[future]
                shard = future.result()
                write_json(path, shard)
                shards.append(shard)
                print(
                    f"full audit shard seed={task['seed']} "
                    f"index={task['shard_index']} inputs={len(shard['rows'])} "
                    f"certified={sum(row['certified'] for row in shard['rows'])} "
                    f"seconds={shard['seconds']:.1f} "
                    f"complete={len(shards)}/{len(tasks)}",
                    flush=True,
                )

    rows = sorted(
        [row for shard in shards for row in shard["rows"]],
        key=lambda row: (row["seed"], row["audit_position"]),
    )
    expected_count = sum(
        int(value["sample_count"]) for value in seed_provenance.values()
    )
    if len(rows) != expected_count or len(
        {(row["seed"], row["audit_position"]) for row in rows}
    ) != expected_count:
        raise AssertionError("full audit shards do not form a unique complete cover")

    per_seed = []
    for seed in config["seeds"]:
        seed_rows = [row for row in rows if row["seed"] == seed]
        per_seed.append(
            {
                "seed": seed,
                "sample_count": len(seed_rows),
                "certified_input_count": sum(row["certified"] for row in seed_rows),
                "certified_input_fraction": float(
                    np.mean([row["certified"] for row in seed_rows])
                ),
                "mean_certified_parameter_fraction": float(
                    np.mean(
                        [row["certified_parameter_fraction"] for row in seed_rows]
                    )
                ),
                "median_seconds": float(
                    np.median([row["seconds"] for row in seed_rows])
                ),
            }
        )
    certified_fraction = float(np.mean([row["certified"] for row in rows]))
    gate = config["full_audit_gate"]
    area_cover_valid = all(
        np.isclose(
            row["certified_parameter_fraction"]
            + row["unresolved_parameter_fraction"],
            1.0,
        )
        for row in rows
    )
    gate_passed = (
        certified_fraction >= float(gate["minimum_certified_input_fraction"])
        and (
            not gate["require_every_seed_nonzero"]
            or all(row["certified_input_count"] > 0 for row in per_seed)
        )
        and (not gate["require_zero_area_cover_failures"] or area_cover_valid)
    )
    reference, target, box, _ = _box_and_certifier(config)
    store = PackedSHD(root / args.data_root / "train.npz")
    report = {
        "schema_version": "SHDHybridFamilyFullAuditResult/v1",
        "status": (
            "frozen five-seed full certificate audit over every available audit "
            "input; only zero unresolved area is certified"
        ),
        "config": config,
        "config_reference": config_reference,
        "condition": config["condition"],
        "box_description": box.box_description,
        "sample_count": len(rows),
        "certified_input_count": sum(row["certified"] for row in rows),
        "certified_input_fraction": certified_fraction,
        "mean_certified_parameter_fraction": float(
            np.mean([row["certified_parameter_fraction"] for row in rows])
        ),
        "area_cover_valid": area_cover_valid,
        "full_audit_gate_passed": gate_passed,
        "per_seed": per_seed,
        "rows": rows,
        "wall_seconds_this_invocation": time.perf_counter() - run_started,
        "worker_count_this_invocation": worker_count_this_invocation,
        "sum_shard_seconds": float(sum(shard["seconds"] for shard in shards)),
        "shards": [
            {
                "path": str(
                    (
                        shard_dir
                        / f"seed_{shard['seed']}_shard_{shard['shard_index']:02d}.json"
                    ).relative_to(root)
                ).replace("\\", "/"),
                "reference": file_reference(
                    shard_dir
                    / f"seed_{shard['seed']}_shard_{shard['shard_index']:02d}.json"
                ),
                "seed": shard["seed"],
                "shard_index": shard["shard_index"],
                "sample_count": len(shard["rows"]),
            }
            for shard in sorted(
                shards, key=lambda value: (value["seed"], value["shard_index"])
            )
        ],
        "seed_provenance": seed_provenance,
        "train_store_reference": store.data_description,
        "reference_semantics": reference.semantics_description,
        "target_semantics": target.semantics_description,
        "code_revision": code_revision(root),
        "interpretation": (
            "This is the declared population-level tractability audit for the "
            "frozen SHD certificate split and reset/timestep/threshold contract. "
            "It measures the fraction of inputs for which the full continuous "
            "semantics box is soundly proved prediction invariant under fixed "
            "resource budgets. It does not cover other semantic axes, other tasks, "
            "or physical hardware conformance."
        ),
    }
    write_json(output_path, report)


if __name__ == "__main__":
    main()
