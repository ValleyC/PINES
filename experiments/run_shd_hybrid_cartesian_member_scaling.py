from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pines.abstract import SemanticsBox
from pines.affine import AdaptiveHybridPolygonCertifier
from pines.artifacts import (
    code_revision,
    sha256_file,
    sha256_json,
    write_json_immutable,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.models import DenseRecurrentSNN
from pines.semantics import (
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)


def _ordering(timing: ThresholdTiming) -> UpdateOrdering:
    if timing is ThresholdTiming.PRE_INTEGRATION:
        return UpdateOrdering.THRESHOLD_RESET_INTEGRATE
    return UpdateOrdering.INTEGRATE_THRESHOLD_RESET


def _run_member(task: dict[str, Any]) -> dict[str, Any]:
    from threadpoolctl import threadpool_limits

    root = Path(task["root"])
    seed = int(task["seed"])
    dataset_index = int(task["dataset_index"])
    model = DenseRecurrentSNN.load(
        root / task["artifact_root"] / f"seed_{seed}" / "model.npz"
    )
    model_path = root / task["artifact_root"] / f"seed_{seed}" / "model.npz"
    store = PackedSHD(root / task["data_root"] / "train.npz")
    frame = store.frames(np.asarray([dataset_index], dtype=np.int64))
    reference = primary_semantic_conditions()["reference"]
    member = replace(
        reference,
        integration_rule=IntegrationRule(task["integration_rule"]),
        threshold_timing=ThresholdTiming(task["threshold_timing"]),
        update_ordering=_ordering(ThresholdTiming(task["threshold_timing"])),
        reset_rule=ResetRule(task["reset_rule"]),
        synaptic_delay_steps=int(task["synaptic_delay"]),
        output_delay_steps=int(task["output_delay"]),
    )
    family = task["family"]
    box = SemanticsBox(
        base=member,
        timestep_bounds=(
            reference.timestep
            * (1.0 - float(family["relative_timestep_radius"])),
            reference.timestep
            * (1.0 + float(family["relative_timestep_radius"])),
        ),
        threshold_scale_bounds=(
            1.0 - float(family["relative_threshold_radius"]),
            1.0 + float(family["relative_threshold_radius"]),
        ),
        integration_rules=(member.integration_rule,),
        threshold_timings=(member.threshold_timing,),
        reset_rules=(member.reset_rule,),
        synaptic_delays=(member.synaptic_delay_steps,),
        output_delays=(member.output_delay_steps,),
        name=f"cartesian-member-{task['member_index']}",
    )
    budgets = task["budgets"]
    certifier = AdaptiveHybridPolygonCertifier(
        max_branches=int(budgets["maximum_local_branches"]),
        max_guard_band_splits=int(budgets["maximum_guard_band_splits"]),
    )
    rows = []
    with threadpool_limits(limits=int(task["blas_threads"]), user_api="blas"):
        for max_leaves in budgets["maximum_polygon_leaves"]:
            started = time.perf_counter()
            result = certifier.certify(
                model,
                frame,
                reference,
                box,
                max_leaves=int(max_leaves),
            )
            rows.append(
                {
                    "maximum_polygon_leaves": int(max_leaves),
                    "certified": result.certified,
                    "certified_parameter_fraction": (
                        result.certified_parameter_fraction
                    ),
                    "unresolved_parameter_fraction": (
                        result.unresolved_parameter_fraction
                    ),
                    "final_leaves": result.final_leaves,
                    "branch_certified_leaves": result.branch_certified_leaves,
                    "affine_certified_leaves": result.affine_certified_leaves,
                    "unresolved_leaves": result.unresolved_leaves,
                    "branch_cap_hits": result.branch_cap_hits,
                    "branch_prediction_rejections": (
                        result.branch_prediction_rejections
                    ),
                    "guard_band_splits": result.guard_band_splits,
                    "axis_fallback_splits": result.axis_fallback_splits,
                    "seconds": time.perf_counter() - started,
                }
            )
    return {
        "member_index": int(task["member_index"]),
        "semantics_hash": member.semantics_hash,
        "semantics": member.to_dict(),
        "box_hash": box.box_hash,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(model_path),
        "train_store_hash": store.data_hash,
        "budget_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/shd_hybrid_cartesian_member_scaling_v1.json",
    )
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root",
        default="artifacts/shd_v73_cartesian_member_scaling_v1",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "SHDHybridCartesianMemberScaling/v1":
        raise ValueError("unsupported Cartesian member-scaling configuration")
    selection_path = root / config["selection"]["report"]
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection["split"] != config["selection"]["split"]:
        raise ValueError("selection split mismatch")
    selected = selection[config["selection"]["field"]]
    if selected is None:
        raise ValueError("selection artifact has no stable input")
    seed = int(selection["seed"])
    dataset_index = int(selected["dataset_index"])
    config_hash = sha256_file(config_path)
    selection_hash = sha256_file(selection_path)
    scientific_source_paths = sorted((root / "src" / "pines").rglob("*.py"))
    scientific_source_paths.append(Path(__file__).resolve())
    scientific_source_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in scientific_source_paths
    }
    scientific_source_manifest_hash = sha256_json(scientific_source_hashes)
    output_dir = root / args.output_root
    output_path = output_dir / "cartesian_member_scaling.json"
    if output_path.exists():
        raise FileExistsError(f"Cartesian member-scaling output exists: {output_path}")
    shard_dir = output_dir / "shards"

    family = config["family"]
    combinations = list(
        itertools.product(
            family["integration_rules"],
            family["threshold_timings"],
            family["reset_rules"],
            family["synaptic_delays"],
            family["output_delays"],
        )
    )
    tasks = []
    for member_index, combination in enumerate(combinations):
        integration, timing, reset, synaptic_delay, output_delay = combination
        tasks.append(
            {
                "root": str(root),
                "data_root": args.data_root,
                "artifact_root": args.artifact_root,
                "seed": seed,
                "dataset_index": dataset_index,
                "member_index": member_index,
                "integration_rule": integration,
                "threshold_timing": timing,
                "reset_rule": reset,
                "synaptic_delay": synaptic_delay,
                "output_delay": output_delay,
                "family": family,
                "budgets": config["certificate_budgets"],
                "blas_threads": config["execution"]["blas_threads_per_worker"],
            }
        )

    started = time.perf_counter()
    member_rows = []
    pending_tasks = []
    for task in tasks:
        shard_path = shard_dir / f"member_{int(task['member_index']):02d}.json"
        if not shard_path.exists():
            pending_tasks.append(task)
            continue
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if shard.get("schema_version") != "SHDHybridCartesianMemberScalingShard/v1":
            raise ValueError(f"unsupported member shard: {shard_path}")
        if shard.get("config_hash") != config_hash:
            raise ValueError(f"member shard config mismatch: {shard_path}")
        if shard.get("selection_report_hash") != selection_hash:
            raise ValueError(f"member shard selection mismatch: {shard_path}")
        if shard.get("scientific_source_manifest_hash") != scientific_source_manifest_hash:
            raise ValueError(f"member shard source mismatch: {shard_path}")
        if int(shard.get("member_index", -1)) != int(task["member_index"]):
            raise ValueError(f"member shard identity mismatch: {shard_path}")
        member_rows.append(shard["result"])
        print(
            f"Cartesian member scaling resume member={int(task['member_index']) + 1}/"
            f"{len(tasks)}",
            flush=True,
        )
    worker_count = min(int(config["execution"]["worker_count"]), len(tasks))
    if pending_tasks:
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(pending_tasks))
        ) as executor:
            future_map = {
                executor.submit(_run_member, task): task for task in pending_tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                row = future.result()
                shard = {
                    "schema_version": "SHDHybridCartesianMemberScalingShard/v1",
                    "config_hash": config_hash,
                    "selection_report_hash": selection_hash,
                    "scientific_source_manifest_hash": scientific_source_manifest_hash,
                    "member_index": int(task["member_index"]),
                    "result": row,
                    "code_revision": code_revision(root),
                }
                shard_path = shard_dir / f"member_{int(task['member_index']):02d}.json"
                write_json_immutable(shard_path, shard)
                member_rows.append(row)
                print(
                    f"Cartesian member scaling complete "
                    f"member={row['member_index'] + 1}/{len(tasks)}",
                    flush=True,
                )
    member_rows.sort(key=lambda row: row["member_index"])
    if len(member_rows) != len(tasks):
        raise ValueError("Cartesian member scaling did not produce every member")
    shard_records = []
    for task in tasks:
        shard_path = shard_dir / f"member_{int(task['member_index']):02d}.json"
        if not shard_path.exists():
            raise ValueError(f"missing immutable member shard: {shard_path}")
        shard_records.append(
            {
                "member_index": int(task["member_index"]),
                "path": str(shard_path.relative_to(root)).replace("\\", "/"),
                "hash": sha256_file(shard_path),
            }
        )

    report = {
        "schema_version": "SHDHybridCartesianMemberScalingResult/v1",
        "status": "development-only sound member-wise budget attribution",
        "config": config,
        "config_hash": config_hash,
        "selection_report": str(selection_path.relative_to(root)).replace(
            "\\", "/"
        ),
        "selection_report_hash": selection_hash,
        "scientific_source_hashes": scientific_source_hashes,
        "scientific_source_manifest_hash": scientific_source_manifest_hash,
        "seed": seed,
        "dataset_index": dataset_index,
        "member_count": len(member_rows),
        "member_rows": member_rows,
        "shards": shard_records,
        "seconds": time.perf_counter() - started,
        "code_revision": code_revision(root),
        "interpretation": (
            "A member certificate applies only to its mutually exclusive discrete "
            "semantics member and the full continuous timestep/threshold box. "
            "Comparing rows identifies which axes consume Cartesian proof budget; "
            "it is not an audit-population estimate."
        ),
    }
    write_json_immutable(output_path, report)


if __name__ == "__main__":
    main()
