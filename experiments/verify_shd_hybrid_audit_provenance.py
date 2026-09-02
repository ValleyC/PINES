from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

from pines.artifacts import file_reference, config_description, write_json


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", *arguments],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _function_manifest(source: str, names: tuple[str, ...]) -> dict[str, str]:
    tree = ast.parse(source)
    definitions = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [name for name in names if not definitions.get(name)]
    if missing:
        raise ValueError(f"driver functions missing from source: {missing}")
    return {name: config_description(definitions[name]) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-root",
        default="artifacts/shd_v68_hybrid_full_audit_soundness_corrected_v1",
    )
    parser.add_argument(
        "--config", default="configs/experiments/shd_hybrid_full_audit_v3.json"
    )
    parser.add_argument(
        "--audit-report",
        default=(
            "artifacts/shd_v68_hybrid_full_audit_soundness_corrected_v1/"
            "hybrid_family_full_audit.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/shd_v68_hybrid_full_audit_soundness_corrected_v1/"
            "scientific_source_provenance.json"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    audit_root = root / args.audit_root
    config_path = root / args.config
    audit_report_path = root / args.audit_report
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_seeds = [int(seed) for seed in config["seeds"]]
    configured_shards_per_seed = int(config["execution"]["shards_per_seed"])
    expected_total = len(configured_seeds) * configured_shards_per_seed
    shard_paths = sorted((audit_root / "shards").glob("seed_*_shard_*.json"))
    if not shard_paths:
        raise ValueError("audit has no saved shards")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    per_seed_shard_counts: dict[int, int] = {}
    for shard in shards:
        seed = int(shard["seed"])
        count = int(shard["shard_count"])
        if seed in per_seed_shard_counts and per_seed_shard_counts[seed] != count:
            raise ValueError(f"inconsistent declared shard count for seed {seed}")
        per_seed_shard_counts[seed] = count
    if set(per_seed_shard_counts) - set(configured_seeds):
        raise ValueError("audit contains a seed absent from the frozen config")
    if any(count != configured_shards_per_seed for count in per_seed_shard_counts.values()):
        raise ValueError("shard declaration differs from the frozen config")
    if len(shard_paths) != expected_total:
        raise ValueError(
            f"audit is incomplete: found {len(shard_paths)} of {expected_total} shards"
        )
    config_reference = file_reference(config_path)
    shard_config_references = {shard["config_reference"] for shard in shards}
    if shard_config_references != {config_reference}:
        raise ValueError("current config does not match every saved shard")

    source_paths = [
        root / path
        for path in (
            "src/pines/abstract.py",
            "src/pines/affine.py",
            "src/pines/artifacts.py",
            "src/pines/emulator.py",
            "src/pines/models.py",
            "src/pines/protocol.py",
            "src/pines/semantics.py",
            "src/pines/statistics.py",
            "src/pines/torch_emulator.py",
            "src/pines/benchmarks/semantic_matrix.py",
            "src/pines/benchmarks/shd.py",
        )
    ]
    driver_path = root / "experiments" / "run_shd_hybrid_family_full_audit.py"
    relative_sources = [path.relative_to(root).as_posix() for path in source_paths]
    head = _git(root, "rev-parse", "HEAD")
    audit_revision = str(audit_report["code_revision"])
    if len(audit_revision) != 40:
        raise ValueError("audit report does not contain a Git revision")
    dirty_dependencies = _git(
        root, "diff", "--name-only", head, "--", *relative_sources
    ).splitlines()
    if dirty_dependencies:
        raise ValueError(
            "scientific source differs from recorded HEAD: "
            + ", ".join(dirty_dependencies)
        )
    changed_since_audit = _git(
        root,
        "diff",
        "--name-only",
        audit_revision,
        head,
        "--",
        *relative_sources,
    ).splitlines()
    if changed_since_audit:
        raise ValueError(
            "scientific core changed after the audit revision: "
            + ", ".join(changed_since_audit)
        )
    source_references = {
        relative: file_reference(root / relative) for relative in relative_sources
    }
    driver_relative = driver_path.relative_to(root).as_posix()
    driver_functions = ("_box_and_certifier", "_run_shard", "_validated_shard")
    audit_driver = _git(root, "show", f"{audit_revision}:{driver_relative}")
    current_driver = driver_path.read_text(encoding="utf-8")
    audit_driver_functions = _function_manifest(audit_driver, driver_functions)
    current_driver_functions = _function_manifest(current_driver, driver_functions)
    if audit_driver_functions != current_driver_functions:
        raise ValueError("scientific driver functions changed after the audit")

    shard_revisions = sorted({str(shard["code_revision"]) for shard in shards})
    report = {
        "schema_version": "SHDHybridAuditScientificSourceProvenance/v2",
        "status": (
            "independent post-run provenance supplement for worker shards whose "
            "Git revision lookup failed"
        ),
        "audit_root": args.audit_root.replace("\\", "/"),
        "audit_report": args.audit_report.replace("\\", "/"),
        "audit_report_description": file_reference(audit_report_path),
        "audit_report_code_revision": audit_revision,
        "shard_count": len(shard_paths),
        "expected_shard_count": expected_total,
        "shard_code_revision_values": shard_revisions,
        "git_head": head,
        "scientific_sources_match_git_head": True,
        "scientific_core_unchanged_since_audit_revision": True,
        "scientific_source_references": source_references,
        "scientific_source_manifest_reference": config_description(source_references),
        "scientific_driver_function_references": current_driver_functions,
        "scientific_driver_functions_unchanged_since_audit_revision": True,
        "config_path": args.config.replace("\\", "/"),
        "config_reference": config_reference,
        "config_reference_matches_every_shard": True,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "assumption": (
            "Worker subprocesses that could not query Git recorded uncommitted. The "
            "scientific dependency closure and the three driver functions that construct "
            "and execute each shard are unchanged from the aggregate report revision. "
            "Scheduling-only driver changes do not alter scientific results."
        ),
    }
    write_json(root / args.output, report)
    print(json.dumps({key: report[key] for key in (
        "shard_count",
        "git_head",
        "scientific_sources_match_git_head",
        "config_reference_matches_every_shard",
    )}, indent=2))


if __name__ == "__main__":
    main()
