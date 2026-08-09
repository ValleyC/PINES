from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

from pines.artifacts import sha256_file, sha256_json, write_json_immutable


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", *arguments],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


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
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_seeds = [int(seed) for seed in config["seeds"]]
    configured_shards_per_seed = int(config["execution"]["shards_per_seed"])
    expected_total = len(configured_seeds) * configured_shards_per_seed
    shard_paths = sorted((audit_root / "shards").glob("seed_*_shard_*.json"))
    if not shard_paths:
        raise ValueError("audit has no immutable shards")
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
    config_hash = sha256_file(config_path)
    shard_config_hashes = {shard["config_hash"] for shard in shards}
    if shard_config_hashes != {config_hash}:
        raise ValueError("current config does not match every immutable shard")

    source_paths = sorted((root / "src" / "pines").rglob("*.py"))
    driver_path = root / "experiments" / "run_shd_hybrid_family_full_audit.py"
    source_paths.append(driver_path)
    relative_sources = [path.relative_to(root).as_posix() for path in source_paths]
    head = _git(root, "rev-parse", "HEAD")
    dirty_dependencies = _git(
        root, "diff", "--name-only", head, "--", *relative_sources
    ).splitlines()
    if dirty_dependencies:
        raise ValueError(
            "scientific source differs from recorded HEAD: "
            + ", ".join(dirty_dependencies)
        )
    source_hashes = {
        relative: sha256_file(root / relative) for relative in relative_sources
    }

    shard_revisions = sorted({str(shard["code_revision"]) for shard in shards})
    report = {
        "schema_version": "SHDHybridAuditScientificSourceProvenance/v1",
        "status": (
            "independent post-run provenance supplement for worker shards whose "
            "Git revision lookup failed"
        ),
        "audit_root": args.audit_root.replace("\\", "/"),
        "shard_count": len(shard_paths),
        "expected_shard_count": expected_total,
        "shard_code_revision_values": shard_revisions,
        "git_head": head,
        "scientific_sources_match_git_head": True,
        "scientific_source_hashes": source_hashes,
        "scientific_source_manifest_hash": sha256_json(source_hashes),
        "config_path": args.config.replace("\\", "/"),
        "config_hash": config_hash,
        "config_hash_matches_every_shard": True,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "assumption": (
            "The long-lived worker processes imported these dependency files before "
            "the first shard. The certifier package and audit driver remain byte-for-byte "
            "at Git HEAD; the only scientific run-time configuration is independently "
            "identified by the hash embedded in every shard."
        ),
    }
    write_json_immutable(root / args.output, report)
    print(json.dumps({key: report[key] for key in (
        "shard_count",
        "git_head",
        "scientific_sources_match_git_head",
        "config_hash_matches_every_shard",
    )}, indent=2))


if __name__ == "__main__":
    main()
