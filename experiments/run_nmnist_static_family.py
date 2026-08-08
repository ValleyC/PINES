from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_shd_static_family import _families
from transportcert.abstract import IntervalFamilyCertifier
from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.benchmarks.nmnist import PackedNMNIST
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.certificates import CertificateEngine
from transportcert.models import DenseRecurrentSNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data/processed/nmnist_v1")
    parser.add_argument("--artifact-root", default="artifacts/nmnist_v1")
    parser.add_argument(
        "--output-root", default="artifacts/nmnist_v1_static_family"
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument(
        "--families",
        nargs="*",
        default=("reset", "integration", "timing", "delay", "full"),
        choices=(
            "reset",
            "integration",
            "timing",
            "delay",
            "full",
            "fixed_nearest",
            "fixed_floor",
        ),
    )
    args = parser.parse_args()
    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "static_family_report.json"
    if output_path.exists():
        raise FileExistsError(f"static family report exists: {output_path}")

    store = PackedNMNIST(root / args.data_root / "train.npz")
    with np.load(seed_dir / "split_indices.npz", allow_pickle=False) as splits:
        complete_audit_indices = np.asarray(
            splits["certificate_audit"], dtype=np.int64
        )
    audit_indices = complete_audit_indices[: args.max_samples]
    inputs = store.frames(audit_indices)
    model = DenseRecurrentSNN.load(seed_dir / "model.npz")
    if np.count_nonzero(model.recurrent_weights) != 0:
        raise ValueError("N-MNIST negative control must be feedforward")

    reference = primary_semantic_conditions()["reference"]
    families = _families()
    certifier = IntervalFamilyCertifier()
    engine = CertificateEngine()
    rows = []
    for name in args.families:
        box, family = families[name]
        started = time.perf_counter()
        static = certifier.certify(model, inputs, reference, box)
        static_seconds = time.perf_counter() - started
        started = time.perf_counter()
        exact = engine.certify_family(model, inputs, reference, family)
        exact_seconds = time.perf_counter() - started
        unsound = int(np.count_nonzero(static.certified & ~exact.certified))
        widths = static.target_logit_upper - static.target_logit_lower
        row = {
            "family": name,
            "box_hash": box.box_hash,
            "member_semantics_hashes": [
                member.semantics_hash for member in family.members
            ],
            "member_count": len(family.members),
            "samples": len(audit_indices),
            "static_certified_inputs": int(np.count_nonzero(static.certified)),
            "static_certified_fraction": static.certified_fraction,
            "exact_family_agreement_inputs": int(np.count_nonzero(exact.certified)),
            "exact_family_agreement_fraction": exact.certified_fraction,
            "observed_unsound_certificates": unsound,
            "median_final_logit_interval_width": float(np.median(widths)),
            "p90_final_logit_interval_width": float(np.percentile(widths, 90)),
            "maximum_final_logit_interval_width": float(np.max(widths)),
            "static_seconds": static_seconds,
            "exact_seconds": exact_seconds,
        }
        rows.append(row)
        print(
            f"seed={args.seed} family={name} "
            f"static={static.certified_fraction:.4f} "
            f"exact={exact.certified_fraction:.4f} unsound={unsound}",
            flush=True,
        )

    report = {
        "schema_version": "NMNISTStaticFamilyExperiment/v1",
        "status": (
            "software interval-family diagnostic on a deterministic audit subset; "
            "not primary or physical evidence"
        ),
        "seed": args.seed,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(seed_dir / "model.npz"),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(seed_dir / "split_indices.npz"),
        "complete_audit_samples": len(complete_audit_indices),
        "audit_sample_indices_hash": array_hash(audit_indices),
        "audit_samples": len(audit_indices),
        "subset_rule": "first max_samples entries of the frozen certificate audit split",
        "reference_semantics_hash": reference.semantics_hash,
        "rows": rows,
        "gate_assessment": {
            "full_family_certifies_at_least_20_percent": next(
                row["static_certified_fraction"] >= 0.20
                for row in rows
                if row["family"] == "full"
            ),
            "no_observed_static_unsoundness": all(
                row["observed_unsound_certificates"] == 0 for row in rows
            ),
        },
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, report)
    print(json.dumps(report["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
