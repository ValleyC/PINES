from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_shd_static_family import _families
from pines.abstract import IntervalFamilyCertifier
from pines.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.certificates import CertificateEngine
from pines.emulator import VectorizedEmulator
from pines.models import DenseRecurrentSNN


def _margin(logits: np.ndarray) -> np.ndarray:
    ordered = np.partition(logits, -2, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--output-root", default="artifacts/shd_v1_horizon_diagnostic"
    )
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=(10, 20, 30, 40, 50)
    )
    args = parser.parse_args()
    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    if any(value <= 0 for value in args.horizons):
        raise ValueError("horizons must be positive")
    if tuple(sorted(set(args.horizons))) != tuple(args.horizons):
        raise ValueError("horizons must be unique and increasing")

    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "horizon_report.json"
    if output_path.exists():
        raise FileExistsError(f"horizon report exists: {output_path}")

    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(seed_dir / "split_indices.npz", allow_pickle=False) as splits:
        complete_audit_indices = np.asarray(
            splits["certificate_audit"], dtype=np.int64
        )
    audit_indices = complete_audit_indices[: args.max_samples]
    inputs = store.frames(audit_indices)
    if max(args.horizons) > inputs.shape[1]:
        raise ValueError("requested horizon exceeds preprocessed input horizon")

    source = DenseRecurrentSNN.load(seed_dir / "model.npz")
    models = {
        "trained_recurrent": source,
        "zero_recurrence": source.with_parameters(
            recurrent_weights=np.zeros_like(source.recurrent_weights),
            name=f"{source.name}-zero-recurrence-ablation",
        ),
    }
    reference = primary_semantic_conditions()["reference"]
    box, family = _families()["full"]
    certifier = IntervalFamilyCertifier()
    engine = CertificateEngine()
    emulator = VectorizedEmulator()
    rows = []
    for model_name, model in models.items():
        for horizon in args.horizons:
            prefix = inputs[:, :horizon, :]
            reference_logits = emulator.run(model, prefix, reference).final_logits
            margins = _margin(reference_logits)
            started = time.perf_counter()
            static = certifier.certify(model, prefix, reference, box)
            static_seconds = time.perf_counter() - started
            started = time.perf_counter()
            exact = engine.certify_family(model, prefix, reference, family)
            exact_seconds = time.perf_counter() - started
            unsound = int(np.count_nonzero(static.certified & ~exact.certified))
            rows.append(
                {
                    "model_condition": model_name,
                    "model_hash": model.model_hash,
                    "horizon": horizon,
                    "samples": len(audit_indices),
                    "static_certified_fraction": static.certified_fraction,
                    "exact_family_agreement_fraction": exact.certified_fraction,
                    "relaxation_gap": float(
                        exact.certified_fraction - static.certified_fraction
                    ),
                    "observed_unsound_certificates": unsound,
                    "median_reference_margin": float(np.median(margins)),
                    "p10_reference_margin": float(np.percentile(margins, 10)),
                    "positive_reference_margin_fraction": float(np.mean(margins > 0)),
                    "static_seconds": static_seconds,
                    "exact_seconds": exact_seconds,
                }
            )
            print(
                f"seed={args.seed} model={model_name} horizon={horizon} "
                f"static={static.certified_fraction:.4f} "
                f"exact={exact.certified_fraction:.4f} unsound={unsound}",
                flush=True,
            )

    report = {
        "schema_version": "SHDHorizonDiagnostic/v1",
        "status": (
            "software mechanism diagnostic on a deterministic audit subset; "
            "not primary or physical evidence"
        ),
        "seed": args.seed,
        "source_model_hash": source.model_hash,
        "model_artifact_hash": sha256_file(seed_dir / "model.npz"),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(seed_dir / "split_indices.npz"),
        "complete_audit_samples": len(complete_audit_indices),
        "audit_samples": len(audit_indices),
        "audit_sample_indices_hash": array_hash(audit_indices),
        "subset_rule": "first max_samples entries of the frozen certificate audit split",
        "horizons": list(args.horizons),
        "family_box_hash": box.box_hash,
        "family_member_semantics_hashes": [
            member.semantics_hash for member in family.members
        ],
        "rows": rows,
        "gate_assessment": {
            "no_observed_static_unsoundness": all(
                row["observed_unsound_certificates"] == 0 for row in rows
            )
        },
        "code_revision": code_revision(root),
    }
    write_json_immutable(output_path, report)
    print(json.dumps(report["gate_assessment"], indent=2))


if __name__ == "__main__":
    main()
