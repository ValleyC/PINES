from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from transportcert.abstract import IntervalFamilyCertifier, SemanticsBox
from transportcert.artifacts import (
    array_hash,
    code_revision,
    sha256_file,
    write_json_immutable,
)
from transportcert.benchmarks.semantic_matrix import primary_semantic_conditions
from transportcert.benchmarks.shd import PackedSHD
from transportcert.certificates import CertificateEngine, SemanticsFamily
from transportcert.models import DenseRecurrentSNN
from transportcert.semantics import (
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
    UpdateOrdering,
)


def _ordering(timing: ThresholdTiming) -> UpdateOrdering:
    if timing is ThresholdTiming.PRE_INTEGRATION:
        return UpdateOrdering.THRESHOLD_RESET_INTEGRATE
    return UpdateOrdering.INTEGRATE_THRESHOLD_RESET


def _families() -> dict[str, tuple[SemanticsBox, SemanticsFamily]]:
    conditions = primary_semantic_conditions()
    reference = conditions["reference"]

    def build(
        name: str,
        integrations: tuple[IntegrationRule, ...] = (
            IntegrationRule.FORWARD_EULER,
        ),
        timings: tuple[ThresholdTiming, ...] = (
            ThresholdTiming.POST_INTEGRATION,
        ),
        resets: tuple[ResetRule, ...] = (ResetRule.SUBTRACTIVE,),
        delays: tuple[int, ...] = (0,),
    ) -> tuple[SemanticsBox, SemanticsFamily]:
        box = SemanticsBox(
            base=reference,
            timestep_bounds=(1.0, 1.0),
            integration_rules=integrations,
            threshold_timings=timings,
            reset_rules=resets,
            synaptic_delays=delays,
            name=name,
        )
        members = tuple(
            replace(
                reference,
                integration_rule=integration,
                threshold_timing=timing,
                update_ordering=_ordering(timing),
                reset_rule=reset,
                synaptic_delay_steps=delay,
            )
            for integration, timing, reset, delay in itertools.product(
                integrations, timings, resets, delays
            )
        )
        return box, SemanticsFamily(members, name=name)

    families = {
        "reset": build(
            "reset-family", resets=(ResetRule.SUBTRACTIVE, ResetRule.TO_VALUE)
        ),
        "integration": build(
            "integration-family",
            integrations=(
                IntegrationRule.FORWARD_EULER,
                IntegrationRule.EXPONENTIAL_EULER,
            ),
        ),
        "timing": build(
            "timing-family",
            timings=(
                ThresholdTiming.POST_INTEGRATION,
                ThresholdTiming.PRE_INTEGRATION,
            ),
        ),
        "delay": build("delay-family", delays=(0, 1)),
        "full": build(
            "full-high-risk-family",
            integrations=(
                IntegrationRule.FORWARD_EULER,
                IntegrationRule.EXPONENTIAL_EULER,
            ),
            timings=(
                ThresholdTiming.POST_INTEGRATION,
                ThresholdTiming.PRE_INTEGRATION,
            ),
            resets=(ResetRule.SUBTRACTIVE, ResetRule.TO_VALUE),
            delays=(0, 1),
        ),
    }
    for name, condition in (
        ("fixed_nearest", "fixed_q8_weights_q16_state"),
        ("fixed_floor", "floor_rounding_saturation"),
    ):
        target = conditions[condition]
        families[name] = (
            SemanticsBox(
                base=target,
                timestep_bounds=(1.0, 1.0),
                integration_rules=(target.integration_rule,),
                threshold_timings=(target.threshold_timing,),
                reset_rules=(target.reset_rule,),
                synaptic_delays=(target.synaptic_delay_steps,),
                output_delays=(target.output_delay_steps,),
                name=f"{name}-target",
            ),
            SemanticsFamily((target,), name=f"{name}-target"),
        )
    return families


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data/processed/shd_v1")
    parser.add_argument("--artifact-root", default="artifacts/shd_v1_final")
    parser.add_argument("--output-root", default="artifacts/shd_v1_static_family")
    parser.add_argument(
        "--families",
        nargs="*",
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
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    seed_dir = root / args.artifact_root / f"seed_{args.seed}"
    output_dir = root / args.output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "static_family_report.json"
    if output_path.exists():
        raise FileExistsError(f"static family report exists: {output_path}")
    store = PackedSHD(root / args.data_root / "train.npz")
    with np.load(seed_dir / "split_indices.npz", allow_pickle=False) as splits:
        audit_indices = np.asarray(splits["certificate_audit"], dtype=np.int64)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        audit_indices = audit_indices[: args.max_samples]
    inputs = store.frames(audit_indices)
    model = DenseRecurrentSNN.load(seed_dir / "model.npz")
    reference = primary_semantic_conditions()["reference"]
    all_families = _families()
    selected_names = tuple(args.families or all_families.keys())
    rows = []
    certifier = IntervalFamilyCertifier()
    engine = CertificateEngine()
    for name in selected_names:
        box, family = all_families[name]
        started = time.perf_counter()
        static = certifier.certify(model, inputs, reference, box)
        static_seconds = time.perf_counter() - started
        started = time.perf_counter()
        exact = engine.certify_family(model, inputs, reference, family)
        exact_seconds = time.perf_counter() - started
        unsound = int(np.count_nonzero(static.certified & ~exact.certified))
        widths = static.target_logit_upper - static.target_logit_lower
        rows.append(
            {
                "family": name,
                "box_hash": box.box_hash,
                "member_semantics_hashes": [
                    member.semantics_hash for member in family.members
                ],
                "member_count": len(family.members),
                "samples": len(audit_indices),
                "static_certified_inputs": int(np.count_nonzero(static.certified)),
                "static_certified_fraction": static.certified_fraction,
                "exact_family_agreement_inputs": int(
                    np.count_nonzero(exact.certified)
                ),
                "exact_family_agreement_fraction": exact.certified_fraction,
                "observed_unsound_certificates": unsound,
                "median_final_logit_interval_width": float(np.median(widths)),
                "p90_final_logit_interval_width": float(np.percentile(widths, 90)),
                "maximum_final_logit_interval_width": float(np.max(widths)),
                "static_seconds": static_seconds,
                "exact_seconds": exact_seconds,
            }
        )
        print(
            f"seed={args.seed} family={name} "
            f"static={static.certified_fraction:.4f} "
            f"exact={exact.certified_fraction:.4f} unsound={unsound}",
            flush=True,
        )
    report = {
        "schema_version": "SHDStaticFamilyExperiment/v1",
        "status": "software interval-family evidence; not physical evidence",
        "seed": args.seed,
        "model_hash": model.model_hash,
        "model_artifact_hash": sha256_file(seed_dir / "model.npz"),
        "train_store_hash": store.data_hash,
        "split_indices_hash": sha256_file(seed_dir / "split_indices.npz"),
        "audit_sample_indices_hash": array_hash(audit_indices),
        "audit_samples": len(audit_indices),
        "reference_semantics_hash": reference.semantics_hash,
        "rows": rows,
        "gate_assessment": {
            "full_family_certifies_at_least_20_percent": next(
                (
                    row["static_certified_fraction"] >= 0.20
                    for row in rows
                    if row["family"] == "full"
                ),
                None,
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
