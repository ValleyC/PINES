from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .abstract import IntervalFamilyCertifier, SemanticsBox
from .adapters.hardware import load_hardware_capture
from .artifacts import array_hash, sha256_json, write_json_immutable
from .certificates import CertificateEngine, SemanticsFamily
from .emulator import VectorizedEmulator
from .models import DenseRecurrentSNN
from .repair import CertificateDirectedRepair
from .reports import CertificateReport
from .reproduce import reproduce_from_config
from .semantics import ExecutionSemantics
from .statistics import disagreement_bound


def _load_array(path: str | Path, key: str = "inputs") -> np.ndarray:
    source = Path(path)
    if source.suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if key not in data:
            raise KeyError(f"{source} does not contain array '{key}'")
        return np.asarray(data[key])
    if source.suffix == ".npy":
        return np.load(source, allow_pickle=False)
    with np.load(source, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{source} does not contain array '{key}'")
        return np.asarray(data[key])


def _load_certificate(path: str | Path) -> CertificateReport:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("target_semantics_hashes", "assumptions", "member_bounds"):
        data[key] = tuple(data.get(key, ()))
    return CertificateReport(**data)


def _add_common_model_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--sample-ids")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transportcert")
    parser.add_argument("--version", action="version", version="transportcert 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    emulate = subparsers.add_parser("emulate", help="execute an SNN under declared semantics")
    _add_common_model_inputs(emulate)
    emulate.add_argument("--semantics", required=True)
    emulate.add_argument("--output", required=True)

    certify = subparsers.add_parser("certify", help="build a conditional or physical certificate")
    _add_common_model_inputs(certify)
    certify.add_argument("--reference", required=True)
    certify.add_argument("--target", action="append", required=True)
    certify.add_argument(
        "--semantics-box",
        help="optional bounded static family; --target remains the statistical family",
    )
    certify.add_argument("--delta", type=float, default=0.05)
    certify.add_argument("--budget", type=float, default=0.05)
    certify.add_argument("--dataset-split", default="certificate-audit")
    certify.add_argument("--hardware-capture")
    certify.add_argument("--hardware-manifest")
    certify.add_argument("--hardware-target-index", type=int, default=0)
    certify.add_argument("--output", required=True)

    repair = subparsers.add_parser("repair", help="run label-free certificate-directed repair")
    repair.add_argument("--model", required=True)
    repair.add_argument("--calibration-inputs", required=True)
    repair.add_argument("--audit-inputs", required=True)
    repair.add_argument("--calibration-ids", required=True)
    repair.add_argument("--audit-ids", required=True)
    repair.add_argument("--reference", required=True)
    repair.add_argument("--target", required=True)
    repair.add_argument("--source-certificate", required=True)
    repair.add_argument("--output-model", required=True)
    repair.add_argument("--output-report", required=True)
    repair.add_argument("--output-certificate", required=True)

    validate = subparsers.add_parser(
        "validate-hardware", help="bound emulator-to-hardware disagreement"
    )
    validate.add_argument("--emulator-predictions", required=True)
    validate.add_argument("--hardware-capture", required=True)
    validate.add_argument("--hardware-manifest", required=True)
    validate.add_argument("--delta", type=float, default=0.025)
    validate.add_argument("--output", required=True)

    reproduce = subparsers.add_parser("reproduce", help="run a frozen reproduction config")
    reproduce.add_argument("--config", required=True)
    reproduce.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    if args.command == "emulate":
        model = DenseRecurrentSNN.load(args.model)
        inputs = _load_array(args.inputs)
        semantics = ExecutionSemantics.load(args.semantics)
        trace = VectorizedEmulator().run(model, inputs, semantics)
        sample_ids = (
            _load_array(args.sample_ids, "sample_ids").astype(str)
            if args.sample_ids
            else np.asarray([str(index) for index in range(len(trace.predictions))])
        )
        if sample_ids.shape != trace.predictions.shape:
            raise ValueError("sample IDs must match emulated predictions")
        write_json_immutable(
            args.output,
            {
                "schema_version": "EmulationReport/v1",
                "model_hash": model.model_hash,
                "data_hash": array_hash(inputs),
                "semantics_hash": semantics.semantics_hash,
                "sample_count": len(trace.predictions),
                "sample_ids": sample_ids,
                "sample_ids_hash": sha256_json(sample_ids.tolist()),
                "predictions": trace.predictions,
                "final_logits": trace.final_logits,
            },
        )
        return 0

    if args.command == "certify":
        model = DenseRecurrentSNN.load(args.model)
        inputs = _load_array(args.inputs)
        reference = ExecutionSemantics.load(args.reference)
        family = SemanticsFamily(tuple(ExecutionSemantics.load(path) for path in args.target))
        static_fraction = None
        static_family_hash = None
        if args.semantics_box:
            semantics_box = SemanticsBox.load(args.semantics_box)
            static_result = IntervalFamilyCertifier().certify(
                model, inputs, reference, semantics_box
            )
            static_fraction = static_result.certified_fraction
            static_family_hash = semantics_box.box_hash
        hardware_predictions = None
        firmware_hash = None
        bitstream_hash = None
        if bool(args.hardware_capture) != bool(args.hardware_manifest):
            raise ValueError("hardware capture and manifest must be supplied together")
        if args.hardware_capture:
            hardware_predictions, hardware_sample_ids, manifest = load_hardware_capture(
                args.hardware_capture, args.hardware_manifest
            )
            if not args.sample_ids:
                raise ValueError("physical certification requires --sample-ids")
            requested_sample_ids = _load_array(args.sample_ids, "sample_ids").astype(str)
            if not np.array_equal(hardware_sample_ids, requested_sample_ids):
                raise ValueError("hardware sample ordering differs from requested audit IDs")
            if manifest.model_hash != model.model_hash:
                raise ValueError("hardware manifest model hash differs from requested model")
            if manifest.dataset_hash != array_hash(inputs):
                raise ValueError("hardware manifest dataset hash differs from audit inputs")
            expected_semantics_hash = family.members[
                args.hardware_target_index
            ].semantics_hash
            if manifest.semantics_hash != expected_semantics_hash:
                raise ValueError("hardware manifest semantics differs from selected target")
            firmware_hash = manifest.firmware_hash
            bitstream_hash = manifest.bitstream_hash
        report = CertificateEngine().build_report(
            model,
            inputs,
            reference,
            family,
            delta=args.delta,
            decision_budget=args.budget,
            dataset_split=args.dataset_split,
            checkpoint_hash=model.model_hash,
            seed_manifest={
                "reference": reference.randomness.seed,
                "targets": [member.randomness.seed for member in family.members],
            },
            hardware_predictions=hardware_predictions,
            hardware_target_index=args.hardware_target_index,
            firmware_hash=firmware_hash,
            bitstream_hash=bitstream_hash,
            repository_root=repository_root,
            static_certified_fraction=static_fraction,
            static_family_hash=static_family_hash,
        )
        report.write(args.output)
        return 0

    if args.command == "repair":
        model = DenseRecurrentSNN.load(args.model)
        calibration = _load_array(args.calibration_inputs)
        audit = _load_array(args.audit_inputs)
        calibration_ids = _load_array(args.calibration_ids, "sample_ids").astype(str)
        audit_ids = _load_array(args.audit_ids, "sample_ids").astype(str)
        outcome = CertificateDirectedRepair().repair(
            model,
            calibration,
            audit,
            calibration_ids,
            audit_ids,
            ExecutionSemantics.load(args.reference),
            ExecutionSemantics.load(args.target),
            _load_certificate(args.source_certificate),
            repository_root=str(repository_root),
        )
        outcome.model.save(args.output_model)
        outcome.report.write(args.output_report)
        outcome.post_repair_certificate.write(args.output_certificate)
        return 0

    if args.command == "validate-hardware":
        emulator_predictions = _load_array(args.emulator_predictions, "predictions")
        hardware_predictions, _, manifest = load_hardware_capture(
            args.hardware_capture, args.hardware_manifest
        )
        bound = disagreement_bound(emulator_predictions, hardware_predictions, args.delta)
        write_json_immutable(
            args.output,
            {
                "schema_version": "HardwareConformanceReport/v1",
                "manifest": manifest,
                "emulator_predictions_hash": array_hash(emulator_predictions),
                "errors": bound.errors,
                "samples": bound.samples,
                "empirical_rate": bound.empirical_rate,
                "upper_bound": bound.upper_bound,
                "confidence_level": 1.0 - bound.alpha,
            },
        )
        return 0

    if args.command == "reproduce":
        reproduce_from_config(args.config, args.output_dir)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
