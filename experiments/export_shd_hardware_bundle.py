from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pines.artifacts import write_json
from pines.benchmarks.semantic_matrix import primary_semantic_conditions
from pines.benchmarks.shd import PackedSHD
from pines.hardware_bundle import (
    destination_major,
    fixed_integer,
    json_scalar,
    leak_reciprocal_q24,
    run_floor_q8q16_integer,
    write_npz,
    write_mem,
    write_spike_mem,
)
from pines.models import DenseRecurrentSNN
from pines.torch_emulator import TorchEmulator


SEEDS = (1701, 2718, 3141, 5772, 8119)
SMOKE_SAMPLES = 8


def _copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)


def _batched_trace(
    emulator: TorchEmulator,
    model: DenseRecurrentSNN,
    frames: np.ndarray,
    semantics: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        trace = emulator.run(model, frames[start : start + batch_size], semantics)
        converted = trace.numpy()
        predictions.append(converted.predictions.astype(np.int16))
        logits.append(converted.final_logits)
    return np.concatenate(predictions), np.concatenate(logits)


def _state_integers(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    scaled = np.rint(np.asarray(values) * (1 << fractional_bits)).astype(np.int64)
    if np.any(scaled < -32768) or np.any(scaled > 32767):
        raise ValueError("golden state lies outside signed Q16 storage")
    return scaled.astype(np.int16)


def _deployment_model(model: DenseRecurrentSNN, target: Any) -> tuple[DenseRecurrentSNN, np.ndarray]:
    """Apply the explicit digital mapping used by the FPGA handoff."""

    leak_q24 = leak_reciprocal_q24(model.tau_mem)
    tau_mapped = (1 << 24) / leak_q24.astype(np.float64)
    return (
        model.with_parameters(
            bias=np.asarray(target.state_format.quantize(model.bias)),
            threshold=np.asarray(target.state_format.quantize(model.threshold)),
            reset_value=np.asarray(target.state_format.quantize(model.reset_value)),
            tau_mem=tau_mapped,
        ),
        leak_q24,
    )


def _export_parameters(
    model: DenseRecurrentSNN,
    output: Path,
    target: Any,
    *,
    leak_q24: np.ndarray | None = None,
) -> list[str]:
    weight = target.weight_format
    state = target.state_format
    assert weight.total_bits == 8 and weight.fractional_bits == 6
    assert state.total_bits == 16 and state.fractional_bits == 8

    w_in = fixed_integer(model.input_weights, weight)
    w_rec = fixed_integer(model.recurrent_weights, weight)
    w_out = fixed_integer(model.output_weights, weight)
    bias = fixed_integer(model.bias, state)
    threshold = fixed_integer(model.threshold, state)
    reset_value = fixed_integer(model.reset_value, state)
    if leak_q24 is None:
        leak_q24 = leak_reciprocal_q24(model.tau_mem)
    else:
        leak_q24 = np.asarray(leak_q24, dtype=np.uint32)

    parameter_path = write_npz(
        output / "parameters_int.npz",
        {
            "input_weights_q8": w_in,
            "recurrent_weights_q8": w_rec,
            "output_weights_q8": w_out,
            "bias_q16": bias,
            "threshold_q16": threshold,
            "reset_value_q16": reset_value,
            "leak_reciprocal_q24": leak_q24,
            "metadata": json_scalar(
                {
                    "schema_version": "SHDHardwareParameters/v1",
                    "canonical_matrix_order": "source,destination",
                    "memory_matrix_order": "destination-major,source-minor",
                    "weight_bits": 8,
                    "weight_fractional_bits": 6,
                    "state_bits": 16,
                    "state_fractional_bits": 8,
                    "weight_to_state_left_shift": 2,
                    "leak_storage_bits": 32,
                    "leak_fractional_bits": 24,
                    "integration_product_right_shift": 24,
                }
            ),
        },
    )
    paths: dict[str, Path] = {
        "parameters_int.npz": parameter_path,
        "input_weights.mem": write_mem(
            output / "input_weights.mem", destination_major(w_in), 8
        ),
        "recurrent_weights.mem": write_mem(
            output / "recurrent_weights.mem", destination_major(w_rec), 8
        ),
        "output_weights.mem": write_mem(
            output / "output_weights.mem", destination_major(w_out), 8
        ),
        "bias.mem": write_mem(output / "bias.mem", bias, 16),
        "threshold.mem": write_mem(output / "threshold.mem", threshold, 16),
        "reset_value.mem": write_mem(output / "reset_value.mem", reset_value, 16),
        "leak_reciprocal_q24.mem": write_mem(
            output / "leak_reciprocal_q24.mem", leak_q24.astype(np.int64), 32
        ),
    }
    return list(paths)


def _write_sample_index(path: Path, sample_ids: np.ndarray, time_bins: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("sample_position", "sample_id", "first_mem_line", "time_bins"))
        for position, sample_id in enumerate(sample_ids):
            writer.writerow((position, str(sample_id), position * time_bins, time_bins))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default="artifacts/shd_v1_final")
    parser.add_argument(
        "--repair-root", default="artifacts/shd_v113_margin_0p25_full_da9c0f9_v1"
    )
    parser.add_argument("--train-store", default="data/processed/shd_v1/train.npz")
    parser.add_argument(
        "--output-root", default="hardware/bundles/shd_floor_q8q16_v1"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = root / args.output_root
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output}")
    output.mkdir(parents=True)

    _copy_new(
        root / "hardware" / "templates" / "shd_hardware_bundle_README.md",
        output / "README.md",
    )
    _copy_new(
        root / "hardware" / "templates" / "shd_hardware_bundle_DATA_LICENSE.md",
        output / "DATA_LICENSE.md",
    )

    source_root = root / args.model_root
    repair_root = root / args.repair_root
    train_path = root / args.train_store
    train_store = PackedSHD(train_path)
    first_split = source_root / f"seed_{args.seeds[0]}" / "split_indices.npz"
    with np.load(first_split, allow_pickle=False) as split_data:
        audit_indices = np.asarray(split_data["certificate_audit"], dtype=np.int64)
    for seed in args.seeds[1:]:
        with np.load(
            source_root / f"seed_{seed}" / "split_indices.npz", allow_pickle=False
        ) as split_data:
            if not np.array_equal(audit_indices, split_data["certificate_audit"]):
                raise ValueError("certificate audit indices differ across seeds")

    audit_ids = train_store.sample_ids[audit_indices]
    audit_packed = train_store.packed[audit_indices]
    audit_frames = train_store.frames(audit_indices)
    common = output / "common"
    audit_metadata = {
        "schema_version": "SHDHardwareAuditInputs/v1",
        "dataset": "Spiking Heidelberg Digits",
        "source_license": "CC BY 4.0",
        "source_url": "https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/",
        "labels_included": False,
        "samples": len(audit_ids),
        "time_bins": train_store.time_bins,
        "input_channels": train_store.input_channels,
        "packed_bit_order": "little",
        "channel_zero": "least-significant bit",
        "split": "certificate-audit",
    }
    audit_npz = write_npz(
        common / "audit_inputs.npz",
        {
            "audit_indices": audit_indices,
            "packed_spikes": audit_packed,
            "sample_ids": audit_ids,
            "metadata": json_scalar(audit_metadata),
        },
    )
    audit_mem = write_spike_mem(
        common / "audit_input_spikes.mem", audit_frames, train_store.input_channels
    )
    sample_index = common / "audit_samples.csv"
    _write_sample_index(sample_index, audit_ids, train_store.time_bins)

    semantics = primary_semantic_conditions()
    reference = semantics["reference"]
    target = semantics["floor_rounding_saturation"]
    semantics_dir = output / "semantics"
    write_json(semantics_dir / "reference.json", reference.to_dict())
    write_json(semantics_dir / "target_floor_q8q16.json", target.to_dict())
    emulator = TorchEmulator(device=args.device)

    seed_manifests: list[dict[str, Any]] = []
    for seed in args.seeds:
        source_dir = source_root / f"seed_{seed}"
        repair_dir = (
            repair_root
            / f"seed_{seed}"
            / "floor_rounding_saturation"
            / "certificate_directed"
        )
        source_path = source_dir / "model.npz"
        repaired_path = repair_dir / "repaired_model.npz"
        source_model = DenseRecurrentSNN.load(source_path)
        repaired_model = DenseRecurrentSNN.load(repaired_path)
        seed_output = output / "seeds" / str(seed)
        source_output = seed_output / "unrepaired"
        repaired_output = seed_output / "repaired"
        source_copy = source_output / "original_model.npz"
        repaired_copy = repaired_output / "original_model.npz"
        _copy_new(source_path, source_copy)
        _copy_new(repaired_path, repaired_copy)
        deployed_source, source_leak_q24 = _deployment_model(source_model, target)
        deployed_repaired, repaired_leak_q24 = _deployment_model(
            repaired_model, target
        )
        deployed_source_path = source_output / "deployed_model.npz"
        deployed_repaired_path = repaired_output / "deployed_model.npz"
        deployed_source.save(deployed_source_path)
        deployed_repaired.save(deployed_repaired_path)
        source_parameter_files = _export_parameters(
            deployed_source,
            source_output,
            target,
            leak_q24=source_leak_q24,
        )
        repaired_parameter_files = _export_parameters(
            deployed_repaired,
            repaired_output,
            target,
            leak_q24=repaired_leak_q24,
        )

        reference_predictions, reference_logits = _batched_trace(
            emulator,
            source_model,
            audit_frames,
            reference,
            args.batch_size,
        )
        target_predictions, target_logits = _batched_trace(
            emulator,
            deployed_source,
            audit_frames,
            target,
            args.batch_size,
        )
        repaired_predictions, repaired_logits = _batched_trace(
            emulator,
            deployed_repaired,
            audit_frames,
            target,
            args.batch_size,
        )
        original_target_predictions, _ = _batched_trace(
            emulator,
            source_model,
            audit_frames,
            target,
            args.batch_size,
        )
        original_repaired_predictions, _ = _batched_trace(
            emulator,
            repaired_model,
            audit_frames,
            target,
            args.batch_size,
        )
        golden_audit = write_npz(
            seed_output / "golden_audit.npz",
            {
                "sample_ids": audit_ids,
                "reference_predictions": reference_predictions,
                "unrepaired_original_software_predictions": original_target_predictions,
                "repaired_original_software_predictions": original_repaired_predictions,
                "unrepaired_emulator_predictions": target_predictions,
                "repaired_emulator_predictions": repaired_predictions,
                "reference_final_logits_float64": reference_logits,
                "unrepaired_final_logits_q16": _state_integers(target_logits, 8),
                "repaired_final_logits_q16": _state_integers(repaired_logits, 8),
                "metadata": json_scalar(
                    {
                        "schema_version": "SHDHardwareGoldenAudit/v1",
                        "reference_semantics": "semantics/reference.json",
                        "target_semantics": "semantics/target_floor_q8q16.json",
                        "source_model": "unrepaired/original_model.npz",
                        "repaired_model": "repaired/original_model.npz",
                        "parameter_mapping": "state parameters floor Q16; leak reciprocal floor unsigned Q0.24",
                    }
                ),
            },
        )
        smoke_frames = audit_frames[:SMOKE_SAMPLES]
        source_smoke = emulator.run(deployed_source, smoke_frames, target).numpy()
        repaired_smoke = emulator.run(deployed_repaired, smoke_frames, target).numpy()
        source_integer = run_floor_q8q16_integer(
            smoke_frames,
            input_weights_q8=fixed_integer(deployed_source.input_weights, target.weight_format),
            recurrent_weights_q8=fixed_integer(deployed_source.recurrent_weights, target.weight_format),
            output_weights_q8=fixed_integer(deployed_source.output_weights, target.weight_format),
            bias_q16=fixed_integer(deployed_source.bias, target.state_format),
            threshold_q16=fixed_integer(deployed_source.threshold, target.state_format),
            leak_reciprocal_q24_values=source_leak_q24,
        )
        repaired_integer = run_floor_q8q16_integer(
            smoke_frames,
            input_weights_q8=fixed_integer(deployed_repaired.input_weights, target.weight_format),
            recurrent_weights_q8=fixed_integer(deployed_repaired.recurrent_weights, target.weight_format),
            output_weights_q8=fixed_integer(deployed_repaired.output_weights, target.weight_format),
            bias_q16=fixed_integer(deployed_repaired.bias, target.state_format),
            threshold_q16=fixed_integer(deployed_repaired.threshold, target.state_format),
            leak_reciprocal_q24_values=repaired_leak_q24,
        )
        expected_source = {
            "membrane_q16": _state_integers(source_smoke.membrane, 8),
            "hidden_spikes": source_smoke.spikes.astype(np.uint8),
            "logits_over_time_q16": _state_integers(source_smoke.logits_over_time, 8),
            "final_logits_q16": _state_integers(source_smoke.final_logits, 8),
            "predictions": source_smoke.predictions.astype(np.int16),
        }
        expected_repaired = {
            "membrane_q16": _state_integers(repaired_smoke.membrane, 8),
            "hidden_spikes": repaired_smoke.spikes.astype(np.uint8),
            "logits_over_time_q16": _state_integers(repaired_smoke.logits_over_time, 8),
            "final_logits_q16": _state_integers(repaired_smoke.final_logits, 8),
            "predictions": repaired_smoke.predictions.astype(np.int16),
        }
        for name, expected in expected_source.items():
            if not np.array_equal(expected, source_integer[name]):
                raise AssertionError(f"integer/source mismatch for {seed}/{name}")
        for name, expected in expected_repaired.items():
            if not np.array_equal(expected, repaired_integer[name]):
                raise AssertionError(f"integer/repaired mismatch for {seed}/{name}")
        golden_smoke = write_npz(
            seed_output / "golden_smoke_traces.npz",
            {
                "sample_ids": audit_ids[:SMOKE_SAMPLES],
                "unrepaired_membrane_q16": source_integer["membrane_q16"],
                "unrepaired_hidden_spikes": source_integer["hidden_spikes"],
                "unrepaired_logits_over_time_q16": source_integer["logits_over_time_q16"],
                "repaired_membrane_q16": repaired_integer["membrane_q16"],
                "repaired_hidden_spikes": repaired_integer["hidden_spikes"],
                "repaired_logits_over_time_q16": repaired_integer["logits_over_time_q16"],
            },
        )
        summary = {
            "schema_version": "SHDHardwareSeedBundle/v1",
            "seed": seed,
            "audit_samples": len(audit_ids),
            "source_model": "unrepaired/original_model.npz",
            "repaired_model": "repaired/original_model.npz",
            "deployed_source_model": "unrepaired/deployed_model.npz",
            "deployed_repaired_model": "repaired/deployed_model.npz",
            "reference_semantics": "../../semantics/reference.json",
            "target_semantics": "../../semantics/target_floor_q8q16.json",
            "unrepaired_reference_target_disagreements": int(
                np.count_nonzero(reference_predictions != target_predictions)
            ),
            "repaired_reference_target_disagreements": int(
                np.count_nonzero(reference_predictions != repaired_predictions)
            ),
            "unrepaired_mapping_prediction_changes": int(
                np.count_nonzero(original_target_predictions != target_predictions)
            ),
            "repaired_mapping_prediction_changes": int(
                np.count_nonzero(original_repaired_predictions != repaired_predictions)
            ),
            "files": [
                "unrepaired/original_model.npz",
                "unrepaired/deployed_model.npz",
                "repaired/original_model.npz",
                "repaired/deployed_model.npz",
                "golden_audit.npz",
                "golden_smoke_traces.npz",
                *[f"unrepaired/{name}" for name in source_parameter_files],
                *[f"repaired/{name}" for name in repaired_parameter_files],
            ],
            "upstream": {
                "training_artifact": f"SHD seed {seed} source model",
                "repair_artifact": f"SHD seed {seed} final certificate-directed repair",
            },
        }
        write_json(seed_output / "manifest.json", summary)
        seed_manifests.append(summary)
        print(
            f"seed={seed} unrepaired_disagreements="
            f"{summary['unrepaired_reference_target_disagreements']} "
            f"repaired_disagreements="
            f"{summary['repaired_reference_target_disagreements']}",
            flush=True,
        )

    top_manifest = {
        "schema_version": "SHDHardwareBundle/v1",
        "purpose": "PINES Virtex-7 SHD physical conformance and repair evaluation",
        "condition": "floor_rounding_saturation",
        "seeds": list(args.seeds),
        "audit_samples": len(audit_ids),
        "labels_included": False,
        "reference_semantics": "semantics/reference.json",
        "target_semantics": "semantics/target_floor_q8q16.json",
        "common_files": [
            "README.md",
            "DATA_LICENSE.md",
            "common/audit_inputs.npz",
            "common/audit_input_spikes.mem",
            "common/audit_samples.csv",
            "semantics/reference.json",
            "semantics/target_floor_q8q16.json",
        ],
        "seed_manifests": [
            f"seeds/{item['seed']}/manifest.json" for item in seed_manifests
        ],
        "dataset": {
            "name": "Spiking Heidelberg Digits",
            "license": "CC BY 4.0",
            "source_url": "https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/",
            "split": "certificate-audit",
        },
    }
    write_json(output / "manifest.json", top_manifest)


if __name__ == "__main__":
    main()
