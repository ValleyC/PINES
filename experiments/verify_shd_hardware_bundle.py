from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pines.hardware_bundle import destination_major, run_floor_q8q16_integer


def _read_mem(path: Path, bits: int) -> np.ndarray:
    unsigned = np.asarray(
        [int(line, 16) for line in path.read_text(encoding="ascii").splitlines()],
        dtype=np.int64,
    )
    sign = 1 << (bits - 1)
    signed = np.where(unsigned & sign, unsigned - (1 << bits), unsigned)
    return signed


def _check_files(root: Path, entries: list[str]) -> None:
    for relative in entries:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", default="hardware/bundles/shd_floor_q8q16_v1"
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    bundle = repository / args.bundle
    with (bundle / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["schema_version"] != "SHDHardwareBundle/v1":
        raise ValueError("unsupported hardware bundle schema")
    _check_files(bundle, manifest["common_files"])
    with np.load(bundle / "common" / "audit_inputs.npz", allow_pickle=False) as data:
        packed = np.asarray(data["packed_spikes"], dtype=np.uint8)
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
    expected_shape = (manifest["audit_samples"], 50, 88)
    if packed.shape != expected_shape or sample_ids.shape != (expected_shape[0],):
        raise ValueError("audit input shape differs from the frozen SHD protocol")
    smoke_frames = np.unpackbits(packed[:8], axis=-1, bitorder="little")[
        ..., :700
    ]

    for seed in manifest["seeds"]:
        relative_manifest = Path("seeds") / str(seed) / "manifest.json"
        with (bundle / relative_manifest).open("r", encoding="utf-8") as handle:
            seed_manifest = json.load(handle)
        seed_root = bundle / "seeds" / str(seed)
        _check_files(seed_root, seed_manifest["files"])
        with np.load(
            seed_root / "golden_audit.npz", allow_pickle=False
        ) as golden:
            if not np.array_equal(golden["sample_ids"].astype(str), sample_ids):
                raise ValueError(f"golden sample ordering differs for seed {seed}")
            for name in (
                "reference_predictions",
                "unrepaired_emulator_predictions",
                "repaired_emulator_predictions",
            ):
                if golden[name].shape != (manifest["audit_samples"],):
                    raise ValueError(f"invalid {name} shape for seed {seed}")
            audit_predictions = {
                "unrepaired": np.asarray(
                    golden["unrepaired_emulator_predictions"], dtype=np.int16
                ),
                "repaired": np.asarray(
                    golden["repaired_emulator_predictions"], dtype=np.int16
                ),
            }
        with np.load(
            seed_root / "golden_smoke_traces.npz", allow_pickle=False
        ) as golden_smoke:
            if not np.array_equal(
                golden_smoke["sample_ids"].astype(str), sample_ids[:8]
            ):
                raise ValueError(f"smoke sample ordering differs for seed {seed}")
            smoke = {name: np.asarray(golden_smoke[name]) for name in golden_smoke.files}
        for variant in ("unrepaired", "repaired"):
            variant_root = seed_root / variant
            with np.load(
                variant_root / "parameters_int.npz", allow_pickle=False
            ) as parameters:
                matrices = {
                    "input_weights": parameters["input_weights_q8"],
                    "recurrent_weights": parameters["recurrent_weights_q8"],
                    "output_weights": parameters["output_weights_q8"],
                }
                oracle = run_floor_q8q16_integer(
                    smoke_frames,
                    input_weights_q8=parameters["input_weights_q8"],
                    recurrent_weights_q8=parameters["recurrent_weights_q8"],
                    output_weights_q8=parameters["output_weights_q8"],
                    bias_q16=parameters["bias_q16"],
                    threshold_q16=parameters["threshold_q16"],
                    leak_reciprocal_q24_values=parameters[
                        "leak_reciprocal_q24"
                    ],
                )
                vectors = {
                    "bias": (parameters["bias_q16"], 16),
                    "threshold": (parameters["threshold_q16"], 16),
                    "reset_value": (parameters["reset_value_q16"], 16),
                    "leak_reciprocal_q24": (
                        parameters["leak_reciprocal_q24"],
                        32,
                    ),
                }
            for name, matrix in matrices.items():
                observed = _read_mem(variant_root / f"{name}.mem", 8)
                if not np.array_equal(observed, destination_major(matrix)):
                    raise ValueError(
                        f"{seed}/{variant}/{name}.mem has the wrong layout"
                    )
            for name, (expected, bits) in vectors.items():
                observed = _read_mem(variant_root / f"{name}.mem", bits)
                if not np.array_equal(observed, np.asarray(expected).reshape(-1)):
                    raise ValueError(f"{seed}/{variant}/{name}.mem differs from NPZ")
            for field in ("membrane_q16", "hidden_spikes", "logits_over_time_q16"):
                if not np.array_equal(smoke[f"{variant}_{field}"], oracle[field]):
                    raise ValueError(
                        f"integer oracle mismatch for {seed}/{variant}/{field}"
                    )
            if not np.array_equal(
                oracle["predictions"], audit_predictions[variant][:8]
            ):
                raise ValueError(
                    f"smoke and audit predictions disagree for {seed}/{variant}"
                )
    print(
        f"Verified SHD hardware bundle: {len(manifest['seeds'])} seeds, "
        f"{manifest['audit_samples']} unlabeled inputs per seed."
    )


if __name__ == "__main__":
    main()
