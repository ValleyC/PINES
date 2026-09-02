from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

@dataclass(frozen=True)
class HardwareRunManifest:
    schema_version: str
    backend: str
    backend_serial: str
    adapter_version: str
    firmware_version: str
    bitstream_file: str | None
    semantics_description: str
    model_description: str
    dataset_name: str
    seed: str
    run_id: str
    timestamp_utc: str
    capture_file: str
    independent_pairing: bool

    def __post_init__(self) -> None:
        if self.schema_version != "HardwareRunManifest/v1":
            raise ValueError("unsupported hardware manifest")
        if self.backend not in {"spinnaker2", "virtex7"}:
            raise ValueError("backend must be spinnaker2 or virtex7")
        required = (
            self.backend_serial,
            self.adapter_version,
            self.firmware_version,
            self.semantics_description,
            self.model_description,
            self.dataset_name,
            self.seed,
            self.run_id,
            self.timestamp_utc,
            self.capture_file,
        )
        if any(not value for value in required):
            raise ValueError("hardware manifest contains an empty required field")
        if self.backend == "virtex7" and not self.bitstream_file:
            raise ValueError("Virtex-7 evidence requires the bitstream filename")
        if not self.independent_pairing:
            raise ValueError(
                "primary physical certificate requires independent (input, run) pairs"
            )

    @classmethod
    def load(cls, path: str | Path) -> "HardwareRunManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


def load_hardware_capture(
    capture_path: str | Path,
    manifest_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, HardwareRunManifest]:
    capture_path = Path(capture_path)
    manifest = HardwareRunManifest.load(manifest_path)
    with np.load(capture_path, allow_pickle=False) as capture:
        predictions = np.asarray(capture["predictions"])
        sample_ids = np.asarray(capture["sample_ids"]).astype(str)
        pair_ids = np.asarray(capture["pair_ids"]).astype(str)
    if predictions.ndim != 1 or sample_ids.shape != predictions.shape:
        raise ValueError("capture arrays must be one-dimensional and shape-matched")
    if pair_ids.shape != predictions.shape or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("each primary hardware observation needs a unique pair_id")
    return predictions, sample_ids, manifest
