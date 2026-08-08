from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import sha256_file


@dataclass(frozen=True)
class HardwareRunManifest:
    schema_version: str
    backend: str
    backend_serial: str
    adapter_version: str
    firmware_hash: str
    bitstream_hash: str | None
    semantics_hash: str
    model_hash: str
    dataset_hash: str
    seed_hash: str
    run_id: str
    timestamp_utc: str
    capture_hash: str
    independent_pairing: bool

    def __post_init__(self) -> None:
        if self.schema_version != "HardwareRunManifest/v1":
            raise ValueError("unsupported hardware manifest")
        if self.backend not in {"spinnaker2", "virtex7"}:
            raise ValueError("backend must be spinnaker2 or virtex7")
        required = (
            self.backend_serial,
            self.adapter_version,
            self.firmware_hash,
            self.semantics_hash,
            self.model_hash,
            self.dataset_hash,
            self.seed_hash,
            self.run_id,
            self.timestamp_utc,
            self.capture_hash,
        )
        if any(not value for value in required):
            raise ValueError("hardware manifest contains an empty required field")
        if self.backend == "virtex7" and not self.bitstream_hash:
            raise ValueError("Virtex-7 evidence requires a bitstream hash")
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
    if sha256_file(capture_path) != manifest.capture_hash:
        raise ValueError("hardware capture hash does not match manifest")
    with np.load(capture_path, allow_pickle=False) as capture:
        predictions = np.asarray(capture["predictions"])
        sample_ids = np.asarray(capture["sample_ids"]).astype(str)
        pair_ids = np.asarray(capture["pair_ids"]).astype(str)
    if predictions.ndim != 1 or sample_ids.shape != predictions.shape:
        raise ValueError("capture arrays must be one-dimensional and shape-matched")
    if pair_ids.shape != predictions.shape or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("each primary hardware observation needs a unique pair_id")
    return predictions, sample_ids, manifest
