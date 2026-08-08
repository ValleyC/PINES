from __future__ import annotations

import json

import numpy as np
import pytest

from transportcert.adapters.hardware import load_hardware_capture
from transportcert.artifacts import sha256_file


def _write_manifest(path, capture, *, independent=True):
    path.write_text(
        json.dumps(
            {
                "schema_version": "HardwareRunManifest/v1",
                "backend": "virtex7",
                "backend_serial": "test-board",
                "adapter_version": "test-1",
                "firmware_hash": "a" * 64,
                "bitstream_hash": "b" * 64,
                "semantics_hash": "c" * 64,
                "model_hash": "d" * 64,
                "dataset_hash": "e" * 64,
                "seed_hash": "f" * 64,
                "run_id": "run-1",
                "timestamp_utc": "2026-08-07T12:00:00Z",
                "capture_hash": sha256_file(capture),
                "independent_pairing": independent,
            }
        ),
        encoding="utf-8",
    )


def test_hardware_capture_requires_unique_pairs(tmp_path) -> None:
    capture = tmp_path / "capture.npz"
    manifest = tmp_path / "manifest.json"
    np.savez(
        capture,
        predictions=np.asarray([0, 1]),
        sample_ids=np.asarray(["a", "b"]),
        pair_ids=np.asarray(["pair-a", "pair-b"]),
    )
    _write_manifest(manifest, capture)
    predictions, sample_ids, record = load_hardware_capture(capture, manifest)
    assert predictions.tolist() == [0, 1]
    assert sample_ids.tolist() == ["a", "b"]
    assert record.backend == "virtex7"


def test_primary_capture_rejects_nonindependent_runs(tmp_path) -> None:
    capture = tmp_path / "capture.npz"
    manifest = tmp_path / "manifest.json"
    np.savez(
        capture,
        predictions=np.asarray([0]),
        sample_ids=np.asarray(["a"]),
        pair_ids=np.asarray(["pair-a"]),
    )
    _write_manifest(manifest, capture, independent=False)
    with pytest.raises(ValueError, match="independent"):
        load_hardware_capture(capture, manifest)
