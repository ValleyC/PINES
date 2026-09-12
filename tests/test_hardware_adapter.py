from __future__ import annotations

import json

import numpy as np
import pytest

from pines.adapters.hardware import load_hardware_capture


def _write_manifest(path, capture, *, independent=True, backend="virtex7"):
    path.write_text(
        json.dumps(
            {
                "schema_version": "HardwareRunManifest/v1",
                "backend": backend,
                "backend_serial": "test-board",
                "adapter_version": "test-1",
                "firmware_version": "test-firmware-1",
                "bitstream_file": "test.bit",
                "semantics_description": "floor Q8/Q16 target",
                "model_description": "small recurrent SNN",
                "dataset_name": "test samples",
                "seed": "7",
                "run_id": "run-1",
                "timestamp_utc": "2026-08-07T12:00:00Z",
                "capture_file": capture.name,
                "independent_pairing": independent,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("backend", ["virtex7", "spinnaker1", "spinnaker2"])
def test_hardware_capture_requires_unique_pairs(tmp_path, backend) -> None:
    capture = tmp_path / "capture.npz"
    manifest = tmp_path / "manifest.json"
    np.savez(
        capture,
        predictions=np.asarray([0, 1]),
        sample_ids=np.asarray(["a", "b"]),
        pair_ids=np.asarray(["pair-a", "pair-b"]),
    )
    _write_manifest(manifest, capture, backend=backend)
    predictions, sample_ids, record = load_hardware_capture(capture, manifest)
    assert predictions.tolist() == [0, 1]
    assert sample_ids.tolist() == ["a", "b"]
    assert record.backend == backend


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
