import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

script = Path(__file__).resolve().parents[1] / "experiments/analyze_virtex7_shd.py"
spec = importlib.util.spec_from_file_location("virtex_analysis", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    audit, captures = tmp_path / "audit", tmp_path / "captures"
    audit.mkdir()
    folder = captures / "1701/unrepaired"
    folder.mkdir(parents=True)
    (audit / "semantic_audit.json").write_text(json.dumps(dict(rows=[dict(seed=1701, variant="unrepaired", semantic_upper=.1)],
        population="SHD held-out test pool", confidence=.95, paper_cells=40, alpha_per_term=.000625)))
    np.savez(audit / "paired_predictions.npz", canary_sample_ids=["a","b"],
        seed1701_canary_source=[0,1], seed1701_canary_unrepaired_emulator=[0,1])
    np.savez(folder / "capture.npz", sample_ids=["b","a"], pair_ids=["run-b","run-a"], predictions=[0,0])
    manifest = dict(schema_version="HardwareRunManifest/v1", backend="virtex7", backend_serial="test-board",
        adapter_version="test", firmware_version="test", bitstream_file="test.bit", semantics_description="test integer target",
        model_description="test model", dataset_name="SHD", seed="1701", run_id="test", timestamp_utc="2026-09-12T00:00:00Z",
        capture_file="capture.npz", independent_pairing=True)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return captures, audit, folder


def test_physical_predictions_align_by_sample_id_before_composition(tmp_path):
    captures, audit, _ = fixture(tmp_path)
    report = module.analyze(captures, audit)
    row = report["rows"][0]
    assert row["hardware_disagreements"] == 1
    assert row["total_upper"] == min(1., .1+row["conformance_upper"])
    assert "absolute_accuracy_change" not in row


def test_incomplete_capture_does_not_read_evaluation_labels(tmp_path):
    _, audit, _ = fixture(tmp_path)
    report = module.analyze(tmp_path / "not_captured", audit, tmp_path / "no_labels.npz")
    assert report["physical_certificate"] is False
    assert report["observations"] == 0
    assert report["expected_observations"] == 2


def test_accuracy_is_computed_only_with_explicit_post_capture_labels(tmp_path):
    captures, audit, _ = fixture(tmp_path)
    labels = tmp_path / "labels.npz"
    np.savez(labels, sample_ids=["b","a"], labels=[1,0])
    row = module.analyze(captures, audit, labels)["rows"][0]
    assert row["source_accuracy"] == 1
    assert row["hardware_accuracy"] == .5
    assert row["absolute_accuracy_change"] == .5


def test_duplicate_inputs_are_not_independent_observations(tmp_path):
    captures, audit, folder = fixture(tmp_path)
    np.savez(folder / "capture.npz", sample_ids=["a","a"], pair_ids=["run-1","run-2"], predictions=[0,0])
    with pytest.raises(ValueError, match="occur once"):
        module.analyze(captures, audit)
