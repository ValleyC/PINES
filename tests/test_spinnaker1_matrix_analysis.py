import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("matrix_analysis",
    Path(__file__).resolve().parents[1] / "experiments/analyze_spinnaker1_matrix.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    capture = tmp_path / "capture"
    capture.mkdir()
    rows = []
    arrays = {"canary_sample_ids": np.array(["a", "b"])}
    for seed in (1701, 2718, 3141, 5772, 8119):
        arrays[f"seed{seed}_canary_source"] = [0, 1]
        for variant in ("original", "reset_repaired"):
            rows.append(dict(seed=seed, variant=variant, semantic_upper=0.1))
            arrays[f"seed{seed}_canary_{variant}_emulator"] = [0, 1]
    np.savez(audit / "paired_predictions.npz", **arrays)
    (audit / "semantic_audit.json").write_text(json.dumps(dict(rows=rows,
        alpha_per_term=0.000625, confidence=0.95, paper_cells=40, population="test")))
    for index, sample in enumerate(("a", "b")):
        folder = capture / f"input_{index:05d}"
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps(dict(sample_id=sample,
            rows=[dict(seed=r["seed"], variant=r["variant"], prediction=index,
                       emulator_prediction=index) for r in rows])))
        (folder / "diagnostics.json").write_text('{"late_spikes":[],"messages":[]}')
    return capture, audit


def test_completed_matrix_composes_the_two_terms(tmp_path):
    capture, audit = fixture(tmp_path)
    report = module.analyze([capture], audit)
    assert report["physical_certificate"] is True
    assert len(report["rows"]) == 10
    assert all(r["hardware_disagreements"] == 0 for r in report["rows"])
    assert all(r["total_upper"] == min(1, .1+r["conformance_upper"]) for r in report["rows"])
    assert "absolute_accuracy_change" not in report["rows"][0]


def test_report_preserves_reused_execution_profile_and_runtime(tmp_path):
    capture, audit = fixture(tmp_path)
    profile = dict(sample_range=[0, 2], execution_profile="aligned_reset_reuse",
                   spike_reader="numpy-current", packages={"sPyNNaker": "1!7.4.2"})
    (capture / "config.json").write_text(json.dumps(profile))
    report = module.analyze([capture], audit)
    assert report["capture_profiles"] == [dict(capture_index=0, **profile)]
    assert report["inputs_with_late_spikes"] == 0
    assert "allocation pairs" not in " ".join(report["assumptions"])


def test_incomplete_matrix_never_loads_labels_or_issues_certificate(tmp_path):
    capture, audit = fixture(tmp_path)
    (capture / "input_00001/summary.json").unlink()
    report = module.analyze([capture], audit, tmp_path / "nonexistent_labels.npz")
    assert report["physical_certificate"] is False
    assert report["observations"] == 10
    assert report["expected_observations"] == 20


def test_duplicate_primary_pairs_are_not_independent_observations(tmp_path):
    capture, audit = fixture(tmp_path)
    with pytest.raises(ValueError, match="Duplicate primary"):
        module.analyze([capture, capture], audit)


def test_labeled_evaluation_uses_source_not_emulator_as_accuracy_reference(tmp_path):
    capture, audit = fixture(tmp_path)
    labels = tmp_path / "labels.npz"
    np.savez(labels, sample_ids=["b", "a"], labels=[0, 0])
    report = module.analyze([capture], audit, labels)
    assert all(r["source_accuracy"] == .5 for r in report["rows"])
    assert all(r["absolute_accuracy_change"] == 0 for r in report["rows"])
