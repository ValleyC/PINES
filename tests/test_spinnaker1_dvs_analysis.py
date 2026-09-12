import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def script(name):
    scripts = Path(__file__).resolve().parents[1] / "experiments"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


analysis = script("analyze_spinnaker1_dvs_matrix")


def fixture(tmp_path):
    audit, capture = tmp_path / "audit", tmp_path / "capture"
    audit.mkdir()
    capture.mkdir()
    variants = ("original", "floor_repaired")
    rows = [dict(seed=1701, variant=v, semantic_upper=.1) for v in variants]
    (capture / "config.json").write_text(json.dumps(dict(seed=1701, variants=variants)))
    arrays = dict(canary_sample_ids=np.array(["dvs-test-00004", "dvs-test-00001"]),
                  canary_dataset_indices=np.array([4,1]), seed1701_canary_source=[0,1])
    for variant in variants:
        arrays[f"seed1701_canary_{variant}_emulator"] = [0,1]
    np.savez(audit / "paired_predictions.npz", **arrays)
    (audit / "semantic_audit.json").write_text(json.dumps(dict(rows=rows,
        alpha_per_term=.000625, confidence=.95, paper_cells=40, population="DVS test")))
    for index, sample in enumerate(arrays["canary_sample_ids"]):
        folder = capture / f"input_{index:05d}"
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps(dict(sample_id=sample,
            windows=[0,1,2,3], status="physical_classification_capture_completed",
            predictions={v:index for v in variants})))
        for window in range(4):
            target = folder / f"window_{window}"
            target.mkdir()
            (target / "summary.json").write_text(json.dumps(dict(
                diagnostics=dict(late_spikes=[], messages=[]))))
    return capture, audit


def test_four_windows_count_as_one_observation(tmp_path):
    capture, audit = fixture(tmp_path)
    report = analysis.analyze([capture], audit)
    assert report["physical_certificate"] is True
    assert len(report["rows"]) == 2
    assert all(row["canary_samples"] == 2 for row in report["rows"])
    assert all(row["total_upper"] == min(1., .1+row["conformance_upper"]) for row in report["rows"])
    assert "absolute_accuracy_change" not in report["rows"][0]
    assert report["capture_profiles"][0]["seed"] == 1701


def test_partial_windows_do_not_issue_a_certificate_or_read_labels(tmp_path):
    capture, audit = fixture(tmp_path)
    path = capture / "input_00001/summary.json"
    record = json.loads(path.read_text())
    record.update(status="development_partial_windows", windows=[0])
    path.write_text(json.dumps(record))
    report = analysis.analyze([capture], audit, tmp_path / "missing_labels.npz")
    assert report["physical_certificate"] is False
    assert report["observations"] == 2
    assert report["expected_observations"] == 4


def test_repeated_captures_are_not_new_independent_recordings(tmp_path):
    capture, audit = fixture(tmp_path)
    with pytest.raises(ValueError, match="Duplicate primary"):
        analysis.analyze([capture, capture], audit)


def test_labeled_evaluation_uses_dataset_rows_not_class_encoded_filenames(tmp_path):
    capture, audit = fixture(tmp_path)
    labels = tmp_path / "labels.npz"
    np.savez(labels, labels=[5,0,5,5,0])
    report = analysis.analyze([capture], audit, labels)
    assert all(row["source_accuracy"] == .5 for row in report["rows"])
    assert all(row["absolute_accuracy_change"] == 0 for row in report["rows"])


def test_source_audit_reconstructs_the_original_float32_executor():
    torch = pytest.importorskip("torch")
    from pines.benchmarks.dvs_gesture import DVSGestureTrainConfig, build_dvs_conv_srnn
    from pines.semantics import ExecutionSemantics, NumericFormat
    audit = script("prepare_spinnaker1_dvs_audit")
    torch.manual_seed(41)
    config = DVSGestureTrainConfig(conv1_channels=2, conv2_channels=3, hidden_size=4,
        tau_mem=4, threshold=.125, aggregation_temperature=.5)
    model = build_dvs_conv_srnn(15, 15, config, output_size=3).eval()
    state = {key:value.detach().numpy() for key,value in model.state_dict().items()}
    meta = dict(sensor_width=15, sensor_height=15, tau_mem=4, threshold=.125, aggregation_temperature=.5)
    events = np.random.default_rng(40).integers(0,2,(2,4,6,2,15,15), dtype=np.uint8)
    packed = np.packbits(events.reshape(2,4,6,-1), axis=-1, bitorder="little")
    np.testing.assert_array_equal(audit.unpack_recordings(packed, meta), events)
    semantics = ExecutionSemantics(state_format=NumericFormat("float32"), weight_format=NumericFormat("float32"))
    with torch.no_grad():
        logits = model(torch.tensor(events.reshape(8,6,2,15,15), dtype=torch.float32), semantics).reshape(2,4,3)
        expected = torch.softmax(logits/.5, dim=-1).mean(dim=1).argmax(dim=1).numpy()
    np.testing.assert_array_equal(audit.source_predictions(state, meta, packed, "cpu"), expected)
    assert audit.mapped_predictions(state, meta, packed, "cpu").shape == (2,)
