"""Synthetic diagnostic fixtures do not represent measured hardware results."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("dvs_development_analysis",
    Path(__file__).resolve().parents[1] / "experiments/analyze_spinnaker1_dvs_development.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.mark.parametrize("windows", [1, 4])
def test_only_four_windows_make_a_recording_comparison(tmp_path, windows):
    capture, bundle = tmp_path / "capture", tmp_path / "bundle"
    capture.mkdir()
    (capture / "config.json").write_text(json.dumps(dict(seed=1701, variants=["original"])))
    traces = bundle / "development_traces/1701/original"
    traces.mkdir(parents=True)
    expected_logits = np.array([[2., 0.]] * 4)
    np.savez(traces / "input_00000.npz", sample_id="development-input-0",
             window_logits=expected_logits, prediction=0)
    model = bundle / "models/1701"
    model.mkdir(parents=True)
    np.savez(model / "original.npz", metadata=json.dumps(dict(aggregation_temperature=.5)))
    for window in range(windows):
        folder = capture / "input_00000" / f"window_{window}"
        folder.mkdir(parents=True)
        # The first window disagrees, but the other three dominate the recording.
        logits = np.array([0., 3.]) if window == 0 else np.array([5., 0.])
        np.savez(folder / "original.npz", window_logits=logits)
    report = analysis.analyze(capture, bundle)
    assert report["physical_certificate"] is False
    assert len(report["rows"]) == windows
    assert report["rows"][0]["physical_window_prediction"] == 1
    assert report["completed_recording_conditions"] == (1 if windows == 4 else 0)
    if windows == 4:
        assert report["recordings"][0]["physical_prediction"] == 0
        assert report["recordings"][0]["emulator_prediction"] == 0
        assert report["recordings"][0]["prediction_agrees"] is True
