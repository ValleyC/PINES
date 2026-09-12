import importlib.util
import json
from pathlib import Path
import sys

scripts = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location("spinnaker_repeats", scripts / "analyze_spinnaker1_repeats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_variability_denominators_are_explicit_and_repeats_are_not_a_certificate():
    report = module.summarize({(1701,"original","a"):[0,0,1],
        (1701,"original","b"):[2,2,2], (1701,"original","c"):[4]})
    assert report["physical_certificate"] is False
    assert report["repeated_input_conditions"] == 2
    assert report["comparisons_to_first_execution"] == 4
    assert report["changed_predictions"] == 1
    assert report["change_rate_from_first"] == .25
    assert report["fraction_conditions_with_any_change"] == .5


def test_unrepeated_captures_do_not_imply_zero_variability():
    report = module.summarize({(1701,"original","a"):[0]})
    assert report["repeated_input_conditions"] == 0
    assert report["change_rate_from_first"] is None
    assert report["fraction_conditions_with_any_change"] is None


def test_shd_capture_predictions_are_grouped_by_input_and_condition(tmp_path):
    captures = []
    for run, prediction in enumerate([1, 1, 2]):
        capture = tmp_path / f"run_{run}"
        sample = capture / "input_00000"
        sample.mkdir(parents=True)
        (sample / "summary.json").write_text(json.dumps(dict(sample_id="shd-test-00001",
            rows=[dict(seed=1701, variant="original", prediction=prediction)])))
        (sample / "diagnostics.json").write_text(json.dumps(dict(late_spikes=[], messages=[])))
        captures.append(capture)
    report = module.analyze(captures, "shd")
    assert report["rows"][0]["predictions"] == [1, 1, 2]
    assert report["fraction_conditions_with_any_change"] == 1
    assert report["change_rate_from_first"] == .5


def test_partial_dvs_windows_are_not_repeated_classifications(tmp_path):
    captures = []
    for run in range(2):
        capture = tmp_path / f"run_{run}"
        sample = capture / "input_00000"
        sample.mkdir(parents=True)
        (capture / "config.json").write_text(json.dumps(dict(seed=1701)))
        (sample / "summary.json").write_text(json.dumps(dict(sample_id="dvs-train-00001",
            status="development_partial_windows", windows=[0])))
        captures.append(capture)
    report = module.analyze(captures, "dvs")
    assert report["repeated_input_conditions"] == 0
    assert report["change_rate_from_first"] is None


def test_embedded_shd_repeat_is_not_a_primary_observation(tmp_path):
    for folder, prediction in [("input_00000", 1), ("repeat_first", 2)]:
        sample = tmp_path / folder
        sample.mkdir()
        (sample / "summary.json").write_text(json.dumps(dict(sample_id="shd-test-00001",
            rows=[dict(seed=1701, variant="original", prediction=prediction)])))
        (sample / "diagnostics.json").write_text(json.dumps(dict(late_spikes=[], messages=[])))
    primary, _ = module.collect_shd([tmp_path])
    assert len(primary) == 1
    report = module.analyze([tmp_path], "shd")
    assert report["rows"][0]["predictions"] == [1, 2]
    assert report["comparisons_to_first_execution"] == 1


def test_embedded_dvs_repeat_requires_complete_four_window_recording(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(seed=1701)))
    for folder, prediction in [("input_00000", 3), ("repeat_first", 3)]:
        sample = tmp_path / folder
        sample.mkdir()
        (sample / "summary.json").write_text(json.dumps(dict(sample_id="dvs-test-00001",
            status="physical_classification_capture_completed", windows=[0, 1, 2, 3],
            predictions={"original": prediction})))
        for window in range(4):
            target = sample / f"window_{window}"
            target.mkdir()
            (target / "summary.json").write_text(json.dumps(dict(
                diagnostics=dict(late_spikes=[], messages=[]))))
    primary, _ = module.collect_dvs([tmp_path])
    assert len(primary) == 1
    report = module.analyze([tmp_path], "dvs")
    assert report["rows"][0]["predictions"] == [3, 3]
    assert report["change_rate_from_first"] == 0
    path = tmp_path / "repeat_first" / "summary.json"
    partial = json.loads(path.read_text())
    partial.update(status="development_partial_windows", windows=[0, 1])
    path.write_text(json.dumps(partial))
    assert module.analyze([tmp_path], "dvs")["repeated_input_conditions"] == 0
