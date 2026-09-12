import importlib.util
from pathlib import Path
import sys

import numpy as np

scripts = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location("dvs_reset_probe", scripts / "probe_spinnaker1_dvs_reset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_schedule_preserves_two_full_recordings_and_separates_repeated_windows():
    schedule = module.development_schedule()
    assert len(schedule) == 10
    for index in range(2):
        assert [c["window"] for c in schedule if c["input_index"] == index and not c["repeat"]] == [0, 1, 2, 3]
    assert [(c["input_index"], c["window"]) for c in schedule if c["repeat"]] == [(0, 0), (0, 0)]


def test_repeat_comparison_exposes_spike_changes_even_when_logits_match():
    first = {name: np.zeros((2, 3)) for name in ("conv1_spikes", "conv2_spikes", "hidden_spikes", "window_logits")}
    later = {name: value.copy() for name, value in first.items()}
    later["hidden_spikes"][0, 0] = 1
    comparison = module.compare_arrays(later, first)
    assert comparison["window_logits"]["equal"]
    assert comparison["hidden_spikes"] == dict(equal=False, differing_entries=1)
