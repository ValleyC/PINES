"""Check capture orchestration without treating mocked execution as hardware."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


scripts = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location("dvs_reuse_runner", scripts / "run_spinnaker1_dvs.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
finally:
    sys.path.pop(0)


def test_complete_recording_repeat_has_a_separate_folder_after_primary_inputs():
    assert list(runner.recording_schedule(10, 2)) == [(10, "input_00010"), (11, "input_00011")]
    assert list(runner.recording_schedule(10, 2, True)) == [
        (10, "input_00010"), (11, "input_00011"), (10, "repeat_first")]


def test_reuse_loads_each_model_once_and_resets_before_replacing_inputs(monkeypatch, tmp_path):
    actions, sources, captured = [], [], []

    class Population:
        def __init__(self, size, spike_times, label):
            self.size, self.spike_times = size, spike_times
            self.label = label
            sources.append(self)

        def set_max_atoms_per_core(self, n):
            self.per_core = n

        def set(self, spike_times):
            actions.append(("set", self.label))
            self.spike_times = spike_times

    sim = SimpleNamespace(
        setup=lambda **kw: actions.append(("setup", kw)),
        Population=Population, SpikeSourceArray=lambda spike_times: spike_times,
        reset=lambda: actions.append(("reset",)),
        run=lambda steps: actions.append(("run", steps)),
        end=lambda: actions.append(("end",)))
    monkeypatch.setitem(sys.modules, "spinn_utilities.config_holder", SimpleNamespace(
        get_config_bool=lambda *a: False, get_config_int=lambda *a: 4,
        set_config=lambda *a: None))
    monkeypatch.setattr(runner, "build_network", lambda *a: actions.append(("build",)) or object())

    def capture(sim, models, networks, horizon, output, args, started, **kw):
        captured.append(dict(horizon=horizon, **kw))
        return {}

    monkeypatch.setattr(runner, "capture_window", capture)
    args = SimpleNamespace(time_scale_factor=100, source_neurons_per_core=32)
    allocation = runner.AlignedDVSAllocation(sim, [({}, {}), ({}, {})], args, object())
    event = np.zeros((60, 2, 32, 32), dtype=np.uint8)
    event[0, 0, 0, 0] = 1
    first = allocation.run(event, tmp_path)
    event[0, 0, 0, 0] = 0
    event[1, 0, 0, 1] = 1
    second = allocation.run(event, tmp_path)
    allocation.close()
    allocation.close()

    assert sum(a[0] == "setup" for a in actions) == 1
    assert sum(a[0] == "build" for a in actions) == 2
    assert sum(a[0] == "end" for a in actions) == 1
    assert [a for a in actions if a[0] == "run"] == [("run", 80), ("run", 80)]
    reset = actions.index(("reset",))
    assert actions[reset:reset+4] == [("reset",), ("set", "dvs_input"), ("set", "bias_clock"), ("run", 80)]
    assert list(sources[0].spike_times[0]) == []
    assert list(sources[0].spike_times[1]) == [3.]
    assert captured == [dict(horizon=60, clear=True, include_machine=False)] * 2
    assert first["allocation_window"] == 0 and second["allocation_window"] == 1
    assert first["run_steps"] == 80 and first["observation_steps"] == 60
