import importlib.util
import sys
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "experiments" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_spinnaker1_shd")
probe = load_script("spinnaker1_probe")
analyzer = load_script("analyze_spinnaker1_shd")
sys.modules["run_spinnaker1_shd"] = runner
matrix_runner = load_script("run_spinnaker1_matrix")


class SpikeTrain(np.ndarray):
    def __new__(cls, times, source_index):
        obj = np.asarray(times, dtype=float).view(cls)
        obj.annotations = {"source_index": source_index}
        return obj


def test_clocked_bias_starts_with_inputs_and_stops_at_horizon():
    trains = runner.input_spike_times(np.array([[0, 1], [1, 0], [0, 0]]), 1.0)
    for actual, expected in zip(trains, [[1.0], [0.0], [0.0, 1.0, 2.0]]):
        np.testing.assert_array_equal(actual, expected)
    assert trains.shape == (3,)
    np.testing.assert_array_equal(trains[np.array([0, 2])][1], [0.0, 1.0, 2.0])


def test_readout_uses_neuron_ids_and_exact_finite_window():
    # Reverse the list to check annotation-based neuron ordering. Spikes at
    # physical t=0 and t=4 lie outside logical t=0..2 for latency=1.
    seg = SimpleNamespace(spiketrains=[SpikeTrain([2.0], 1), SpikeTrain([0.0, 1.0, 3.0, 4.0], 0)])
    spikes, logits, raw = runner.spike_readout(seg, 2, 3, 1, 1.0, np.array([[2., -1.], [-3., 2.]]))
    np.testing.assert_array_equal(spikes, [[1, 0], [0, 1], [1, 0]])
    np.testing.assert_array_equal(logits, [1, 0])
    assert len(raw) == 5
    array_spikes, array_logits, array_raw = runner.spike_rows_readout(
        raw, 2, 3, 1, 1., np.array([[2., -1.], [-3., 2.]]))
    np.testing.assert_array_equal(array_spikes, spikes)
    np.testing.assert_array_equal(array_logits, logits)
    np.testing.assert_array_equal(array_raw, raw)


def test_array_reader_handles_empty_and_population_local_indices():
    for raw in (np.empty((0, 0)), np.array([[3, 4.], [0, 2.], [3, 2.], [3, 9.]])):
        population = SimpleNamespace(size=4, spinnaker_get_data=lambda name: raw)
        spike_trains = [SpikeTrain(raw[raw[:, 0] == index, 1] if raw.size else [], index) for index in range(4)]
        weights = np.arange(12).reshape(4, 3)
        expected = runner.spike_readout(SimpleNamespace(spiketrains=spike_trains), 4, 5, 2, 1., weights)
        observed = runner.current_spike_readout(population, 5, 2, 1., weights)
        np.testing.assert_array_equal(observed[0], expected[0])
        np.testing.assert_array_equal(observed[1], expected[1])


def test_probe_covers_signed_weights_reset_bias_delay_and_recurrence():
    specs = probe.probe_specification()
    assert len({s["name"] for s in specs}) == 8
    assert any(s["weight"] < 0 for s in specs)
    assert any(s["reset"] != 0 for s in specs)
    assert any(s["delay"] == 2 for s in specs)
    assert any(s["bias"] != 0 for s in specs)
    assert "recurrent" in {s["name"] for s in specs}


def test_source_update_shim_keeps_exact_ticks_and_supports_vertex_selection():
    ticks = [np.array([0, 2], dtype=np.int64), np.array([], dtype=np.int64),
             np.array([1, 3, 4], dtype=np.int64)]
    indexed = runner.source_tick_array(ticks)
    assert indexed.shape == (3,)
    for a, b in zip(indexed[np.array([0, 2])], [ticks[0], ticks[2]]):
        np.testing.assert_array_equal(a, b)
    single = np.array([1, 2, 3])
    assert runner.source_tick_array(single) is single


@pytest.mark.parametrize("version", ["1!7.4.1", "1!7.4.2"])
def test_host_shim_is_installed_for_both_observed_runtimes(monkeypatch, version):
    tick_module = SimpleNamespace(_send_buffer_times=lambda values, dt: values)

    class Vertex:
        def _install_send_buffer(self, times):
            self.times = times

    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: tick_module)
    monkeypatch.setitem(sys.modules,
        "spinn_front_end_common.utility_models.reverse_ip_tag_multicast_source_machine_vertex",
        SimpleNamespace(ReverseIPTagMulticastSourceMachineVertex=Vertex))
    description = runner.install_source_update_compatibility()
    times = [np.array([2, 5]), np.array([3])]
    indexed = tick_module._send_buffer_times(times, 1.)
    np.testing.assert_array_equal(indexed[np.array([1])][0], [3])
    vertex = Vertex()
    vertex._first_machine_time_step, vertex._run_until_timesteps = 0, 80
    vertex._install_send_buffer(indexed)
    assert vertex.times is indexed
    assert vertex._first_machine_time_step is None and vertex._run_until_timesteps is None
    assert version in description


def test_input_offset_moves_events_and_clocked_bias_together():
    trains = runner.input_spike_times(np.array([[0, 1], [1, 0], [0, 0]]), 1.0, 2)
    for actual, expected in zip(trains, [[3.0], [2.0], [2.0, 3.0, 4.0]]):
        np.testing.assert_array_equal(actual, expected)


def test_trace_diagnostic_counts_time_neuron_cells_and_first_divergence():
    a = np.array([[0, 1], [1, 0], [0, 0]])
    b = np.array([[0, 1], [0, 1], [0, 0]])
    assert analyzer.compare_trace(a, b) == dict(hardware_spikes=2, emulator_spikes=2,
        differing_spike_cells=2, first_differing_step=1)
    assert analyzer.compare_trace(a, a)["first_differing_step"] is None


def test_replacing_source_buffer_invalidates_same_window_cache():
    vertex = SimpleNamespace(_first_machine_time_step=0, _run_until_timesteps=55)
    runner.invalidate_source_buffer_cache(vertex)
    assert vertex._first_machine_time_step is None
    assert vertex._run_until_timesteps is None


def test_paper_matrix_keeps_all_five_original_repaired_conditions():
    paths = matrix_runner.model_paths(Path("bundle"))
    assert len(paths) == 10
    assert {seed for seed, _, _ in paths} == {1701, 2718, 3141, 5772, 8119}
    assert {variant for _, variant, _ in paths} == {"original", "reset_repaired"}


def test_packet_query_uses_coordinate_view_from_installed_schema():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE core_provenance(core_id, description, the_value)")
    db.execute("CREATE VIEW core_provenance_view AS SELECT 0 AS x, 1 AS y, 2 AS p, description, the_value FROM core_provenance")
    db.execute("INSERT INTO core_provenance VALUES (1, 'Number_of_late_spikes', 7)")
    reader = SimpleNamespace(run_query=lambda q: db.execute(q).fetchall(), messages=lambda: [])
    assert matrix_runner.read_packet_diagnostics(reader)["late_spikes"] == [(0, 1, 2, 7)]
    db.close()


def test_matrix_reuse_preserves_horizon_and_resets_before_new_input(monkeypatch, tmp_path):
    actions = []

    class Population:
        def __init__(self, size, cell, label, **kwargs):
            self.size, self.label = size, label
            actions.append(("population", label))
        def initialize(self, **kwargs): pass
        def record(self, name): pass
        def set(self, **kwargs): actions.append(("set", self.label))
        def get_data(self, name, clear):
            assert clear
            return SimpleNamespace(segments=[object()])

    sim = SimpleNamespace(setup=lambda **kw: actions.append(("setup",)),
        reset=lambda: actions.append(("reset",)),
        run=lambda steps: actions.append(("run", steps)),
        end=lambda: actions.append(("end",)), Population=Population,
        SpikeSourceArray=lambda **kw: kw, IF_curr_delta=lambda **kw: kw,
        set_number_of_neurons_per_core=lambda *a: None,
        Projection=lambda *a, **kw: object(), FromListConnector=lambda x: x,
        StaticSynapse=lambda: None)
    monkeypatch.setitem(sys.modules, "spinn_utilities.config_holder", SimpleNamespace(
        get_config_bool=lambda *a: False, get_config_int=lambda *a: 4, set_config=lambda *a: None))

    class Reader:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setitem(sys.modules, "spinn_front_end_common.interface.provenance",
                        SimpleNamespace(ProvenanceReader=Reader))
    monkeypatch.setattr(matrix_runner, "read_packet_diagnostics", lambda r: dict(late_spikes=[], messages=[]))
    monkeypatch.setattr(matrix_runner, "spike_readout", lambda *a:
        (np.zeros((50, 1)), np.array([1., 0.]), np.empty((0, 2))))
    model = SimpleNamespace(hidden_size=1, bias=np.zeros(1), output_weights=np.ones((1, 2)))
    mapping = SimpleNamespace(neuron_parameters=lambda m: {},
        projection_weights=lambda m: (np.ones((1, 1)), np.ones((1, 1))),
        emulate=lambda *a: dict(predictions=np.array([0]), spikes=np.zeros((1, 50, 1))))
    models = [(1701, variant, model) for variant in matrix_runner.VARIANTS]
    allocation = {}
    for index in range(2):
        folder = tmp_path / str(index)
        folder.mkdir()
        result = matrix_runner.run_input(sim, models, np.zeros((50, 1)), f"calib-{index}", index,
            folder, SimpleNamespace(time_scale_factor=100, spike_reader="neo-history"), mapping, allocation)
        assert result["observation_steps"] == 50 and result["run_steps"] == 64
        assert len(result["rows"]) == 2
    assert actions.count(("setup",)) == 1
    assert sum(a[0] == "population" for a in actions) == 3
    assert [a for a in actions if a[0] == "run"] == [("run", 64), ("run", 64)]
    reset = actions.index(("reset",))
    assert actions[reset:reset+3] == [("reset",), ("set", "shared_input_and_bias"), ("run", 64)]
    assert ("end",) not in actions
