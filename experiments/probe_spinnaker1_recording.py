"""Compare the two public spike readers on the same physical reset executions.

This checks data extraction only. It is not a classification or certificate run.
"""
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np


def sorted_rows(rows):
    rows = np.asarray(rows, dtype=float).reshape(-1, 2)
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


def run(output=Path("recording_reader_probe")):
    import pyNN.spiNNaker as sim
    from spinn_utilities.config_holder import get_config_bool

    output.mkdir(exist_ok=False)
    config = dict(purpose="Development-only recording reader equivalence",
        physical_certificate=False, run_steps=80, repeats=3, timestep_ms=1.,
        time_scale_factor=100,
        packages={name: importlib.metadata.version(name) for name in
                  ("sPyNNaker", "SpiNNFrontEndCommon", "SpiNNMan", "PyNN", "numpy")})
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    rows = []
    sim.setup(timestep=1., min_delay=1., time_scale_factor=100)
    try:
        if get_config_bool("Machine", "virtual_board"):
            raise RuntimeError("This recording probe requires physical hardware")
        parameters = dict(tau_m=3., cm=3., v_rest=0., v_reset=0.,
                          v_thresh=1., tau_refrac=0., i_offset=0.)
        source = sim.Population(3, sim.SpikeSourceArray(spike_times=[[2., 5.], [], [9., 10., 17.]]))
        first = sim.Population(3, sim.IF_curr_delta(**parameters), label="first")
        second = sim.Population(3, sim.IF_curr_delta(**parameters), label="second")
        silent = sim.Population(2, sim.IF_curr_delta(**parameters), label="silent")
        projections = [sim.Projection(pre, post, sim.OneToOneConnector(),
            synapse_type=sim.StaticSynapse(weight=6., delay=1.))
            for pre, post in ((source, first), (first, second))]
        for population in (first, second, silent):
            population.initialize(v=0.)
            population.record("spikes")
        for repetition in range(3):
            if repetition:
                sim.reset()
            sim.run(80)
            for population in (first, second, silent):
                started = time.monotonic()
                array_rows = sorted_rows(population.spinnaker_get_data("spikes"))
                array_seconds = time.monotonic()-started
                started = time.monotonic()
                segment = population.get_data("spikes", clear=True).segments[-1]
                neo_rows = sorted_rows([(int(train.annotations.get("source_index", index)), float(ms))
                    for index, train in enumerate(segment.spiketrains) for ms in np.asarray(train)])
                neo_seconds = time.monotonic()-started
                np.savez_compressed(output / f"run{repetition}_{population.label}.npz",
                                    numpy_rows=array_rows, neo_rows=neo_rows)
                row = dict(repetition=repetition, population=population.label,
                    equal=bool(np.array_equal(array_rows, neo_rows)), spike_count=len(array_rows),
                    array_seconds=array_seconds, neo_history_seconds=neo_seconds)
                rows.append(row)
                print("READER_COMPARISON", json.dumps(row), flush=True)
        report = dict(status="completed_development_recording_probe",
                      physical_certificate=False, all_equal=all(row["equal"] for row in rows), rows=rows)
        (output / "completed.json").write_text(json.dumps(report, indent=2)+"\n")
    finally:
        try:
            sim.end()
        finally:
            shutil.make_archive(str(output)+"_capture", "zip", root_dir=output)


if __name__ == "__main__":
    run()
