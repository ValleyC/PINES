"""Isolate packet-colour alignment across PyNN resets on physical SpiNNaker.

This small two-layer diagnostic tests 67-step and 80-step executions. It is
not a benchmark observation and never supplies a manuscript certificate.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np


def run(output=Path("reset_alignment_probe")):
    import pyNN.spiNNaker as sim
    from spinn_utilities.config_holder import get_config_bool, get_config_int

    output.mkdir(exist_ok=False)
    summaries = []
    config = dict(purpose="development-only two-layer reset diagnostic",
        physical_certificate=False, durations_ms=[67, 80], repeats=3,
        source_spikes_ms=[2, 5, 9], timestep_ms=1, time_scale_factor=100,
        observation_horizon_ms=20,
        packages={name: importlib.metadata.version(name) for name in
                  ("sPyNNaker", "SpiNNFrontEndCommon", "SpiNNMan", "PyNN", "numpy")})
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    try:
        for duration in config["durations_ms"]:
            sim.setup(timestep=1., min_delay=1., time_scale_factor=100)
            try:
                if get_config_bool("Machine", "virtual_board"):
                    raise RuntimeError("The reset diagnostic requires physical hardware")
                colour_bits = get_config_int("Simulation", "n_colour_bits")
                neurons = dict(tau_m=3., cm=3., v_rest=0., v_reset=0.,
                               v_thresh=1., tau_refrac=0., i_offset=0.)
                source = sim.Population(1, sim.SpikeSourceArray(spike_times=[2.,5.,9.]), label="input")
                first = sim.Population(1, sim.IF_curr_delta(**neurons), label="first")
                second = sim.Population(1, sim.IF_curr_delta(**neurons), label="second")
                projections = [sim.Projection(pre, post, sim.OneToOneConnector(),
                    synapse_type=sim.StaticSynapse(weight=6., delay=1.))
                    for pre,post in ((source,first),(first,second))]
                for pop in (first,second):
                    pop.initialize(v=0.)
                    pop.record("spikes")
                baseline = {}
                for repetition in range(config["repeats"]):
                    started = time.monotonic()
                    if repetition:
                        sim.reset()
                    sim.run(duration)
                    arrays = {}
                    for name,pop in (("first",first),("second",second)):
                        segment = pop.get_data("spikes", clear=True).segments[-1]
                        arrays[name] = np.asarray(segment.spiketrains[0], dtype=np.float64)
                    if repetition == 0:
                        baseline = {name: values.copy() for name,values in arrays.items()}
                    row = dict(duration_ms=duration, repetition=repetition,
                        colour_bits=colour_bits, seconds=time.monotonic()-started,
                        spikes_ms={name: values.tolist() for name,values in arrays.items()},
                        matches_first_run={name: bool(np.array_equal(values,baseline[name]))
                                           for name,values in arrays.items()})
                    summaries.append(row)
                    np.savez_compressed(output / f"duration{duration}_repeat{repetition}.npz", **arrays)
                    (output / "progress.json").write_text(json.dumps(summaries,indent=2)+"\n")
                    print("RESET_ALIGNMENT",json.dumps(row),flush=True)
            finally:
                sim.end()
        report = dict(status="completed_development_reset_diagnostic",
                      physical_certificate=False, rows=summaries)
        (output / "completed.json").write_text(json.dumps(report,indent=2)+"\n")
    finally:
        shutil.make_archive(str(output)+"_capture", "zip", root_dir=output)


if __name__ == "__main__":
    run()
