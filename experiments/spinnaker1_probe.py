"""Run handcrafted IF_curr_delta traces on physical SpiNNaker-1 via EBRAINS.

Run this in the official EBRAINS kernel, after the official SpiNNaker Setup
notebook. No dataset, labels, model checkpoint or PINES installation is needed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def probe_specification() -> list[dict]:
    # Each case is an independent neuron. Subthreshold cases expose integration
    # and synapse timing without reset obscuring the voltage update.
    return [
        dict(name="impulse", weight=1.0, threshold=100.0, delay=1.0, bias=0.0, reset=0.0),
        dict(name="delay_two", weight=1.0, threshold=100.0, delay=2.0, bias=0.0, reset=0.0),
        dict(name="inhibitory", weight=-1.0, threshold=100.0, delay=1.0, bias=0.0, reset=0.0),
        dict(name="bias", weight=0.0, threshold=100.0, delay=1.0, bias=0.5, reset=0.0),
        dict(name="zero_reset", weight=8.0, threshold=1.0, delay=1.0, bias=0.0, reset=0.0),
        dict(name="nonzero_reset", weight=8.0, threshold=1.0, delay=1.0, bias=0.0, reset=0.25),
        dict(name="signed_cancel", weight=1.0, threshold=100.0, delay=1.0, bias=0.0, reset=0.0),
        dict(name="recurrent", weight=8.0, threshold=1.0, delay=1.0, bias=0.0, reset=0.0),
    ]


def run_probe(output: str | Path) -> dict:
    import pyNN.spiNNaker as sim
    from spinn_utilities.config_holder import get_config_bool

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specs = probe_specification()
    config = dict(timestep_ms=1.0, duration_ms=25.0, input_times_ms=[2.0, 8.0, 14.0],
                  tau_m_ms=5.0, cm=5.0, tau_refrac_ms=0.0, cases=specs)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sim.setup(timestep=1.0, min_delay=1.0, time_scale_factor=10)
    try:
        if get_config_bool("Machine", "virtual_board"):
            raise RuntimeError("A virtual board cannot supply physical probe evidence")
        source = sim.Population(1, sim.SpikeSourceArray(spike_times=config["input_times_ms"]), label="probe_input")
        source.record("spikes")
        populations = []
        for spec in specs:
            pop = sim.Population(1, sim.IF_curr_delta(
                tau_m=5.0, cm=5.0, v_rest=0.0, v_reset=spec["reset"],
                v_thresh=spec["threshold"], tau_refrac=0.0, i_offset=spec["bias"]),
                label=spec["name"])
            pop.initialize(v=0.0)
            pop.record(["spikes", "v", "gsyn_exc", "gsyn_inh"])
            if spec["weight"]:
                sim.Projection(source, pop, sim.OneToOneConnector(),
                    sim.StaticSynapse(weight=abs(spec["weight"]), delay=spec["delay"]),
                    receptor_type="excitatory" if spec["weight"] > 0 else "inhibitory")
            if spec["name"] == "signed_cancel":
                sim.Projection(source, pop, sim.OneToOneConnector(),
                    sim.StaticSynapse(weight=1.0, delay=1.0), receptor_type="inhibitory")
            if spec["name"] == "recurrent":
                sim.Projection(pop, pop, sim.OneToOneConnector(),
                    sim.StaticSynapse(weight=8.0, delay=1.0), receptor_type="excitatory")
            populations.append(pop)
        sim.run(config["duration_ms"])
        arrays = {}
        summary = []
        for spec, pop in zip(specs, populations):
            seg = pop.get_data().segments[-1]
            spike_times = np.asarray(seg.spiketrains[0], dtype=float)
            arrays[spec["name"] + "__spikes_ms"] = spike_times
            for signal in seg.analogsignals:
                arrays[spec["name"] + "__" + signal.name] = np.asarray(signal)
                arrays[spec["name"] + "__times_ms"] = np.asarray(signal.times)
            summary.append(dict(case=spec["name"], spikes_ms=spike_times.tolist()))
        arrays["source_spikes_ms"] = np.asarray(source.get_data("spikes").segments[-1].spiketrains[0])
        np.savez_compressed(output / "traces.npz", **arrays)
        machine = sim.get_machine()
        versions = {}
        for name in ("sPyNNaker", "SpiNNMan", "SpiNNFrontEndCommon", "PyNN", "numpy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not provided by package metadata"
        result = dict(status="physical_probe_completed", backend="spinnaker1",
                      timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      machine=str(machine), versions=versions, cases=summary,
                      voltage_recording="pre-update", certificate=False)
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return result
    finally:
        sim.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run_probe(parser.parse_args().output)
