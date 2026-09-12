"""Test allocation reuse on full DVS calibration networks before a primary run.

Two complete calibration recordings are followed by exact input-window repeats.
This diagnostic never reads held-out audit/canary inputs or issues a certificate.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from run_spinnaker1_dvs import (
    SpiNNaker1DVSMapping, aggregate_windows, build_network, load_model,
    input_spike_times, install_source_update_compatibility, spike_readout,
    install_bounded_memory_transfer, read_packet_diagnostics,
)


def development_schedule():
    cases = []
    for index in range(2):
        cases.extend(dict(input_index=index, window=window, repeat=False) for window in range(4))
        cases.append(dict(input_index=0, window=0, repeat=True))
    return cases


def compare_arrays(observed, expected):
    return {key: dict(equal=bool(np.array_equal(observed[key], expected[key])),
                      differing_entries=int(np.count_nonzero(observed[key] != expected[key])))
            for key in ("conv1_spikes", "conv2_spikes", "hidden_spikes", "window_logits")}


def read_layers(populations, weights, horizon, offset):
    data = {}
    for name, population, latency in zip(("conv1", "conv2", "hidden"), populations, (1, 2, 3)):
        segment = population.get_data("spikes", clear=True).segments[-1]
        readout = weights if name == "hidden" else np.empty((population.size, 0))
        spikes, logits, raw = spike_readout(segment, population.size, horizon, offset+latency, 1., readout)
        data[name+"_spikes"] = spikes
        data["raw_"+name+"_spikes_neuron_ms"] = raw
        if name == "hidden":
            data["window_logits"] = logits
    return data


def run(args):
    import pyNN.spiNNaker as sim
    from spinn_utilities.config_holder import get_config_bool, set_config
    from spinn_front_end_common.interface.provenance import ProvenanceReader

    compatibility = install_source_update_compatibility()
    transfer = install_bounded_memory_transfer(args.transfer_chunk_bytes)
    with np.load(args.bundle / "development_inputs.npz") as archive:
        packed = archive["packed_spikes"][:2]
        sample_ids = archive["sample_ids"][:2].astype(str)
    models = [load_model(args.bundle / "models" / str(args.seed) / (v+".npz")) for v in args.variants]
    mapping = SpiNNaker1DVSMapping()
    cases = development_schedule()
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(status="allocation_reuse_development_probe", physical_certificate=False,
        seed=args.seed, variants=args.variants, sample_ids=sample_ids.tolist(), cases=cases,
        mapping=mapping.contract(), input_offset_steps=args.input_offset_steps,
        time_scale_factor=args.time_scale_factor, conv_neurons_per_core=args.conv_neurons_per_core,
        hidden_neurons_per_core=args.hidden_neurons_per_core, source_neurons_per_core=args.source_neurons_per_core,
        host_compatibility=compatibility, memory_transfer=transfer,
        packages={p: importlib.metadata.version(p) for p in ("sPyNNaker", "SpiNNFrontEndCommon", "SpiNNMan", "PyNN", "numpy")},
        scope="Development-only reset reuse. This profile is not yet selected for primary physical certification.")
    (args.output / "config.json").write_text(json.dumps(config, indent=2)+"\n")

    def trains_for(case):
        event = np.unpackbits(packed[case["input_index"], case["window"]], axis=-1, bitorder="little")[..., :2048]
        return input_spike_times(event, 1., args.input_offset_steps)

    sim.setup(timestep=1., min_delay=1., time_scale_factor=args.time_scale_factor)
    results = []
    try:
        if get_config_bool("Machine", "virtual_board"):
            raise RuntimeError("The reset probe requires physical hardware")
        for setting in ("read_placements_provenance_data", "read_router_provenance_data"):
            set_config("Reports", setting, "True")
        trains = trains_for(cases[0])
        source = sim.Population(2048, sim.SpikeSourceArray(spike_times=trains[:-1]), label="dvs_input")
        source.set_max_atoms_per_core(args.source_neurons_per_core)
        clock = sim.Population(1, sim.SpikeSourceArray(spike_times=trains[-1]), label="bias_clock")
        networks = [build_network(sim, source, clock, state, meta, args, mapping) for state, meta in models]
        for step, case in enumerate(cases):
            started = time.monotonic()
            if step:
                # Do not query get_machine() between runs: sPyNNaker then forces
                # a hard reset and reloads the model instead of testing reuse.
                sim.reset()
                trains = trains_for(case)
                source.set(spike_times=trains[:-1])
                clock.set(spike_times=trains[-1])
            sim.run(60+args.input_offset_steps+5)
            folder = args.output / f"step_{step:02d}"
            folder.mkdir()
            rows = []
            for (_, meta), (populations, projections, weights) in zip(models, networks):
                data = read_layers(populations, weights, 60, args.input_offset_steps)
                variant = meta["variant"]
                np.savez_compressed(folder / (variant+".npz"), **data)
                row = dict(variant=variant, window_logits=data["window_logits"].tolist(),
                           physical_window_prediction=int(data["window_logits"].argmax()))
                expected_path = args.bundle / "development_traces" / str(args.seed) / variant / f"input_{case['input_index']:05d}.npz"
                with np.load(expected_path) as oracle:
                    expected = {name+"_spikes": oracle[name][case["window"]] for name in ("conv1", "conv2", "hidden")}
                    expected["window_logits"] = oracle["window_logits"][case["window"]]
                    row["emulator_comparison"] = compare_arrays(data, expected)
                    row["emulator_window_prediction"] = int(expected["window_logits"].argmax())
                if case["repeat"]:
                    with np.load(args.output / "step_00" / (variant+".npz")) as first:
                        row["repeat_comparison"] = compare_arrays(data, first)
                rows.append(row)
            with ProvenanceReader() as reader:
                diagnostics = read_packet_diagnostics(reader)
            result = dict(step=step, **case, seconds=time.monotonic()-started, rows=rows, diagnostics=diagnostics)
            (folder / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
            results.append(result)
            print(f"step {step+1}/{len(cases)}: input {case['input_index']} window {case['window']} repeat={case['repeat']}, {result['seconds']:.1f}s", flush=True)
        predictions = []
        for index in range(2):
            recording = [r for r in results if r["input_index"] == index and not r["repeat"]]
            for position, (_, meta) in enumerate(models):
                logits = [r["rows"][position]["window_logits"] for r in recording]
                predictions.append(dict(sample_id=sample_ids[index], variant=meta["variant"],
                    prediction=int(aggregate_windows(logits, meta["aggregation_temperature"]).argmax())))
        report = dict(status="completed_allocation_reuse_development_probe", physical_certificate=False,
            machine=str(sim.get_machine()), rows=results, calibration_predictions=predictions)
        (args.output / "completed.json").write_text(json.dumps(report, indent=2)+"\n")
    finally:
        sim.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--variants", nargs="+", choices=["original", "floor_repaired"], default=["original", "floor_repaired"])
    parser.add_argument("--input-offset-steps", type=int, default=2)
    parser.add_argument("--time-scale-factor", type=int, default=100)
    parser.add_argument("--conv-neurons-per-core", type=int, default=128)
    parser.add_argument("--hidden-neurons-per-core", type=int, default=32)
    parser.add_argument("--source-neurons-per-core", type=int, default=32)
    parser.add_argument("--transfer-chunk-bytes", type=int, default=256*1024)
    parser.set_defaults(record_layers=True)
    run(parser.parse_args())
