"""Run all five SHD seeds before/after repair on physical SpiNNaker-1.

By default each input uses a fresh allocation. Ten model populations receive
the same input, so each condition contributes one observation per allocation.
Dependence across conditions is permitted by the simultaneous confidence bound.
Hidden dynamics run on the board and linear readouts on the host.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.adapters.spinnaker1 import SpiNNaker1Mapping, signed_connections
from pines.models import DenseRecurrentSNN
from pines.adapters.spinnaker1_timing import align_run_steps
from run_spinnaker1_shd import (
    input_spike_times, spike_readout, current_spike_readout,
    install_source_update_compatibility,
)


SEEDS = (1701, 2718, 3141, 5772, 8119)
VARIANTS = ("original", "reset_repaired")


def model_paths(bundle: Path, seeds=SEEDS):
    return [(seed, variant, bundle / "models" / str(seed) / (variant + ".npz"))
            for seed in seeds for variant in VARIANTS]


def read_packet_diagnostics(reader):
    # 7.4.1 stores coordinates in the joined view, not core_provenance itself.
    late = reader.run_query("SELECT x, y, p, the_value FROM core_provenance_view "
                            "WHERE description = 'Number_of_late_spikes' AND the_value > 0")
    return dict(late_spikes=late, messages=reader.messages())


def run_input(sim, models, event, sample_id, index, output, args, mapping, allocation=None):
    from spinn_utilities.config_holder import set_config, get_config_bool, get_config_int
    started = time.monotonic()
    reuse = allocation is not None
    initialize = not reuse or not allocation
    if initialize:
        sim.setup(timestep=1.0, min_delay=1.0, time_scale_factor=args.time_scale_factor)
        for key in ("read_placements_provenance_data", "read_router_provenance_data"):
            set_config("Reports", key, "True")
    try:
        if get_config_bool("Machine", "virtual_board"):
            raise RuntimeError("physical evidence requires a real board")
        if initialize:
            sim.set_number_of_neurons_per_core(sim.IF_curr_delta, 32)
            sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 32)
            source = sim.Population(event.shape[1] + 1,
                sim.SpikeSourceArray(spike_times=input_spike_times(event, 1.0, 2)),
                label="shared_input_and_bias")
            hidden_populations, projections = [], []
            for seed, variant, model in models:
                params = mapping.neuron_parameters(model)
                params["i_offset"] = np.zeros(model.hidden_size)
                hidden = sim.Population(model.hidden_size, sim.IF_curr_delta(**params),
                    label=f"seed{seed}_{variant}",
                    additional_parameters={"incoming_spike_buffer_size": 2048})
                hidden.initialize(v=0.0)
                hidden.record("spikes")
                w_in, w_rec = mapping.projection_weights(model)
                w_in = np.vstack((w_in, model.bias))
                for pre, weights in ((source, w_in), (hidden, w_rec)):
                    for receptor, rows in signed_connections(weights, 1.0).items():
                        if rows:
                            projections.append(sim.Projection(pre, hidden, sim.FromListConnector(rows),
                                synapse_type=sim.StaticSynapse(), receptor_type=receptor))
                hidden_populations.append(hidden)
            if reuse:
                allocation.update(source=source, hidden=hidden_populations, projections=projections,
                                  colour_bits=get_config_int("Simulation", "n_colour_bits"))
        else:
            sim.reset()
            allocation["source"].set(spike_times=input_spike_times(event, 1.0, 2))
            hidden_populations = allocation["hidden"]
        run_steps = align_run_steps(len(event)+5, allocation["colour_bits"]) if reuse else len(event)+5
        sim.run(run_steps)
        rows = []
        for (seed, variant, model), hidden in zip(models, hidden_populations):
            if args.spike_reader == "numpy-current":
                spikes, logits, raw = current_spike_readout(hidden, len(event), 3, 1., model.output_weights)
            else:
                segment = hidden.get_data("spikes", clear=reuse).segments[-1]
                spikes, logits, raw = spike_readout(segment, model.hidden_size, len(event),
                                                   3, 1.0, model.output_weights)
            emu = mapping.emulate(model, event[None])
            prediction = int(logits.argmax())
            emulator_prediction = int(emu["predictions"][0])
            name = f"seed{seed}_{variant}"
            np.savez_compressed(output / (name + ".npz"),
                sample_id=np.asarray(sample_id),
                pair_id=np.asarray(f"input{index}-{name}"),
                spikes=spikes, logits=logits, raw_spikes_neuron_ms=raw,
                prediction=prediction, emulator_prediction=emulator_prediction)
            rows.append(dict(seed=seed, variant=variant, prediction=prediction,
                             emulator_prediction=emulator_prediction,
                             hardware_spikes=int(spikes.sum()),
                             emulator_spikes=int(emu["spikes"].sum())))
        # Save per-allocation diagnostic counters alongside the predictions.
        from spinn_front_end_common.interface.provenance import ProvenanceReader
        with ProvenanceReader() as reader:
            diagnostics = read_packet_diagnostics(reader)
        (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        result = dict(status="physical_capture_completed", sample_id=sample_id, input_index=index,
                      timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      seconds=time.monotonic()-started, rows=rows, run_steps=run_steps,
                      observation_steps=len(event))
        if reuse:
            result.update(execution_profile="aligned_reset_reuse", colour_bits=allocation["colour_bits"])
        else:
            result["machine"] = str(sim.get_machine())
        return result
    finally:
        if not reuse:
            sim.end()


def run(args):
    import pyNN.spiNNaker as sim
    compatibility = install_source_update_compatibility()
    paths = model_paths(args.bundle, args.seeds)
    models = [(seed, variant, DenseRecurrentSNN.load(path)) for seed, variant, path in paths]
    with np.load(args.inputs, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        ids = data["sample_ids"].astype(str)
        events = np.unpackbits(data["packed_spikes"], axis=-1, bitorder="little")[..., :700]
    stop = args.start + args.count
    if args.count < 1 or args.start < 0 or stop > len(ids):
        raise ValueError("requested input range is unavailable")
    mapping = SpiNNaker1Mapping(integration="match_source_euler")
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(dataset=metadata, inputs_file=args.inputs.name,
                  sample_range=[args.start, stop], seeds=args.seeds, variants=VARIANTS,
                  conditions=len(models), mapping=mapping.contract(),
                  time_scale_factor=args.time_scale_factor, input_offset_steps=2,
                  source_neurons_per_core=32, hidden_neurons_per_core=32,
                  incoming_spike_buffer_size=2048,
                  execution_profile="aligned_reset_reuse" if args.reuse_reset else "fresh_allocation_per_input",
                  sampling="one unique input per reset execution, shared across conditions" if args.reuse_reset else
                           "one unique input per fresh allocation, shared across conditions",
                  independence="Independent inputs and execution-level hardware noise are assumed. Conditions need not be independent.",
                  repeat_first=args.repeat_first,
                  spike_reader=args.spike_reader,
                  host_compatibility=compatibility,
                  packages={p: importlib.metadata.version(p) for p in
                            ("sPyNNaker", "SpiNNFrontEndCommon", "SpiNNMan", "PyNN", "numpy")})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    allocation = {} if args.reuse_reset else None
    try:
        for index in range(args.start, stop):
            folder = args.output / f"input_{index:05d}"
            folder.mkdir()
            result = run_input(sim, models, events[index], str(ids[index]), index, folder, args, mapping, allocation)
            (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            mismatches = sum(r["prediction"] != r["emulator_prediction"] for r in result["rows"])
            print(f"{index-args.start+1}/{args.count} {ids[index]}: "
                  f"{mismatches}/{len(models)} condition disagreements, {result['seconds']:.1f}s", flush=True)
        if args.repeat_first:
            folder = args.output / "repeat_first"
            folder.mkdir()
            result = run_input(sim, models, events[args.start], str(ids[args.start]), args.start,
                               folder, args, mapping, allocation)
            result["status"] = "additional_repeat_not_a_primary_observation"
            (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            comparisons = []
            for seed, variant, _ in models:
                name = f"seed{seed}_{variant}.npz"
                with np.load(args.output / f"input_{args.start:05d}" / name) as first, np.load(folder / name) as repeated:
                    comparisons.append(dict(seed=seed, variant=variant,
                        spikes_equal=bool(np.array_equal(first["spikes"], repeated["spikes"])),
                        logits_equal=bool(np.array_equal(first["logits"], repeated["logits"]))))
            (folder / "comparison.json").write_text(json.dumps(comparisons, indent=2)+"\n")
            print("REPEAT_FIRST", json.dumps(comparisons), flush=True)
        completed = dict(samples=args.count, conditions=len(models), status="physical_matrix_capture_completed")
        if allocation is not None:
            completed["machine"] = str(sim.get_machine())
        (args.output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    finally:
        if allocation is not None:
            sim.end()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=861)
    p.add_argument("--time-scale-factor", type=int, default=10)
    p.add_argument("--reuse-reset", action="store_true")
    p.add_argument("--repeat-first", action="store_true",
                   help="Capture a separate repeat for variability or calibration, excluded from primary counts")
    p.add_argument("--spike-reader", choices=("neo-history", "numpy-current"), default="neo-history")
    run(p.parse_args())
