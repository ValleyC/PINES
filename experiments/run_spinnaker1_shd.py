"""Run a frozen SHD mapping on physical SpiNNaker-1, one reset input per run.

Hidden dynamics execute on SpiNNaker. The original linear readout executes on
the host from recorded hidden spikes. This is explicitly a hybrid backend.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.adapters.spinnaker1 import SpiNNaker1Mapping, signed_connections
from pines.models import DenseRecurrentSNN


def input_spike_times(events: np.ndarray, timestep_ms: float, offset_steps: int = 0) -> np.ndarray:
    # Bias is a clocked source channel. This prevents i_offset from changing
    # initial voltage during the input-delivery warm-up ticks.
    trains = [(np.flatnonzero(events[:, i]) + offset_steps) * timestep_ms
              for i in range(events.shape[1])] + [(np.arange(len(events)) + offset_steps) * timestep_ms]
    # Keep the per-neuron trains ragged even when their lengths happen to match.
    result = np.empty(len(trains), dtype=object)
    result[:] = trains
    return result


def source_tick_array(converted):
    """Preserve converted tick values while enabling NumPy vertex indexing."""
    if isinstance(converted, list):
        indexed = np.empty(len(converted), dtype=object)
        indexed[:] = converted
        return indexed
    return converted


def invalidate_source_buffer_cache(vertex) -> None:
    # Replacing spike times also replaces the buffer. A repeated (0, T) run
    # must refill it, even when its time window equals the preceding run.
    vertex._first_machine_time_step = None
    vertex._run_until_timesteps = None


def install_source_update_compatibility() -> str | None:
    # In 7.4.1 _send_buffer_times returns a Python list for per-neuron trains,
    # but ReverseIpTagMultiCastSource.send_buffer_times indexes that result
    # with vertex_slice.get_raster_ids(), a NumPy array. This host-only shim
    # changes the container, not the time conversion or any neuron binary.
    if importlib.metadata.version("sPyNNaker") != "1!7.4.1":
        return None
    module = importlib.import_module("spynnaker.pyNN.models.spike_source.spike_source_array_vertex")
    original = module._send_buffer_times

    def indexed_times(spike_times, time_step):
        return source_tick_array(original(spike_times, time_step))

    module._send_buffer_times = indexed_times
    if importlib.metadata.version("SpiNNFrontEndCommon") == "1!7.4.1":
        from spinn_front_end_common.utility_models.reverse_ip_tag_multicast_source_machine_vertex import ReverseIPTagMulticastSourceMachineVertex
        install = ReverseIPTagMulticastSourceMachineVertex._install_send_buffer

        def install_and_invalidate(vertex, times):
            install(vertex, times)
            invalidate_source_buffer_cache(vertex)

        ReverseIPTagMulticastSourceMachineVertex._install_send_buffer = install_and_invalidate
    return "7.4.1 host ragged tick-array indexing and replaced-buffer cache invalidation, no numeric changes"


def spike_readout(segment, hidden_size: int, horizon: int, latency: int,
                  timestep_ms: float, output_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spikes = np.zeros((horizon, hidden_size), dtype=np.uint8)
    raw_rows = []
    for index, train in enumerate(segment.spiketrains):
        neuron = int(train.annotations.get("source_index", index))
        for ms in np.asarray(train, dtype=float):
            raw_rows.append((neuron, float(ms)))
            step = int(round(ms / timestep_ms)) - latency
            if 0 <= step < horizon:
                spikes[step, neuron] += 1
    logits = spikes.sum(axis=0, dtype=np.int64) @ output_weights
    return spikes, logits, np.asarray(raw_rows, dtype=float).reshape(-1, 2)


def run(args) -> dict:
    import pyNN.spiNNaker as sim
    from spinn_utilities.config_holder import get_config_bool, set_config

    compatibility = install_source_update_compatibility()

    probe = json.loads(Path(args.probe_analysis).read_text())
    if not probe["ready_for_shd_smoke"]:
        raise ValueError("Resolve the handcrafted probe mismatch before SHD")
    latency = probe["input_latency_steps"]
    # Extra recurrence latency would change the emulator's recurrent state.
    if latency != 1:
        raise ValueError("Extend the recurrent-delay emulator for the observed latency before SHD")
    model = DenseRecurrentSNN.load(args.model)
    with np.load(args.inputs, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        stop = args.start + args.count
        packed = data["packed_spikes"][args.start:stop]
        sample_ids = data["sample_ids"][args.start:stop].astype(str)
        events = np.unpackbits(packed, axis=-1, bitorder="little")[..., :metadata["input_channels"]]
    if len(events) != args.count or args.count < 1:
        raise ValueError("requested sample range is unavailable")
    mapping = SpiNNaker1Mapping(integration=args.integration)
    params = mapping.neuron_parameters(model)
    params["i_offset"] = np.zeros(model.hidden_size)
    w_in, w_rec = mapping.projection_weights(model)
    w_in = np.vstack((w_in, model.bias * mapping.timestep_ms))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    contract = mapping.contract() | {"bias_delivery": "clocked source channel", "input_latency_steps": latency,
                                    "input_offset_steps": args.input_offset_steps}
    config = dict(model_file=str(args.model), inputs_file=str(args.inputs),
                  sample_range=[args.start, stop], seed=args.seed,
                  contract=contract, time_scale_factor=args.time_scale_factor,
                  source_neurons_per_core=args.source_neurons_per_core,
                  hidden_neurons_per_core=args.hidden_neurons_per_core,
                  incoming_spike_buffer_size=args.incoming_spike_buffer_size,
                  provenance_enabled=args.provenance,
                  record_inputs=args.record_inputs,
                  fresh_allocations=args.fresh_allocations,
                  host_compatibility=compatibility,
                  primary_sampling="one unique input per reset execution, no repeated-run pooling",
                  independence="Population confidence additionally assumes independent inputs and hardware noise")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sim.setup(timestep=mapping.timestep_ms, min_delay=mapping.timestep_ms,
              time_scale_factor=args.time_scale_factor)
    if args.provenance:
        for key in ("read_router_provenance_data", "read_placements_provenance_data",
                    "read_graph_provenance_data", "extract_iobuf"):
            set_config("Reports", key, "True")
    results = []
    try:
        if get_config_bool("Machine", "virtual_board"):
            raise RuntimeError("A virtual board cannot supply physical SHD evidence")
        installed = importlib.metadata.version("sPyNNaker")
        if installed != probe["versions"]["sPyNNaker"]:
            raise ValueError("Run the probe with the currently installed sPyNNaker version")
        sim.set_number_of_neurons_per_core(sim.IF_curr_delta, args.hidden_neurons_per_core)
        if args.source_neurons_per_core:
            sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, args.source_neurons_per_core)
        inputs = sim.Population(model.input_size + 1,
            sim.SpikeSourceArray(spike_times=input_spike_times(events[0], mapping.timestep_ms, args.input_offset_steps)),
            label="SHD_inputs_and_bias")
        if args.record_inputs:
            inputs.record("spikes")
        additional = ({"incoming_spike_buffer_size": args.incoming_spike_buffer_size}
                      if args.incoming_spike_buffer_size else {})
        hidden = sim.Population(model.hidden_size, sim.IF_curr_delta(**params),
                                label="SHD_recurrent", additional_parameters=additional)
        hidden.initialize(v=0.0)
        hidden.record("spikes")
        projections = []
        for source, weights in ((inputs, w_in), (hidden, w_rec)):
            for receptor, rows in signed_connections(weights, mapping.timestep_ms).items():
                if rows:
                    projections.append(sim.Projection(source, hidden, sim.FromListConnector(rows),
                        synapse_type=sim.StaticSynapse(), receptor_type=receptor))
        for i, (event, sample_id) in enumerate(zip(events, sample_ids)):
            if i:
                if args.fresh_allocations:
                    # sPyNNaker documents that reset after get_machine()
                    # performs a hard reset, including a fresh allocation.
                    sim.get_machine()
                sim.reset()
                inputs.set(spike_times=input_spike_times(event, mapping.timestep_ms, args.input_offset_steps))
            sim.run((len(event) + latency + args.input_offset_steps + 2) * mapping.timestep_ms)
            segment = hidden.get_data("spikes", clear=True).segments[-1]
            spikes, logits, raw_spikes = spike_readout(segment, model.hidden_size,
                len(event), latency + args.input_offset_steps, mapping.timestep_ms, model.output_weights)
            ideal = mapping.emulate(model, event[None])
            prediction = int(logits.argmax())
            pair_id = f"seed{args.seed}-{sample_id}-run{i}"
            if args.record_inputs:
                source_segment = inputs.get_data("spikes", clear=True).segments[-1]
                _, _, raw_inputs = spike_readout(source_segment, model.input_size + 1,
                    len(event), args.input_offset_steps, mapping.timestep_ms,
                    np.zeros((model.input_size + 1, 1)))
                np.savez_compressed(output / f"input_{args.start+i:05d}.npz",
                                    raw_spikes_neuron_ms=raw_inputs)
            np.savez_compressed(output / f"sample_{args.start+i:05d}.npz",
                sample_id=np.asarray(sample_id), pair_id=np.asarray(pair_id),
                raw_spikes_neuron_ms=raw_spikes, spikes=spikes, logits=logits,
                prediction=prediction, emulator_prediction=int(ideal["predictions"][0]))
            results.append(dict(sample_id=str(sample_id), pair_id=pair_id, prediction=prediction,
                                emulator_prediction=int(ideal["predictions"][0])))
            print(f"{i+1}/{len(events)} {sample_id}: hw={prediction}, emulator={results[-1]['emulator_prediction']}", flush=True)
        np.savez_compressed(output / "capture.npz",
            predictions=np.asarray([r["prediction"] for r in results]),
            emulator_predictions=np.asarray([r["emulator_prediction"] for r in results]),
            sample_ids=sample_ids, pair_ids=np.asarray([r["pair_id"] for r in results]))
        metadata = dict(status="physical_capture_completed", backend="spinnaker1_hybrid",
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        spynnaker_version=installed, neuron_binary="IF_curr_delta.aplx",
                        machine=str(sim.get_machine()), samples=len(results),
                        disagreements=sum(r["prediction"] != r["emulator_prediction"] for r in results),
                        scope="Hardware hidden dynamics with host linear readout. No family or population certificate is inferred by this runner.")
        (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    finally:
        sim.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--probe-analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--integration", choices=["native_exponential", "match_source_euler"], default="match_source_euler")
    parser.add_argument("--time-scale-factor", type=int, default=10)
    parser.add_argument("--input-offset-steps", type=int, default=0)
    parser.add_argument("--source-neurons-per-core", type=int, default=0)
    parser.add_argument("--hidden-neurons-per-core", type=int, default=32)
    parser.add_argument("--incoming-spike-buffer-size", type=int, default=0)
    parser.add_argument("--provenance", action="store_true")
    parser.add_argument("--record-inputs", action="store_true")
    parser.add_argument("--fresh-allocations", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))
