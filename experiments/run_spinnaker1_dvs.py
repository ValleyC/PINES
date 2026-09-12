"""Run complete DVS convolution/recurrent dynamics on physical SpiNNaker-1.

This development runner allocates fresh hardware for each input window. Four
window captures make one classification observation, not four independent ones.
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pines.adapters.spinnaker1_dvs import (
    SpiNNaker1DVSMapping, aggregate_windows, bias_connections,
    conv2d_connections, dense_connections, split_signed_rows,
)
from run_spinnaker1_shd import input_spike_times, install_source_update_compatibility, spike_readout
from run_spinnaker1_matrix import read_packet_diagnostics


def load_model(path):
    with np.load(path) as archive:
        meta = json.loads(str(archive["metadata"]))
        state = {key:archive[key] for key in archive.files if key != "metadata"}
    return state,meta


def connect(sim, source, target, rows, projections):
    for receptor, selected in split_signed_rows(rows).items():
        projections.append(sim.Projection(source,target,sim.FromListConnector(selected),
            synapse_type=sim.StaticSynapse(),receptor_type=receptor))


def build_network(sim, source, clock, state, meta, args, mapping):
    parameters = mapping.prepare_parameters(state)
    neurons = mapping.neuron_parameters(meta["tau_mem"],meta["threshold"])
    rows1, shape1 = conv2d_connections(parameters["conv1.weight"],(2,32,32))
    rows2, shape2 = conv2d_connections(parameters["conv2.weight"],shape1)
    hidden_size = parameters["recurrent.weight"].shape[0]
    populations = []
    for name,size,cores in (("conv1",int(np.prod(shape1)),args.conv_neurons_per_core),
                            ("conv2",int(np.prod(shape2)),args.conv_neurons_per_core),
                            ("hidden",hidden_size,args.hidden_neurons_per_core)):
        pop = sim.Population(size,sim.IF_curr_delta(**neurons),label=f"{meta['variant']}_{name}",
            additional_parameters={"incoming_spike_buffer_size":4096})
        pop.set_max_atoms_per_core(cores)
        pop.initialize(v=0.)
        if args.record_layers or name == "hidden":
            pop.record("spikes")
        populations.append(pop)
    conv1,conv2,hidden = populations
    projections = []
    connect(sim,source,conv1,rows1,projections)
    connect(sim,conv1,conv2,rows2,projections)
    connect(sim,conv2,hidden,dense_connections(parameters["hidden_input.weight"]),projections)
    connect(sim,hidden,hidden,dense_connections(parameters["recurrent.weight"]),projections)
    connect(sim,clock,conv1,bias_connections(parameters["conv1.bias"],shape1[1:],1),projections)
    connect(sim,clock,conv2,bias_connections(parameters["conv2.bias"],shape2[1:],2),projections)
    connect(sim,clock,hidden,bias_connections(parameters["hidden_input.bias"],(),3),projections)
    return populations,projections,parameters["readout.weight"].T


def run_window(sim, models, event, output, args, mapping):
    from spinn_utilities.config_holder import get_config_bool,set_config
    from spinn_front_end_common.interface.provenance import ProvenanceReader
    started = time.monotonic()
    sim.setup(timestep=1.,min_delay=1.,time_scale_factor=args.time_scale_factor)
    try:
        if get_config_bool("Machine","virtual_board"):
            raise RuntimeError("This runner requires physical hardware")
        for setting in ("read_placements_provenance_data","read_router_provenance_data"):
            set_config("Reports",setting,"True")
        trains = input_spike_times(event.reshape(len(event),-1),1.,2)
        source = sim.Population(2048,sim.SpikeSourceArray(spike_times=trains[:-1]),label="dvs_input")
        source.set_max_atoms_per_core(args.source_neurons_per_core)
        clock = sim.Population(1,sim.SpikeSourceArray(spike_times=trains[-1]),label="bias_clock")
        networks = [build_network(sim,source,clock,state,meta,args,mapping) for state,meta in models]
        sim.run(len(event)+7)
        results = []
        for (state,meta),(pops,projections,weights) in zip(models,networks):
            hidden = pops[-1]
            spikes,logits,raw = spike_readout(hidden.get_data("spikes").segments[-1],hidden.size,len(event),5,1.,weights)
            data = dict(hidden_spikes=spikes,window_logits=logits,raw_hidden_spikes_neuron_ms=raw)
            if args.record_layers:
                for name,pop,latency in zip(("conv1","conv2"),pops[:2],(3,4)):
                    q,_,times = spike_readout(pop.get_data("spikes").segments[-1],pop.size,len(event),latency,1.,np.empty((pop.size,0)))
                    data[name+"_spikes"] = q
                    data["raw_"+name+"_spikes_neuron_ms"] = times
            np.savez_compressed(output / (meta["variant"]+".npz"),**data)
            results.append(dict(variant=meta["variant"],window_logits=logits.tolist(),hidden_spikes=int(spikes.sum())))
        with ProvenanceReader() as reader:
            diagnostics = read_packet_diagnostics(reader)
        return dict(status="physical_window_capture_completed",timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    seconds=time.monotonic()-started,machine=str(sim.get_machine()),
                    java_transfer=get_config_bool("Java","use_java"),diagnostics=diagnostics,rows=results)
    finally:
        sim.end()


def run(args):
    import pyNN.spiNNaker as sim
    compatibility = install_source_update_compatibility()
    models = [load_model(args.bundle / "models" / str(args.seed) / (variant+".npz")) for variant in args.variants]
    with np.load(args.inputs) as archive:
        packed = archive["packed_spikes"]
        ids = archive["sample_ids"].astype(str)
        metadata = json.loads(str(archive["metadata"]))
    if args.count < 1 or args.start < 0 or args.start+args.count > len(ids):
        raise ValueError("Requested input range is unavailable")
    args.output.mkdir(parents=True,exist_ok=False)
    mapping = SpiNNaker1DVSMapping()
    config = dict(seed=args.seed,variants=args.variants,input_metadata=metadata,start=args.start,count=args.count,
        windows=args.windows,mapping=mapping.contract(),time_scale_factor=args.time_scale_factor,
        conv_neurons_per_core=args.conv_neurons_per_core,hidden_neurons_per_core=args.hidden_neurons_per_core,
        source_neurons_per_core=args.source_neurons_per_core,host_compatibility=compatibility,
        observation="One complete four-window recording per condition, never count individual windows as independent inputs.",
        packages={p:importlib.metadata.version(p) for p in ("sPyNNaker","SpiNNFrontEndCommon","SpiNNMan","PyNN","numpy")})
    (args.output / "config.json").write_text(json.dumps(config,indent=2)+"\n")
    for index in range(args.start,args.start+args.count):
        folder = args.output / f"input_{index:05d}"
        folder.mkdir()
        records = []
        for window in args.windows:
            event = np.unpackbits(packed[index,window],axis=-1,bitorder="little")[...,:2048].reshape(60,2,32,32)
            target = folder / f"window_{window}"
            target.mkdir()
            record = run_window(sim,models,event,target,args,mapping)
            (target / "summary.json").write_text(json.dumps(record,indent=2)+"\n")
            records.append(record)
            print(f"input {index} window {window} completed in {record['seconds']:.1f}s",flush=True)
        result = dict(sample_id=ids[index],windows=args.windows,status="development_partial_windows")
        if args.windows == [0,1,2,3]:
            result["status"] = "physical_classification_capture_completed"
            result["predictions"] = {meta["variant"]:int(aggregate_windows(
                [record["rows"][position]["window_logits"] for record in records],meta["aggregation_temperature"]).argmax())
                for position,(_,meta) in enumerate(models)}
        (folder / "summary.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.output / "completed.json").write_text(json.dumps(dict(samples=args.count,windows=args.windows))+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=1701)
    parser.add_argument("--variants",nargs="+",choices=["original","floor_repaired"],default=["original","floor_repaired"])
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--count",type=int,required=True)
    parser.add_argument("--windows",type=int,nargs="+",choices=[0,1,2,3],default=[0,1,2,3])
    parser.add_argument("--time-scale-factor",type=int,default=100)
    parser.add_argument("--conv-neurons-per-core",type=int,default=128)
    parser.add_argument("--hidden-neurons-per-core",type=int,default=32)
    parser.add_argument("--source-neurons-per-core",type=int,default=32)
    parser.add_argument("--record-layers",action="store_true")
    run(parser.parse_args())
