"""Check DVS FPGA memory images and reproduce their integer golden outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.dvs_hardware import (BIASES, WEIGHTS, TRACE_FIELDS, load_model, load_parameters,
    mapping_contract, quantize_model, run_numpy, run_recordings, unpack_inputs)


def read_mem(path, bits):
    words = np.asarray([int(line,16) for line in path.read_text(encoding="ascii").splitlines()], dtype=np.int64)
    return np.where(words >= (1 << (bits-1)), words-(1 << bits), words)


def verify(bundle, device="cpu", full=False):
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert json.loads((bundle / "target_contract.json").read_text()) == mapping_contract()
    splits = {}
    for split in manifest["samples"]:
        with np.load(bundle / "common" / (split+"_inputs.npz"), allow_pickle=False) as store:
            packed, ids = store["packed_spikes"], store["sample_ids"]
            assert not any("label" in key for key in store.files)
        assert packed.shape == (manifest["samples"][split],4,60,256)
        assert len(np.unique(ids)) == len(ids)
        with (bundle / "common" / (split+"_input_spikes.mem")).open(encoding="ascii") as handle:
            lines = handle.read().splitlines()
        assert len(lines) == len(ids)*240 and all(len(line)==512 for line in lines)
        unpacked_bytes = b"".join(int(line,16).to_bytes(256,"little") for line in lines)
        np.testing.assert_array_equal(np.frombuffer(unpacked_bytes,dtype=np.uint8).reshape(packed.shape), packed)
        splits[split] = (packed, ids)
    assert not (set(splits["audit"][1]) & set(splits["canary"][1]))
    configurations, comparisons, independent_windows = 0, 0, 0
    for seed in manifest["seeds"]:
        seed_root = bundle / "seeds" / str(seed)
        for variant in manifest["variants"]:
            folder = seed_root / variant
            parameters = load_parameters(folder / "parameters_int.npz")
            model, meta = load_model(folder / "source_model.npz")
            requantized = quantize_model(model,meta)
            for key in parameters:
                np.testing.assert_array_equal(parameters[key], requantized[key], err_msg=key)
            for name in WEIGHTS:
                np.testing.assert_array_equal(read_mem(folder/(name+"_weights.mem"),8), parameters[name+"_weights_q8"].reshape(-1))
            for name in BIASES:
                np.testing.assert_array_equal(read_mem(folder/(name+"_bias.mem"),16), parameters[name+"_bias_q16"].reshape(-1))
            np.testing.assert_array_equal(read_mem(folder/"threshold.mem",16), parameters["threshold_q16"])
            np.testing.assert_array_equal(read_mem(folder/"leak_reciprocal_q24.mem",32), parameters["leak_reciprocal_q24"])
            for split,(packed,ids) in splits.items():
                with np.load(seed_root/("golden_"+split+".npz"),allow_pickle=False) as golden:
                    np.testing.assert_array_equal(golden["sample_ids"],ids)
                    logits = golden[variant+"_window_logits_q16"]
                    np.testing.assert_array_equal(read_mem(folder/("golden_"+split+"_window_logits.mem"),16), logits.reshape(-1))
                    if split != "development" and not full:
                        continue
                    actual = run_recordings(packed,parameters,device=device,record=split=="development")
                    for key in ("window_logits_q16","predictions","scores"):
                        np.testing.assert_array_equal(actual[key],golden[variant+"_"+key],err_msg=f"{seed} {variant} {split} {key}")
                    comparisons += len(ids)
                if split == "development":
                    with np.load(folder/"golden_development_traces.npz",allow_pickle=False) as traces:
                        np.testing.assert_array_equal(traces["sample_ids"],ids)
                        for key in TRACE_FIELDS:
                            np.testing.assert_array_equal(actual[key],traces[key],err_msg=key)
                    independent = run_numpy(unpack_inputs(packed[:1])[:,0],parameters)
                    for key in (*TRACE_FIELDS,"window_logits_q16"):
                        np.testing.assert_array_equal(actual[key][:1,0],independent[key],err_msg=f"NumPy {seed} {variant} {key}")
                    independent_windows += 1
            configurations += 1
            print(f"Verified seed {seed} {variant}",flush=True)
    return dict(status="passed", physical_results=False, full=full, device=device,
        configurations=configurations, reproduced_recording_predictions=comparisons,
        independent_numpy_int64_windows=independent_windows,
        input_memory_images="all splits match packed inputs byte for byte",
        parameter_memory_images="all variants match floor-quantized source arrays and declared layout",
        traces="all development membranes, spikes and timestep logits reproduced",
        golden_window_logits="all compared as signed integers, not tolerance-based")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=Path("hardware/bundles/dvs_floor_q8q16_v1"))
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--full",action="store_true")
    parser.add_argument("--report",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = verify(args.bundle,args.device,args.full)
    if args.report:
        args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
