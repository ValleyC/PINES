"""Export frozen DVS models and inputs as an FPGA-specific handoff, no training."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.dvs_hardware import (BIASES, WEIGHTS, load_model, mapping_contract,
                                quantize_model, run_recordings, unpack_inputs)
from pines.hardware_bundle import json_scalar, write_mem, write_spike_mem
from pines.benchmarks.dvs_gesture import DVSGestureTrainConfig, build_dvs_conv_srnn
from pines.semantics import ExecutionSemantics, NumericFormat

SEEDS = (1701, 2718, 3141, 5772, 8119)
VARIANTS = {"unrepaired": "original", "repaired": "floor_repaired"}


def source_predictions(state, meta, packed, device, batch_size):
    config = DVSGestureTrainConfig(conv1_channels=state["conv1.weight"].shape[0],
        conv2_channels=state["conv2.weight"].shape[0], hidden_size=state["recurrent.weight"].shape[0],
        tau_mem=meta["tau_mem"], threshold=meta["threshold"],
        aggregation_temperature=meta["aggregation_temperature"])
    model = build_dvs_conv_srnn(32, 32, config).to(device).eval()
    model.load_state_dict({key: torch.as_tensor(value) for key, value in state.items()})
    semantics = ExecutionSemantics(state_format=NumericFormat("float32"), weight_format=NumericFormat("float32"))
    outputs = []
    with torch.no_grad():
        for start in range(0, len(packed), batch_size):
            x = unpack_inputs(packed[start:start+batch_size])
            logits = model(torch.as_tensor(x.reshape(-1, *x.shape[2:]), dtype=torch.float32, device=device), semantics)
            scores = torch.softmax(logits.reshape(len(x), 4, -1)/meta["aggregation_temperature"], dim=-1).mean(dim=1)
            outputs.append(scores.argmax(dim=-1).cpu().numpy().astype(np.int16))
    return np.concatenate(outputs)


def export(source, output, device, batch_size, source_bundle=False):
    root = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    output.mkdir(parents=True)
    common = output / "common"
    common.mkdir()
    for filename in ("README.md", "DATA_LICENSE.md"):
        shutil.copyfile(root / "hardware/templates" / ("dvs_hardware_bundle_"+filename), output / filename)
    contract = mapping_contract()
    (output / "target_contract.json").write_text(json.dumps(contract, indent=2)+"\n", encoding="utf-8")
    splits = {}
    for name in ("development", "audit", "canary"):
        path = (source / "common" if source_bundle else source) / (name+"_inputs.npz")
        with np.load(path, allow_pickle=False) as store:
            packed, ids, indices = store["packed_spikes"], store["sample_ids"], store["dataset_indices"]
        splits[name] = dict(packed=packed, ids=ids, indices=indices)
        shutil.copyfile(path, common / path.name)
        # Same packed events and row ordering, with an additional $readmemh image.
        frames = unpack_inputs(packed).reshape(len(packed), 4*60, 2048)
        write_spike_mem(common / (name+"_input_spikes.mem"), frames, 2048)
        with (common / (name+"_samples.csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("sample_index", "sample_id", "dataset_index", "first_mem_line", "windows", "steps_per_window"))
            writer.writerows((i, str(sid), int(indices[i]), i*240, 4, 60) for i, sid in enumerate(ids))
    if set(splits["audit"]["ids"]) & set(splits["canary"]["ids"]):
        raise ValueError("audit and canary splits overlap")
    manifest = dict(version="DVSFPGAHandoff/v1", status="software_golden_not_board_results",
        task="DVS Gesture", seeds=list(SEEDS), variants=list(VARIANTS),
        samples={name: len(split["ids"]) for name, split in splits.items()},
        windows_per_sample=4, timesteps_per_window=60, input_shape=[2,32,32],
        layer_shapes=dict(conv1=[32,14,14],conv2=[64,6,6],hidden=[256],readout=[11]),
        primary_hardware_split="canary", development_split="repair-calibration-development",
        audit_and_canary_selection_seed=20260914, labels_included=False,
        source_models="Frozen original and final floor-repaired manuscript checkpoints, unchanged",
        source_export="experiments/export_spinnaker1_dvs_bundle.py (models and inputs only, not target execution)",
        generator="experiments/export_dvs_hardware_bundle.py", contract_file="target_contract.json",
        evaluation_scope="Development evidence. Test performance informed earlier source-pipeline design.",
        hardware_scope="Both convolutions, recurrence and integer readout on FPGA. Four-window softmax aggregation on host.")
    for seed in SEEDS:
        seed_root = output / "seeds" / str(seed)
        seed_root.mkdir(parents=True)
        def model_path_for(variant, source_name):
            return (source / "seeds" / str(seed) / variant / "source_model.npz" if source_bundle
                    else source / "models" / str(seed) / (source_name+".npz"))
        original, source_meta = load_model(model_path_for("unrepaired", "original"))
        golden = {name: dict(sample_ids=split["ids"], dataset_indices=split["indices"],
                            reference_predictions=source_predictions(original, source_meta, split["packed"], device, batch_size))
                  for name, split in splits.items()}
        for variant, source_name in VARIANTS.items():
            model_path = model_path_for(variant, source_name)
            state, meta = load_model(model_path)
            if (meta["tau_mem"], meta["threshold"], meta["aggregation_temperature"], meta["sensor_width"], meta["sensor_height"]) != (3., 1., .5, 32, 32):
                raise ValueError("Unexpected source model contract")
            variant_root = seed_root / variant
            variant_root.mkdir()
            shutil.copyfile(model_path, variant_root / "source_model.npz")
            parameters = quantize_model(state, meta)
            np.savez_compressed(variant_root / "parameters_int.npz", **parameters, metadata=json_scalar(contract))
            for name in WEIGHTS:
                write_mem(variant_root / (name+"_weights.mem"), parameters[name+"_weights_q8"].reshape(-1), 8)
            for name in BIASES:
                write_mem(variant_root / (name+"_bias.mem"), parameters[name+"_bias_q16"], 16)
            write_mem(variant_root / "threshold.mem", parameters["threshold_q16"], 16)
            write_mem(variant_root / "leak_reciprocal_q24.mem", parameters["leak_reciprocal_q24"], 32)
            for name, split in splits.items():
                result = run_recordings(split["packed"], parameters, device=device, batch_size=batch_size, record=name=="development")
                for key in ("predictions", "window_logits_q16", "scores"):
                    golden[name][variant+"_"+key] = result[key]
                write_mem(variant_root / ("golden_"+name+"_window_logits.mem"), result["window_logits_q16"].reshape(-1), 16)
                if name == "development":
                    np.savez_compressed(variant_root / "golden_development_traces.npz", sample_ids=split["ids"], **result)
                print(f"seed {seed} {variant} {name}: {len(split['ids'])} recordings", flush=True)
        for name, arrays in golden.items():
            np.savez_compressed(seed_root / ("golden_"+name+".npz"), **arrays)
            with (seed_root / ("golden_"+name+"_predictions.csv")).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(("sample_index","sample_id","reference_prediction","unrepaired_prediction","repaired_prediction"))
                writer.writerows((i,str(sid),int(arrays["reference_predictions"][i]),int(arrays["unrepaired_predictions"][i]),int(arrays["repaired_predictions"][i])) for i,sid in enumerate(arrays["sample_ids"]))
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Export complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/spinnaker1_dvs_development_v2"))
    parser.add_argument("--source-bundle", type=Path, help="Re-export from an existing FPGA bundle")
    parser.add_argument("--output", type=Path, default=Path("hardware/bundles/dvs_floor_q8q16_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(args.source_bundle or args.source, args.output, args.device, args.batch_size, args.source_bundle is not None)
