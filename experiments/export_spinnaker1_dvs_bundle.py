"""Export DVS models, disjoint held-out inputs, and label-free development traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.adapters.spinnaker1_dvs import SpiNNaker1DVSMapping, aggregate_windows

SEEDS = (1701,2718,3141,5772,8119)


def export(root, output, device):
    output.mkdir(parents=True, exist_ok=False)
    source_root = root / "artifacts/dvs_gesture_v3"
    summary = json.loads((root / "results/dvs_gesture_v3/repair_floor_task_tuned_clean_v5_summary.json").read_text())
    repair_root = root / summary["protocol"]["method_artifact_roots"]["certificate_directed"].replace("\\", "/")
    with np.load(source_root / "seed_1701/split_indices.npz") as split:
        development = split["repair_calibration"][:2]
    mapping = SpiNNaker1DVSMapping()
    with np.load(root / "data/processed/dvs_gesture_v2/train.npz") as store:
        packed = store["packed"][development]
        # Original dataset filenames encode the class. Hardware bundles use
        # row identifiers instead, so input names do not reveal annotations.
        ids = np.asarray([f"dvs-train-{index:05d}" for index in development])
        metadata = json.loads(str(store["metadata"]))
        np.savez_compressed(output / "development_inputs.npz", packed_spikes=packed,
            sample_ids=ids, dataset_indices=development, metadata=json.dumps(dict(dataset="DVS Gesture", split="repair-calibration-development",
                labels_included=False, layout="sample,window,time,packed-CHW", preprocessing=metadata["config"])))
    events = np.unpackbits(packed, axis=-1, bitorder="little")[...,:2048].reshape(len(ids),4,60,2,32,32)
    with np.load(root / "data/processed/dvs_gesture_v2/test.npz") as store:
        total = len(store["sample_ids"])
        audit = np.sort(np.random.default_rng(20260914).choice(total,104,replace=False))
        canary = np.setdiff1d(np.arange(total),audit)
        for name, indices in (("audit",audit), ("canary",canary)):
            np.savez_compressed(output / f"{name}_inputs.npz", packed_spikes=store["packed"][indices],
                sample_ids=np.asarray([f"dvs-test-{index:05d}" for index in indices]), dataset_indices=indices, metadata=json.dumps(dict(dataset="DVS Gesture",
                    split=f"held-out-test-{name}", labels_included=False, selection_seed=20260914,
                    layout="sample,window,time,packed-CHW", preprocessing=metadata["config"])))
    development_rows = []
    for seed in SEEDS:
        folder = output / "models" / str(seed)
        folder.mkdir(parents=True)
        for variant, path in (("original",source_root / f"seed_{seed}/checkpoint.pt"),
                              ("floor_repaired",repair_root / f"seed_{seed}/floor_rounding_saturation/certificate_directed/repaired_checkpoint.pt")):
            checkpoint = torch.load(path,map_location="cpu",weights_only=False)
            state = {name:value.detach().cpu().numpy() for name,value in checkpoint["state_dict"].items()}
            config = checkpoint["config"]
            model_meta = dict(seed=seed,variant=variant,tau_mem=config["tau_mem"],threshold=config["threshold"],
                aggregation_temperature=config["aggregation_temperature"],sensor_width=checkpoint["sensor_width"],
                sensor_height=checkpoint["sensor_height"],parameter_orientation="PyTorch OIHW kernels and destination,source linear weights")
            np.savez_compressed(folder / (variant+".npz"), metadata=json.dumps(model_meta), **state)
            for index, event in enumerate(events):
                result = mapping.emulate(state,event,model_meta["tau_mem"],model_meta["threshold"],device=device,record=True)
                prediction = int(aggregate_windows(result["window_logits"],model_meta["aggregation_temperature"]).argmax())
                trace_folder = output / "development_traces" / str(seed) / variant
                trace_folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(trace_folder / f"input_{index:05d}.npz", sample_id=ids[index],
                    prediction=prediction, **result)
                development_rows.append(dict(seed=seed,variant=variant,sample_id=str(ids[index]),prediction=prediction,
                    hidden_spikes=int(result["hidden"].sum()), conv1_spikes=int(result["conv1"].sum()),conv2_spikes=int(result["conv2"].sum())))
            print(f"Exported seed {seed} {variant} and four-window development traces",flush=True)
    protocol = dict(status="development_mapping_not_physical_results",task="DVS Gesture",
        mapping=mapping.contract(),seeds=SEEDS,variants=["original","floor_repaired"],development_samples=len(ids),
        audit_samples=len(audit),canary_samples=len(canary),windows_per_input=4,horizon_per_window=60,
        neuron_counts=dict(conv1=6272,conv2=2304,recurrent=256),population="held-out test pool",
        repair="Existing floor-target repair is evaluated under this new mapped target, not claimed to be retrained for native SpiNNaker arithmetic.",
        selection="Development traces only. Audit and canary predictions are not used to select the mapping.",
        confidence=dict(level=.95,paper_cells=40,alpha_per_term=.05/(2*40)),development_rows=development_rows)
    (output / "bundle.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print(f"Saved {output}; audit={len(audit)}, canary={len(canary)}; no labels exported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    export(Path(__file__).resolve().parents[1],args.output.resolve(),args.device)
