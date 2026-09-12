"""Compute label-free DVS semantic evidence after freezing the device mapping.

The source is the original float32 checkpoint executor for both model variants.
One prediction aggregates all four windows of a recording. No labels are loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.adapters.spinnaker1_dvs import SpiNNaker1DVSMapping, aggregate_windows
from pines.benchmarks.dvs_gesture import DVSGestureTrainConfig, build_dvs_conv_srnn
from pines.semantics import ExecutionSemantics, NumericFormat
from pines.statistics import clopper_pearson_upper
from run_spinnaker1_dvs import load_model


def unpack_recordings(packed, meta):
    width, height = meta["sensor_width"], meta["sensor_height"]
    events = np.unpackbits(packed, axis=-1, bitorder="little")[..., :2*width*height]
    return events.reshape(*packed.shape[:-1], 2, height, width)


def source_predictions(state, meta, packed, device, batch_size=4):
    config = DVSGestureTrainConfig(
        conv1_channels=state["conv1.weight"].shape[0],
        conv2_channels=state["conv2.weight"].shape[0],
        hidden_size=state["recurrent.weight"].shape[0],
        tau_mem=meta["tau_mem"], threshold=meta["threshold"],
        aggregation_temperature=meta["aggregation_temperature"],
    )
    model = build_dvs_conv_srnn(meta["sensor_width"], meta["sensor_height"], config,
                               output_size=state["readout.weight"].shape[0]).to(device).eval()
    model.load_state_dict({key: torch.as_tensor(value) for key, value in state.items()})
    reference = ExecutionSemantics(state_format=NumericFormat("float32"),
                                   weight_format=NumericFormat("float32"))
    predictions = []
    with torch.no_grad():
        for start in range(0, len(packed), batch_size):
            events = unpack_recordings(packed[start:start+batch_size], meta)
            samples, windows = events.shape[:2]
            x = torch.as_tensor(events.reshape(samples*windows, *events.shape[2:]),
                                dtype=torch.float32, device=device)
            logits = model(x, reference).reshape(samples, windows, -1)
            # Match the original source executor's float32 aggregation.
            if meta["aggregation_temperature"] > 0:
                logits = torch.softmax(logits/meta["aggregation_temperature"], dim=-1)
            predictions.extend(logits.mean(dim=1).argmax(dim=1).cpu().tolist())
    return np.asarray(predictions, dtype=np.int64)


def mapped_predictions(state, meta, packed, device):
    mapping = SpiNNaker1DVSMapping()
    predictions = []
    for sample in packed:
        events = unpack_recordings(sample, meta)
        result = mapping.emulate(state, events, meta["tau_mem"], meta["threshold"], device=device)
        predictions.append(int(aggregate_windows(result["window_logits"], meta["aggregation_temperature"]).argmax()))
    return np.asarray(predictions, dtype=np.int64)


def run(args):
    protocol = json.loads((args.bundle / "bundle.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    splits, arrays = {}, {}
    for split in ("audit", "canary"):
        with np.load(args.bundle / f"{split}_inputs.npz") as data:
            splits[split] = data["packed_spikes"]
            arrays[f"{split}_sample_ids"] = data["sample_ids"].astype(str)
            arrays[f"{split}_dataset_indices"] = data["dataset_indices"]
    if set(arrays["audit_sample_ids"]) & set(arrays["canary_sample_ids"]):
        raise ValueError("The declared DVS audit and canary splits must be disjoint")
    rows = []
    alpha = .05/(2*args.paper_cells)
    for seed in protocol["seeds"]:
        folder = args.bundle / "models" / str(seed)
        original, meta = load_model(folder / "original.npz")
        source = {split: source_predictions(original, meta, packed, args.device)
                  for split, packed in splits.items()}
        arrays.update({f"seed{seed}_{split}_source": values for split, values in source.items()})
        for variant in protocol["variants"]:
            state, target_meta = load_model(folder / f"{variant}.npz")
            target = {split: mapped_predictions(state, target_meta, packed, args.device)
                      for split, packed in splits.items()}
            arrays.update({f"seed{seed}_{split}_{variant}_emulator": values for split, values in target.items()})
            n = len(splits["audit"])
            k = int(np.count_nonzero(source["audit"] != target["audit"]))
            upper = clopper_pearson_upper(k, n, alpha)
            rows.append(dict(seed=seed, variant=variant, audit_samples=n, disagreements=k,
                             disagreement_rate=k/n, semantic_upper=upper))
            print(f"seed {seed} {variant}: {k}/{n}, U_sem={100*upper:.3f} points", flush=True)
    np.savez_compressed(args.output / "paired_predictions.npz", **arrays)
    report = dict(status="frozen_label_free_semantic_audit", physical_certificate=False,
        labels_accessed=False, task="DVS Gesture", population="DVS Gesture held-out test pool",
        confidence=.95, paper_cells=args.paper_cells, alpha_per_term=alpha,
        correction="Bonferroni over two terms for every task/backend/seed/repair cell",
        source="Original float32 source executor and original four-window aggregation for both variants",
        target_mapping=SpiNNaker1DVSMapping().contract(),
        repair=protocol["repair"], windows_per_observation=4,
        audit_samples=len(splits["audit"]), canary_samples=len(splits["canary"]),
        zero_disagreement_total_floor=min(1.,
            clopper_pearson_upper(0, len(splits["audit"]), alpha)
            + clopper_pearson_upper(0, len(splits["canary"]), alpha)),
        evaluation_scope="Development evidence. Earlier test performance informed the source DVS pipeline.",
        rows=rows)
    (args.output / "semantic_audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-cells", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(parser.parse_args())
