"""Audit the frozen integer FPGA mapping on the physical SHD test population.

Reuses the existing FPGA parameter files without changing them. New held-out
canary inputs and integer predictions are exported separately from the old RTL
simulation bundle. Ground-truth labels are not loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.hardware_bundle import run_floor_q8q16_integer, write_spike_mem
from pines.models import DenseRecurrentSNN
from pines.semantics import ExecutionSemantics
from pines.statistics import clopper_pearson_upper
from pines.torch_emulator import TorchEmulator

SEEDS = (1701, 2718, 3141, 5772, 8119)


def integer_predictions(frames, parameters, batch_size=64):
    predictions, logits = [], []
    args = {key: parameters[key] for key in (
        "input_weights_q8", "recurrent_weights_q8", "output_weights_q8", "bias_q16", "threshold_q16")}
    args["leak_reciprocal_q24_values"] = parameters["leak_reciprocal_q24"]
    for start in range(0, len(frames), batch_size):
        result = run_floor_q8q16_integer(frames[start:start+batch_size], **args)
        predictions.append(result["predictions"])
        logits.append(result["final_logits_q16"])
    return np.concatenate(predictions), np.concatenate(logits)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    args.audit_output.mkdir(parents=True, exist_ok=False)
    common = args.output / "common"
    common.mkdir()
    splits, arrays, input_metadata = {}, {}, {}
    for split in ("audit", "canary"):
        with np.load(args.input_bundle / f"{split}_inputs.npz") as data:
            meta = json.loads(str(data["metadata"]))
            splits[split] = np.unpackbits(data["packed_spikes"], axis=-1, bitorder="little")[..., :meta["input_channels"]]
            arrays[f"{split}_sample_ids"] = data["sample_ids"].astype(str)
            input_metadata[split] = meta
    if set(arrays["audit_sample_ids"]) & set(arrays["canary_sample_ids"]):
        raise ValueError("The semantic audit and physical canary must be disjoint")
    shutil.copyfile(args.input_bundle / "canary_inputs.npz", common / "canary_inputs.npz")
    write_spike_mem(common / "canary_input_spikes.mem", splits["canary"], splits["canary"].shape[-1])
    with (common / "canary_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_position", "sample_id", "first_mem_line", "time_bins"])
        for index, sample in enumerate(arrays["canary_sample_ids"]):
            writer.writerow([index, sample, index*splits["canary"].shape[1], splits["canary"].shape[1]])
    attribution = (args.parameter_bundle / "DATA_LICENSE.md").read_text().replace(
        "unlabeled certificate-audit subset", "unlabeled held-out physical-canary subset")
    (args.output / "DATA_LICENSE.md").write_text(attribution)
    template = Path(__file__).resolve().parents[1] / "hardware/templates/shd_virtex7_canary_README.md"
    shutil.copyfile(template, args.output / "README.md")
    semantics = json.loads((args.parameter_bundle / "semantics/reference.json").read_text())
    reference = ExecutionSemantics.from_dict(semantics)
    executor = TorchEmulator(device=args.device)
    alpha = .05/(2*args.paper_cells)
    rows = []
    with threadpool_limits(limits=1), torch.no_grad():
        for seed in SEEDS:
            folder = args.parameter_bundle / "seeds" / str(seed)
            model = DenseRecurrentSNN.load(folder / "unrepaired/original_model.npz")
            source = {}
            for split, frames in splits.items():
                source[split] = np.concatenate([
                    executor.run(model, frames[start:start+64], reference).predictions.cpu().numpy()
                    for start in range(0, len(frames), 64)])
                arrays[f"seed{seed}_{split}_source"] = source[split]
            golden = dict(sample_ids=arrays["canary_sample_ids"], reference_predictions=source["canary"])
            for variant in ("unrepaired", "repaired"):
                with np.load(folder / variant / "parameters_int.npz") as parameters:
                    for split, frames in splits.items():
                        predictions, logits = integer_predictions(frames, parameters)
                        arrays[f"seed{seed}_{split}_{variant}_emulator"] = predictions
                        if split == "canary":
                            golden[f"{variant}_emulator_predictions"] = predictions
                            golden[f"{variant}_final_logits_q16"] = logits
                n = len(splits["audit"])
                k = int(np.count_nonzero(source["audit"] != arrays[f"seed{seed}_audit_{variant}_emulator"]))
                upper = clopper_pearson_upper(k, n, alpha)
                rows.append(dict(seed=seed, variant=variant, audit_samples=n, disagreements=k,
                    disagreement_rate=k/n, semantic_upper=upper))
                print(f"seed {seed} {variant}: {k}/{n}, U_sem={100*upper:.3f} points", flush=True)
            destination = args.output / "seeds" / str(seed)
            destination.mkdir(parents=True)
            np.savez_compressed(destination / "golden_canary.npz", **golden)
    np.savez_compressed(args.audit_output / "paired_predictions.npz", **arrays)
    report = dict(status="frozen_label_free_semantic_audit", physical_certificate=False,
        labels_accessed=False, population="SHD held-out test pool", audit=input_metadata["audit"],
        canary=input_metadata["canary"], confidence=.95, paper_cells=args.paper_cells, alpha_per_term=alpha,
        correction="Bonferroni over two terms for every task/backend/seed/repair cell",
        source_semantics=semantics,
        target_mapping="Existing floor Q8.6 weights and Q16.8 state parameters with Q0.24 leak, evaluated by exact integer transitions",
        parameter_bundle="hardware/bundles/shd_floor_q8q16_v1", rows=rows,
        five_seed_semantic_means={variant: float(np.mean([r["semantic_upper"] for r in rows if r["variant"] == variant]))
                                 for variant in ("unrepaired", "repaired")})
    (args.audit_output / "semantic_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    bundle = dict(purpose="Held-out physical Virtex-7 SHD canary, not the earlier training-pool RTL batch",
        parameter_bundle="../shd_floor_q8q16_v1", parameter_changes=False, seeds=list(SEEDS),
        canary_samples=len(splits["canary"]), audit_samples=len(splits["audit"]), labels_included=False,
        audit_results="results/virtex7_shd", physical_measurements_completed=False,
        input_order="common/canary_samples.csv", input_memory="common/canary_input_spikes.mem",
        expected_predictions="seeds/<seed>/golden_canary.npz")
    (args.output / "bundle.json").write_text(json.dumps(bundle, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-bundle", type=Path, default=Path("hardware/bundles/shd_floor_q8q16_v1"))
    parser.add_argument("--input-bundle", type=Path, default=Path("hardware/bundles/shd_spinnaker1_v1"))
    parser.add_argument("--output", type=Path, default=Path("hardware/bundles/shd_virtex7_canary_v2"))
    parser.add_argument("--audit-output", type=Path, default=Path("results/virtex7_shd"))
    parser.add_argument("--paper-cells", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(parser.parse_args())
