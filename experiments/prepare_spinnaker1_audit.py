"""Freeze label-free predictions and semantic bounds for the physical SHD audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.adapters.spinnaker1 import SpiNNaker1Mapping
from pines.models import DenseRecurrentSNN
from pines.semantics import ExecutionSemantics, NumericFormat
from pines.statistics import clopper_pearson_upper
from pines.torch_emulator import TorchEmulator

SEEDS = (1701, 2718, 3141, 5772, 8119)


def load_inputs(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata"]))
        events = np.unpackbits(d["packed_spikes"], axis=-1, bitorder="little")[..., :700]
        return events, d["sample_ids"].astype(str), meta


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    audit, audit_ids, audit_meta = load_inputs(args.bundle / "audit_inputs.npz")
    canary, canary_ids, canary_meta = load_inputs(args.bundle / "canary_inputs.npz")
    if set(audit_ids) & set(canary_ids):
        raise ValueError("physical semantic audit and canary must be disjoint")
    if audit_meta["split"] != "held-out-test-semantic-audit":
        raise ValueError("physical terms must refer to the same held-out population")
    reference = ExecutionSemantics(state_format=NumericFormat("float32"),
                                   weight_format=NumericFormat("float32"))
    mapping = SpiNNaker1Mapping(integration="match_source_euler")
    executor = TorchEmulator(device=args.device)
    alpha = 0.05 / (2 * args.paper_cells)
    rows = []
    arrays = {"audit_sample_ids": audit_ids, "canary_sample_ids": canary_ids}
    with threadpool_limits(limits=1), torch.no_grad():
        for seed in SEEDS:
            original = DenseRecurrentSNN.load(args.bundle / "models" / str(seed) / "original.npz")
            source = {}
            for split, events in (("audit", audit), ("canary", canary)):
                source[split] = np.concatenate([
                    executor.run(original, events[start:start+64], reference).predictions.cpu().numpy()
                    for start in range(0, len(events), 64)])
                arrays[f"seed{seed}_{split}_source"] = source[split]
            for variant in ("original", "reset_repaired"):
                model = DenseRecurrentSNN.load(args.bundle / "models" / str(seed) / (variant + ".npz"))
                mapped = {}
                for split, events in (("audit", audit), ("canary", canary)):
                    # Match the hardware runner's per-input floating emulator.
                    mapped[split] = np.asarray([mapping.emulate(model, x[None])["predictions"][0]
                                                for x in events], dtype=np.int64)
                    arrays[f"seed{seed}_{split}_{variant}_emulator"] = mapped[split]
                k = int(np.count_nonzero(source["audit"] != mapped["audit"]))
                upper = clopper_pearson_upper(k, len(audit), alpha)
                rows.append(dict(seed=seed, variant=variant, audit_samples=len(audit),
                                 disagreements=k, disagreement_rate=k/len(audit),
                                 semantic_upper=upper))
                print(f"seed {seed} {variant}: audit {k}/{len(audit)}, U_sem={100*upper:.3f} points", flush=True)
    np.savez_compressed(args.output / "paired_predictions.npz", **arrays)
    report = dict(status="frozen_label_free_semantic_audit", labels_accessed=False,
                  population="SHD held-out test pool", audit=audit_meta, canary=canary_meta,
                  confidence=0.95, paper_cells=args.paper_cells, alpha_per_term=alpha,
                  correction="Bonferroni over two terms for every task/backend/seed/repair cell",
                  source_semantics=reference.to_dict(), target_mapping=mapping.contract(),
                  rows=rows, physical_certificate=False)
    (args.output / "semantic_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--paper-cells", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())
