"""Compare saved physical SHD traces with the declared ideal mapped executor.

This is a trace diagnostic. It neither accesses labels nor issues a population
certificate. Development captures cannot be promoted to a held-out audit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.adapters.spinnaker1 import SpiNNaker1Mapping
from pines.models import DenseRecurrentSNN


def compare_trace(hardware: np.ndarray, ideal: np.ndarray) -> dict:
    if hardware.shape != ideal.shape:
        raise ValueError("hardware and emulator spike windows differ")
    differing = np.argwhere(hardware != ideal)
    return dict(hardware_spikes=int(hardware.sum()), emulator_spikes=int(ideal.sum()),
                differing_spike_cells=len(differing),
                first_differing_step=int(differing[0, 0]) if len(differing) else None)


def analyze(capture_dir: Path, model_path: Path, inputs_path: Path) -> dict:
    config = json.loads((capture_dir / "config.json").read_text())
    mapping = SpiNNaker1Mapping(integration=config["contract"]["integration_mapping"])
    model = DenseRecurrentSNN.load(model_path)
    with np.load(inputs_path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        ids = data["sample_ids"].astype(str)
        inputs = np.unpackbits(data["packed_spikes"], axis=-1,
                              bitorder="little")[..., :meta["input_channels"]]
    lookup = {sid: i for i, sid in enumerate(ids)}
    rows = []
    for path in sorted(capture_dir.glob("sample_*.npz")):
        with np.load(path, allow_pickle=False) as sample:
            sid = str(sample["sample_id"])
            ideal = mapping.emulate(model, inputs[lookup[sid]:lookup[sid]+1])
            replay_prediction = int(ideal["predictions"][0])
            if replay_prediction != int(sample["emulator_prediction"]):
                raise ValueError("saved emulator prediction differs from replay")
            hw_logits = sample["spikes"].sum(axis=0) @ model.output_weights
            np.testing.assert_allclose(hw_logits, sample["logits"], atol=1e-12)
            if int(hw_logits.argmax()) != int(sample["prediction"]):
                raise ValueError("saved hardware prediction differs from recorded-spike readout")
            rows.append(dict(sample_id=sid, pair_id=str(sample["pair_id"]),
                             hardware_prediction=int(sample["prediction"]),
                             emulator_prediction=replay_prediction,
                             **compare_trace(sample["spikes"], ideal["spikes"][0])))
    return dict(status="trace_diagnostic", dataset_split=meta["split"],
                samples=len(rows),
                disagreements=sum(r["hardware_prediction"] != r["emulator_prediction"] for r in rows),
                exact_spike_matches=sum(r["differing_spike_cells"] == 0 for r in rows),
                scope="Hybrid hardware hidden dynamics and host readout. No accuracy or population guarantee.",
                rows=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = json.dumps(analyze(args.capture_dir, args.model, args.inputs), indent=2)
    if args.output:
        with args.output.open("x") as f:
            f.write(result + "\n")
    print(result)
