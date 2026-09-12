"""Compare physical DVS development windows with frozen ideal layer traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(capture,bundle):
    config = json.loads((capture / "config.json").read_text())
    rows = []
    for folder in sorted(capture.glob("input_*")):
        index = int(folder.name.split("_")[1])
        for window_folder in sorted(folder.glob("window_*")):
            window = int(window_folder.name.split("_")[1])
            for variant in config["variants"]:
                physical = window_folder / (variant+".npz")
                if not physical.exists():
                    continue
                expected = bundle / "development_traces" / str(config["seed"]) / variant / f"input_{index:05d}.npz"
                with np.load(physical) as hw, np.load(expected) as emulator:
                    row = dict(input_index=index,window=window,variant=variant,
                               physical_window_prediction=int(hw["window_logits"].argmax()),
                               emulator_window_prediction=int(emulator["window_logits"][window].argmax()),
                               max_absolute_logit_difference=float(np.max(np.abs(hw["window_logits"]-emulator["window_logits"][window]))))
                    for layer in ("conv1","conv2","hidden"):
                        key = layer+"_spikes"
                        if key not in hw:
                            continue
                        observed,target = hw[key],emulator[layer][window]
                        differences = observed != target
                        steps = np.flatnonzero(differences.any(axis=1))
                        row[layer] = dict(physical_spikes=int(observed.sum()),emulator_spikes=int(target.sum()),
                            differing_time_neuron_cells=int(differences.sum()),
                            first_difference_step=int(steps[0]) if len(steps) else None)
                    rows.append(row)
    return dict(status="development_window_diagnostic",physical_certificate=False,
        note="Window predictions are diagnostic. One DVS classification requires all four windows. No population bound is estimated here.",
        rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",type=Path,required=True)
    parser.add_argument("--bundle",type=Path,required=True)
    args = parser.parse_args()
    report = analyze(args.capture,args.bundle)
    (args.capture / "development_analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
