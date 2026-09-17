"""Compare captured DVS window logits with the frozen FPGA golden outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.dvs_hardware import aggregate_q16


def read_capture(path, sample_count):
    rows = {}
    with path.open(newline="",encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sample,window = int(row["sample_index"]),int(row["window_index"])
            if not (0 <= sample < sample_count and 0 <= window < 4):
                raise ValueError("capture index outside the declared split")
            if (sample,window) in rows:
                raise ValueError("duplicate window: save repeated runs in separate files")
            logits = np.array([int(row[f"logit_{c}"]) for c in range(11)],dtype=np.int64)
            if np.any(logits < -32768) or np.any(logits > 32767):
                raise ValueError("capture logits must be signed Q16 integers")
            rows[sample,window] = logits
    complete = sorted(i for i in {i for i,_ in rows} if all((i,w) in rows for w in range(4)))
    logits = np.asarray([[rows[i,w] for w in range(4)] for i in complete],dtype=np.int16).reshape(-1,4,11)
    partial = sorted({i for i,_ in rows} - set(complete))
    return np.asarray(complete,dtype=np.int64), logits, partial


def analyze(bundle, capture, seed, variant, split, execution):
    with np.load(bundle/"seeds"/str(seed)/("golden_"+split+".npz"),allow_pickle=False) as data:
        sample_ids = data["sample_ids"]
        target_predictions = data[variant+"_predictions"]
        expected_logits = data[variant+"_window_logits_q16"]
        source_predictions = data["reference_predictions"]
    indices,logits,partial = read_capture(capture,len(sample_ids))
    predictions = aggregate_q16(logits)[0] if len(indices) else np.empty(0,dtype=np.int16)
    report = dict(execution=execution,seed=seed,variant=variant,split=split,
        complete_recordings=len(indices),expected_recordings=len(sample_ids),
        incomplete_sample_indices=partial,complete_split=len(indices)==len(sample_ids),
        prediction_disagreements=int(np.count_nonzero(predictions!=target_predictions[indices])),
        source_prediction_changes=int(np.count_nonzero(predictions!=source_predictions[indices])),
        differing_logit_entries=int(np.count_nonzero(logits!=expected_logits[indices])),
        max_absolute_logit_error_q16=int(np.abs(logits.astype(np.int64)-expected_logits[indices]).max()) if len(indices) else None,
        note="Comparison only. No ground-truth labels, accuracy estimate, or physical certificate is inferred.")
    return report,dict(sample_indices=indices,sample_ids=sample_ids[indices],predictions=predictions,window_logits_q16=logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=Path("hardware/bundles/dvs_floor_q8q16_v1"))
    parser.add_argument("--capture",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True,choices=[1701,2718,3141,5772,8119])
    parser.add_argument("--variant",choices=["unrepaired","repaired"],required=True)
    parser.add_argument("--split",choices=["development","audit","canary"],default="canary")
    parser.add_argument("--execution",choices=["rtl","board"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report,arrays = analyze(args.bundle,args.capture,args.seed,args.variant,args.split,args.execution)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/"predictions.npz",**arrays)
    (args.output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
