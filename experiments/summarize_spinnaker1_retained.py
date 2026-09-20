"""Reproduce manuscript measurements from the retained primary SpiNNaker runs.

The returned inputs are sorted prefixes, not new random canary samples. Bounds
therefore retain the original planned denominator and charge every unresolved
pair as a disagreement. Existing complete-campaign analyzers are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pines.adapters.spinnaker1_dvs import aggregate_windows
from pines.statistics import clopper_pearson_upper


def unresolved_upper(errors, observed, planned, alpha):
    if not 0 <= errors <= observed <= planned or planned < 1:
        raise ValueError("Require 0 <= errors <= observed <= planned and planned > 0")
    return clopper_pearson_upper(errors+planned-observed,planned,alpha)


def extract(root, task):
    folder = root/"artifacts"/(
        "spinnaker1_shd_primary_first100" if task=="shd" else "spinnaker1_dvs_primary_seed1701_first10")/"capture"
    audit = root/"results"/("spinnaker1_"+task)
    with np.load(audit/"paired_predictions.npz",allow_pickle=False) as store:
        paired = {key:store[key] for key in store.files}
    semantic = json.loads((audit/"semantic_audit.json").read_text())
    ids = paired["canary_sample_ids"].astype(str)
    positions = {sid:i for i,sid in enumerate(ids)}
    config = json.loads((folder/"config.json").read_text())
    observations,spike_arrays,logit_arrays = [],[],[]
    for path in sorted(folder.glob("input_*/summary.json")):
        sample = json.loads(path.read_text())
        sid = sample["sample_id"]
        index = positions[sid]
        if task=="shd":
            assert sample["status"]=="physical_capture_completed"
            entries = [(r["seed"],r["variant"],r["prediction"],r["emulator_prediction"]) for r in sample["rows"]]
        else:
            assert sample["status"]=="physical_classification_capture_completed" and sample["windows"]==[0,1,2,3]
            entries = [(config["seed"],variant,prediction,None) for variant,prediction in sample["predictions"].items()]
        for seed,variant,prediction,stored_emulator in entries:
            emulator = int(paired[f"seed{seed}_canary_{variant}_emulator"][index])
            if stored_emulator is not None:
                assert emulator==stored_emulator
            if task=="shd":
                with np.load(path.parent/f"seed{seed}_{variant}.npz",allow_pickle=False) as data:
                    assert str(data["sample_id"])==sid and int(data["prediction"])==prediction
                    logits,spikes = data["logits"],data["spikes"]
                    assert logits.argmax()==prediction
            else:
                windows,spikes = [],[]
                for window in range(4):
                    with np.load(path.parent/f"window_{window}"/(variant+".npz"),allow_pickle=False) as data:
                        windows.append(data["window_logits"])
                        spikes.append(data["hidden_spikes"])
                logits,spikes=np.stack(windows),np.stack(spikes)
                assert aggregate_windows(logits,.5).argmax()==prediction
            observations.append(dict(task=task,seed=seed,variant=variant,sample_id=sid,
                canary_index=index,source_prediction=int(paired[f"seed{seed}_canary_source"][index]),
                emulator_prediction=emulator,hardware_prediction=int(prediction)))
            spike_arrays.append(spikes)
            logit_arrays.append(logits)
    expected = 1000 if task=="shd" else 20
    assert len(observations)==expected
    assert len({(r["seed"],r["variant"],r["sample_id"]) for r in observations})==expected
    # Measurements above and the complete retained index set are fixed before
    # evaluation labels are read. Labels never choose observations or models.
    label_path=root/"data/processed"/("shd_v1" if task=="shd" else "dvs_gesture_v2")/"test.npz"
    with np.load(label_path,allow_pickle=False) as data:
        if task=="shd":
            by_id=dict(zip(data["sample_ids"].astype(str),data["labels"]))
            truth={sid:int(by_id[sid]) for sid in ids}
        else:
            truth=dict(zip(ids,data["labels"][paired["canary_dataset_indices"]].astype(int)))
    for row in observations:
        row["evaluation_label"]=int(truth[row["sample_id"]])
    traces=dict(sample_ids=np.array([r["sample_id"] for r in observations]),
        seeds=np.array([r["seed"] for r in observations]),variants=np.array([r["variant"] for r in observations]),
        hidden_spikes=np.stack(spike_arrays),logits=np.stack(logit_arrays))
    return observations,traces,semantic,config


def summarize(observations,semantics):
    rows=[]
    for s in semantics["rows"]:
        selected=[r for r in observations if (r["seed"],r["variant"])==(s["seed"],s["variant"])]
        planned=861 if observations[0]["task"]=="shd" else 160
        n=len(selected)
        k=sum(r["hardware_prediction"]!=r["emulator_prediction"] for r in selected)
        upper=unresolved_upper(k,n,planned,semantics["alpha_per_term"])
        row=dict(seed=s["seed"],variant=s["variant"],observed_pairs=n,planned_pairs=planned,
            observed_hardware_disagreements=k,unresolved_pairs=planned-n,
            bound_failure_count=k+planned-n,semantic_upper=s["semantic_upper"],
            conformance_upper=upper,total_upper=min(1.,s["semantic_upper"]+upper))
        row["verdicts"]={str(b):"accept" if row["total_upper"]<=b else "reject" for b in (.01,.02,.05)}
        if n:
            source=np.array([r["source_prediction"] for r in selected])
            emulator=np.array([r["emulator_prediction"] for r in selected])
            hardware=np.array([r["hardware_prediction"] for r in selected])
            labels=np.array([r["evaluation_label"] for r in selected])
            a0,ah=float(np.mean(source==labels)),float(np.mean(hardware==labels))
            row.update(source_accuracy=a0,hardware_accuracy=ah,accuracy_loss=a0-ah,
                absolute_accuracy_change=abs(a0-ah),hardware_disagreement_rate=k/n,
                source_emulator_disagreement=float(np.mean(source!=emulator)),
                source_hardware_disagreement=float(np.mean(source!=hardware)))
        rows.append(row)
    means={}
    metrics=("source_accuracy","hardware_accuracy","accuracy_loss","absolute_accuracy_change",
        "hardware_disagreement_rate","source_emulator_disagreement","source_hardware_disagreement",
        "semantic_upper","conformance_upper","total_upper")
    for variant in dict.fromkeys(r["variant"] for r in rows):
        selected=[r for r in rows if r["variant"]==variant and r["observed_pairs"]]
        means[variant]={metric:float(np.mean([r[metric] for r in selected])) for metric in metrics}
        means[variant].update(seeds=[r["seed"] for r in selected],samples_per_seed=[r["observed_pairs"] for r in selected])
    return dict(alpha_per_term=semantics["alpha_per_term"],rows=rows,measured_seed_means=means)


def verify_saved(output):
    root=Path(__file__).resolve().parents[1]
    report=json.loads((output/"summary.json").read_text())
    with (output/"observations.csv").open(newline="",encoding="utf-8") as handle:
        observations=list(csv.DictReader(handle))
    for row in observations:
        for key in ("seed","canary_index","source_prediction","emulator_prediction","hardware_prediction","evaluation_label"):
            row[key]=int(row[key])
    for task in ("shd","dvs"):
        rows=[r for r in observations if r["task"]==task]
        semantic=json.loads((root/"results"/("spinnaker1_"+task)/"semantic_audit.json").read_text())
        calculated=summarize(rows,semantic)
        assert calculated["rows"]==report[task]["rows"]
        assert calculated["measured_seed_means"]==report[task]["measured_seed_means"]
        with np.load(output/(task+"_captured_traces.npz"),allow_pickle=False) as data:
            assert data["sample_ids"].tolist()==[r["sample_id"] for r in rows]
            assert data["seeds"].tolist()==[r["seed"] for r in rows]
            assert data["variants"].tolist()==[r["variant"] for r in rows]
            preds=data["logits"].argmax(axis=-1) if task=="shd" else aggregate_windows(data["logits"],.5).argmax(axis=-1)
            np.testing.assert_array_equal(preds,[r["hardware_prediction"] for r in rows])
        with np.load(root/"results"/("spinnaker1_"+task)/"paired_predictions.npz",allow_pickle=False) as data:
            for row in rows:
                seed,variant,i=row["seed"],row["variant"],row["canary_index"]
                assert str(data["canary_sample_ids"][i])==row["sample_id"]
                assert int(data[f"seed{seed}_canary_source"][i])==row["source_prediction"]
                assert int(data[f"seed{seed}_canary_{variant}_emulator"][i])==row["emulator_prediction"]
    print(f"Verified {len(observations)} physical predictions, paired source/emulator values, per-seed metrics and unresolved-pair bounds.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/spinnaker1_retained"))
    parser.add_argument("--verify-only",action="store_true",help="Check the published files without private raw captures or datasets")
    args=parser.parse_args()
    if args.verify_only:
        verify_saved(args.output)
        return
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(status="retained_primary_measurements_with_unresolved_pair_bounds",
        completed_jobs=dict(shd="shd-primary",dvs="dvs-primary"),
        selection="All returned observations from the two completed aligned-reset primary jobs. No outcome filtering.",
        sampling="Completed batches are prefixes of dataset-index-sorted canary sets, not independently sampled smaller audits.",
        confidence=.95,paper_cells=40,alpha_per_term=.05/80,
        bound="Clopper-Pearson(k + N - n, N, alpha), unresolved planned pairs counted as disagreements; total capped at one.",
        justification="For the predeclared N-pair audit, its eventual error count is at most k+N-n. Monotonicity preserves the original simultaneous bound without assuming the returned prefix is representative.",
        assumptions=["The original planned audit sampling and independent execution assumptions hold.",
                     "Models, mapping and execution profile remain fixed.",
                     "Accuracy and disagreement point estimates describe recorded inputs only."],
        labels="Used only for descriptive accuracy after freezing the retained observation set.",
        calibrated_repeat_variability="Not estimated from these primary batches.")
    all_rows=[]
    for task in ("shd","dvs"):
        observations,traces,semantic,config=extract(root,task)
        np.savez_compressed(args.output/(task+"_captured_traces.npz"),**traces)
        all_rows.extend(observations)
        report[task]=summarize(observations,semantic)
        report[task]["execution_profile"]=config
    with (args.output/"observations.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(all_rows)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({task:report[task]["measured_seed_means"] for task in ("shd","dvs")},indent=2))


if __name__=="__main__":
    main()
