"""Combine complete four-window DVS hardware observations with the frozen audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.statistics import clopper_pearson_upper
from analyze_spinnaker1_matrix import capture_profiles


def collect(captures):
    observations, diagnostics = {}, []
    for capture in captures:
        config = json.loads((capture / "config.json").read_text())
        for path in sorted(capture.glob("input_*/summary.json")):
            sample = json.loads(path.read_text())
            if sample["status"] != "physical_classification_capture_completed":
                continue
            if sample["windows"] != [0, 1, 2, 3]:
                raise ValueError("A DVS observation requires the four declared windows")
            for variant, prediction in sample["predictions"].items():
                key = (config["seed"], variant, sample["sample_id"])
                if key in observations:
                    raise ValueError(f"Duplicate primary DVS observation: {key}")
                observations[key] = int(prediction)
            for window in sample["windows"]:
                summary = json.loads((path.parent / f"window_{window}" / "summary.json").read_text())
                diagnostics.append(summary["diagnostics"])
    return observations, diagnostics


def analyze(captures, audit, labels=None):
    semantic = json.loads((audit / "semantic_audit.json").read_text())
    observations, diagnostics = collect(captures)
    with np.load(audit / "paired_predictions.npz") as paired:
        ids = paired["canary_sample_ids"].astype(str)
        counts = [sum((r["seed"], r["variant"], sample) in observations for sample in ids)
                  for r in semantic["rows"]]
        if sum(counts) != len(observations):
            raise ValueError("Captures contain recordings or conditions outside the frozen DVS canary")
        if any(n != len(ids) for n in counts):
            return dict(status="incomplete_capture", physical_certificate=False,
                observations=len(observations), expected_observations=len(ids)*len(counts),
                samples_per_condition=counts, windows_per_observation=4)
        truth = None
        if labels is not None:
            with np.load(labels) as data:
                truth = data["labels"][paired["canary_dataset_indices"]]
        rows = []
        for s in semantic["rows"]:
            seed, variant = s["seed"], s["variant"]
            hardware = np.asarray([observations[(seed, variant, sample)] for sample in ids])
            emulator = paired[f"seed{seed}_canary_{variant}_emulator"]
            source = paired[f"seed{seed}_canary_source"]
            k = int(np.count_nonzero(hardware != emulator))
            upper = clopper_pearson_upper(k, len(ids), semantic["alpha_per_term"])
            total = min(1., s["semantic_upper"]+upper)
            row = dict(seed=seed, variant=variant, canary_samples=len(ids),
                hardware_disagreements=k, hardware_disagreement_rate=k/len(ids),
                semantic_upper=s["semantic_upper"], conformance_upper=upper, total_upper=total,
                verdicts={str(b): "accept" if total <= b else "reject" for b in (.01,.02,.05)},
                observed_source_hardware_disagreement=float((source != hardware).mean()))
            if truth is not None:
                source_acc, hardware_acc = float((source == truth).mean()), float((hardware == truth).mean())
                change = source_acc-hardware_acc
                row.update(source_accuracy=source_acc, hardware_accuracy=hardware_acc,
                    accuracy_loss=change, absolute_accuracy_change=abs(change), observed_slack=total-abs(change))
            rows.append(row)
    means = {}
    for variant in dict.fromkeys(r["variant"] for r in rows):
        selected = [r for r in rows if r["variant"] == variant]
        metrics = ["hardware_disagreement_rate", "semantic_upper", "conformance_upper", "total_upper"]
        if truth is not None:
            metrics += ["absolute_accuracy_change", "source_accuracy", "hardware_accuracy", "observed_slack"]
        means[variant] = {key: float(np.mean([r[key] for r in selected])) for key in metrics}
    return dict(status="completed_physical_capture_analysis", physical_certificate=True,
        backend="SpiNNaker-1 convolutional and recurrent dynamics with host readout and window aggregation",
        population=semantic["population"], confidence=semantic["confidence"],
        paper_cells=semantic["paper_cells"], alpha_per_term=semantic["alpha_per_term"],
        capture_profiles=capture_profiles(captures),
        windows_per_observation=4, rows=rows, five_seed_means=means,
        assumptions=["Representative independent recording/hardware-execution pairs for the declared population.",
                     "Fixed checkpoints, execution mapping, input preprocessing and four-window horizon.",
                     "Labels, when supplied, are used only for post-capture evaluation."],
        windows_with_late_spikes=sum(bool(d["late_spikes"]) for d in diagnostics),
        windows_with_provenance_messages=sum(bool(d["messages"]) for d in diagnostics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.captures, args.audit, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
