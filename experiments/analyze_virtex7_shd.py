"""Analyze complete physical Virtex-7 SHD captures against the frozen audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.adapters.hardware import load_hardware_capture
from pines.statistics import clopper_pearson_upper


def analyze(captures, audit, labels=None):
    semantic = json.loads((audit / "semantic_audit.json").read_text())
    records, manifests, counts = {}, [], []
    with np.load(audit / "paired_predictions.npz") as paired:
        ids = paired["canary_sample_ids"].astype(str)
        for row in semantic["rows"]:
            folder = captures / str(row["seed"]) / row["variant"]
            if not (folder / "capture.npz").exists() or not (folder / "manifest.json").exists():
                counts.append(0)
                continue
            predictions, samples, manifest = load_hardware_capture(folder / "capture.npz", folder / "manifest.json")
            if manifest.backend != "virtex7" or manifest.seed != str(row["seed"]):
                raise ValueError("Capture backend and seed must match the declared condition")
            if len(set(samples)) != len(samples) or set(samples) - set(ids):
                raise ValueError("Primary samples must occur once and belong to the frozen canary")
            records[(row["seed"], row["variant"])] = dict(zip(samples, predictions))
            counts.append(len(samples))
            manifests.append(dict(variant=row["variant"], **asdict(manifest)))
        if any(count != len(ids) for count in counts):
            return dict(status="incomplete_capture", physical_certificate=False,
                observations=sum(counts), expected_observations=len(ids)*len(semantic["rows"]),
                samples_per_condition=counts)
        truth = None
        if labels is not None:
            with np.load(labels) as data:
                by_id = dict(zip(data["sample_ids"].astype(str), data["labels"]))
            truth = np.asarray([by_id[sample] for sample in ids])
        rows = []
        for s in semantic["rows"]:
            seed, variant = s["seed"], s["variant"]
            hardware = np.asarray([records[(seed, variant)][sample] for sample in ids])
            emulator = paired[f"seed{seed}_canary_{variant}_emulator"]
            source = paired[f"seed{seed}_canary_source"]
            k = int(np.count_nonzero(hardware != emulator))
            upper = clopper_pearson_upper(k, len(ids), semantic["alpha_per_term"])
            total = min(1., s["semantic_upper"]+upper)
            row = dict(seed=seed, variant=variant, canary_samples=len(ids),
                hardware_disagreements=k, hardware_disagreement_rate=k/len(ids),
                semantic_upper=s["semantic_upper"], conformance_upper=upper, total_upper=total,
                observed_source_hardware_disagreement=float((source != hardware).mean()),
                verdicts={str(b): "accept" if total <= b else "reject" for b in (.01,.02,.05)})
            if truth is not None:
                source_acc, hardware_acc = float((source == truth).mean()), float((hardware == truth).mean())
                change = source_acc-hardware_acc
                row.update(source_accuracy=source_acc, hardware_accuracy=hardware_acc,
                    accuracy_loss=change, absolute_accuracy_change=abs(change), observed_slack=total-abs(change))
            rows.append(row)
    means = {}
    for variant in dict.fromkeys(row["variant"] for row in rows):
        selected = [row for row in rows if row["variant"] == variant]
        metrics = ["hardware_disagreement_rate", "semantic_upper", "conformance_upper", "total_upper"]
        if truth is not None:
            metrics += ["source_accuracy", "hardware_accuracy", "absolute_accuracy_change", "observed_slack"]
        means[variant] = {key: float(np.mean([row[key] for row in selected])) for key in metrics}
    return dict(status="completed_physical_capture_analysis", physical_certificate=True,
        backend="Virtex-7 SHD floor-rounded integer executor", population=semantic["population"],
        confidence=semantic["confidence"], paper_cells=semantic["paper_cells"],
        alpha_per_term=semantic["alpha_per_term"], rows=rows, five_seed_means=means,
        hardware_runs=manifests,
        assumptions=["The operator confirms physical execution of the frozen parameters and inputs, supported by retained board traces.",
                     "Representative independent input/hardware-run pairs for the declared population.",
                     "Labels, when supplied, are used only for post-capture evaluation."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("results/virtex7_shd"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.captures, args.audit, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
