"""Combine completed SHD canary captures with the frozen semantic audit.

An incomplete campaign produces progress counts only, never a final certificate.
The optional labeled evaluation is read only after the complete capture is found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pines.statistics import clopper_pearson_upper


def collect(captures):
    observations = {}
    diagnostics = []
    for root in captures:
        for path in sorted(root.glob("input_*/summary.json")):
            sample = json.loads(path.read_text())
            for row in sample["rows"]:
                key = (row["seed"], row["variant"], sample["sample_id"])
                if key in observations:
                    raise ValueError(f"Duplicate primary observation: {key}")
                observations[key] = row
            diagnostic_path = path.with_name("diagnostics.json")
            diagnostics.append(dict(sample_id=sample["sample_id"],
                                    **json.loads(diagnostic_path.read_text())))
    return observations, diagnostics


def analyze(captures, audit, labels=None):
    semantic = json.loads((audit / "semantic_audit.json").read_text())
    observations, diagnostics = collect(captures)
    with np.load(audit / "paired_predictions.npz") as paired:
        ids = paired["canary_sample_ids"].astype(str)
        counts = [sum((r["seed"], r["variant"], sample) in observations for sample in ids)
                  for r in semantic["rows"]]
        expected = len(ids) * len(semantic["rows"])
        if sum(counts) != len(observations):
            raise ValueError("Captures include inputs or conditions outside the frozen canary")
        if any(count != len(ids) for count in counts):
            return dict(status="incomplete_capture", physical_certificate=False,
                        observations=len(observations), expected_observations=expected,
                        samples_per_condition=counts)
        truth = None
        if labels is not None:
            with np.load(labels) as data:
                by_id = dict(zip(data["sample_ids"].astype(str), data["labels"]))
            truth = np.asarray([by_id[sample] for sample in ids])
        rows = []
        for semantic_row in semantic["rows"]:
            seed, variant = semantic_row["seed"], semantic_row["variant"]
            records = [observations[(seed, variant, sample)] for sample in ids]
            hardware = np.asarray([r["prediction"] for r in records])
            emulator = paired[f"seed{seed}_canary_{variant}_emulator"]
            if not np.array_equal(emulator, [r["emulator_prediction"] for r in records]):
                raise ValueError("Capture emulator predictions differ from the frozen audit executor")
            source = paired[f"seed{seed}_canary_source"]
            k = int((hardware != emulator).sum())
            upper = clopper_pearson_upper(k, len(ids), semantic["alpha_per_term"])
            total = min(1.0, semantic_row["semantic_upper"] + upper)
            row = dict(seed=seed, variant=variant, canary_samples=len(ids),
                       hardware_disagreements=k, hardware_disagreement_rate=k/len(ids),
                       semantic_upper=semantic_row["semantic_upper"], conformance_upper=upper,
                       total_upper=total,
                       verdicts={str(budget): "accept" if total <= budget else "reject"
                                 for budget in (0.01, 0.02, 0.05)},
                       observed_source_hardware_disagreement=float((source != hardware).mean()))
            if truth is not None:
                source_acc = float((source == truth).mean())
                hardware_acc = float((hardware == truth).mean())
                change = source_acc - hardware_acc
                row.update(source_accuracy=source_acc, hardware_accuracy=hardware_acc,
                           accuracy_loss=change, absolute_accuracy_change=abs(change),
                           observed_slack=total-abs(change))
            rows.append(row)
    means = {}
    for variant in ("original", "reset_repaired"):
        selected = [r for r in rows if r["variant"] == variant]
        metrics = ["hardware_disagreement_rate", "semantic_upper", "conformance_upper", "total_upper"]
        if truth is not None:
            metrics += ["absolute_accuracy_change", "source_accuracy", "hardware_accuracy", "observed_slack"]
        means[variant] = {key: float(np.mean([r[key] for r in selected])) for key in metrics}
    return dict(status="completed_physical_capture_analysis", physical_certificate=True,
                backend="SpiNNaker-1 hidden recurrence with host linear readout",
                population=semantic["population"], confidence=semantic["confidence"],
                paper_cells=semantic["paper_cells"], alpha_per_term=semantic["alpha_per_term"],
                assumptions=["Representative independent input/allocation pairs for the declared population.",
                             "Fixed model, mapping, finite horizon and input preprocessing.",
                             "Labels, when supplied, are used only for post-capture evaluation."],
                rows=rows, five_seed_means=means,
                allocations_with_late_spikes=sum(bool(d["late_spikes"]) for d in diagnostics),
                allocations_with_provenance_messages=sum(bool(d["messages"]) for d in diagnostics),
                log_review="Retain and review the execution logs, which can contain additional shutdown warnings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.captures, args.audit, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
