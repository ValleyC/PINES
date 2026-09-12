"""Package frozen SHD checkpoints and label-free inputs for EBRAINS."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


def export_bundle(root: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((root / "results/shd_v1/repair_task_tuned_clean_v5_summary.json").read_text())
    repair_root = root / summary["protocol"]["method_artifact_roots"]["certificate_directed"].replace("\\", "/")
    source_root = root / "artifacts/shd_v1_final"
    with np.load(source_root / "seed_1701/split_indices.npz") as split:
        development = split["repair_calibration"][:8]
        audit = split["certificate_audit"]
    with np.load(root / "data/processed/shd_v1/train.npz") as train:
        for name, indices, split_name in (("development", development, "repair-calibration"),
                                          ("training_audit", audit, "software-training-audit")):
            metadata = dict(dataset="SHD", split=split_name, labels_included=False,
                            time_bins=50, input_channels=700, bit_order="little")
            np.savez_compressed(output / f"{name}_inputs.npz", packed_spikes=train["packed"][indices],
                                sample_ids=train["sample_ids"][indices], metadata=json.dumps(metadata))
    with np.load(root / "data/processed/shd_v1/test.npz") as test:
        indices = np.sort(np.random.default_rng(20260912).choice(len(test["sample_ids"]), size=861, replace=False))
        np.savez_compressed(output / "canary_inputs.npz", packed_spikes=test["packed"][indices],
            sample_ids=test["sample_ids"][indices], metadata=json.dumps(dict(dataset="SHD",
                split="held-out-test-canary", labels_included=False, time_bins=50,
                input_channels=700, bit_order="little", selection_seed=20260912)))
        # Both terms of a physical certificate concern the same target-input
        # population. Do not combine a training-pool semantic audit with a
        # held-out-test hardware canary, since SHD has a speaker split.
        remaining = np.setdiff1d(np.arange(len(test["sample_ids"])), indices)
        semantic_indices = np.sort(np.random.default_rng(20260913).choice(remaining, size=861, replace=False))
        np.savez_compressed(output / "audit_inputs.npz", packed_spikes=test["packed"][semantic_indices],
            sample_ids=test["sample_ids"][semantic_indices], metadata=json.dumps(dict(dataset="SHD",
                split="held-out-test-semantic-audit", labels_included=False, time_bins=50,
                input_channels=700, bit_order="little", selection_seed=20260913)))
    for seed in (1701, 2718, 3141, 5772, 8119):
        dest = output / "models" / str(seed)
        dest.mkdir(parents=True)
        shutil.copy2(source_root / f"seed_{seed}/model.npz", dest / "original.npz")
        shutil.copy2(repair_root / f"seed_{seed}/reset_to_value/certificate_directed/repaired_model.npz",
                     dest / "reset_repaired.npz")
    for script in ("spinnaker1_probe.py", "analyze_spinnaker1_probe.py", "run_spinnaker1_shd.py",
                   "analyze_spinnaker1_shd.py", "run_spinnaker1_matrix.py"):
        dest = output / "experiments" / script
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(root / "experiments" / script, dest)
    shutil.copytree(root / "src/pines", output / "src/pines", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    note = dict(status="frozen_inputs_not_hardware_results", seeds=[1701,2718,3141,5772,8119],
                source_checkpoint_root="artifacts/shd_v1_final",
                repair_checkpoint_root=str(repair_root.relative_to(root)).replace("\\", "/"),
                repair_condition="reset_to_value", mapping="match_source_euler",
                readout="host linear readout from hardware hidden spikes",
                development_samples=8, audit_samples=861, canary_samples=861,
                labels_included=False,
                physical_population="SHD held-out test pool, disjoint semantic-audit and hardware-canary samples",
                purpose="Development inputs for debugging. Audit and canary inputs only after mapping freeze.")
    (output / "bundle.json").write_text(json.dumps(note, indent=2) + "\n")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob("*"):
            if path.is_file():
                z.write(path, str(Path(output.name) / path.relative_to(output)))
    print(json.dumps(note, indent=2))
    print(archive)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export_bundle(Path(__file__).resolve().parents[1], parser.parse_args().output.resolve())
