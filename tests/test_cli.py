from __future__ import annotations

import json

import numpy as np

from transportcert.cli import main
from transportcert.semantics import ExecutionSemantics


def test_emulate_and_certify_cli(tmp_path, small_model, event_batch) -> None:
    model_path = tmp_path / "model.npz"
    inputs_path = tmp_path / "inputs.npy"
    reference_path = tmp_path / "reference.json"
    target_path = tmp_path / "target.json"
    emulation_path = tmp_path / "emulation.json"
    certificate_path = tmp_path / "certificate.json"
    small_model.save(model_path)
    np.save(inputs_path, event_batch)
    reference_path.write_text(
        json.dumps(ExecutionSemantics().to_dict()), encoding="utf-8"
    )
    target = ExecutionSemantics(synaptic_delay_steps=1)
    target_path.write_text(json.dumps(target.to_dict()), encoding="utf-8")
    assert main(
        [
            "emulate",
            "--model",
            str(model_path),
            "--inputs",
            str(inputs_path),
            "--semantics",
            str(reference_path),
            "--output",
            str(emulation_path),
        ]
    ) == 0
    assert main(
        [
            "certify",
            "--model",
            str(model_path),
            "--inputs",
            str(inputs_path),
            "--reference",
            str(reference_path),
            "--target",
            str(target_path),
            "--output",
            str(certificate_path),
        ]
    ) == 0
    emulation = json.loads(emulation_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    assert emulation["schema_version"] == "EmulationReport/v1"
    assert certificate["conditional_on_emulator"] is True

