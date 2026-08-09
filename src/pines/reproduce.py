from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import array_hash, write_json_immutable
from .certificates import CertificateEngine, SemanticsFamily
from .emulator import ScalarInterpreter, VectorizedEmulator
from .models import DenseRecurrentSNN
from .semantics import ExecutionSemantics, ResetRule


def toy_model(seed: int = 7) -> DenseRecurrentSNN:
    rng = np.random.default_rng(seed)
    return DenseRecurrentSNN(
        input_weights=rng.normal(0.7, 0.25, size=(3, 5)),
        recurrent_weights=rng.normal(0.0, 0.18, size=(5, 5)),
        output_weights=rng.normal(0.0, 1.0, size=(5, 3)),
        bias=np.zeros(5),
        threshold=np.full(5, 0.75),
        tau_mem=np.full(5, 2.0),
        reset_value=np.zeros(5),
        name="pines-toy-srnn",
    )


def run_toy_study(output_dir: str | Path, samples: int = 256, seed: int = 11) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    inputs = (rng.random((samples, 20, 3)) < 0.18).astype(np.float64)
    model = toy_model()
    reference = ExecutionSemantics()
    reset_target = ExecutionSemantics(reset_rule=ResetRule.TO_VALUE)
    delayed_target = ExecutionSemantics(synaptic_delay_steps=1)
    scalar = ScalarInterpreter().run(model, inputs[:8], reference)
    vector = VectorizedEmulator().run(model, inputs[:8], reference)
    if not (
        np.array_equal(scalar.spikes, vector.spikes)
        and np.allclose(scalar.membrane, vector.membrane)
    ):
        raise AssertionError("scalar/vector differential check failed")
    family = SemanticsFamily((reset_target, delayed_target), "toy-high-risk-family")
    report = CertificateEngine().build_report(
        model,
        inputs,
        reference,
        family,
        delta=0.05,
        decision_budget=0.05,
        dataset_split="synthetic-audit",
        checkpoint_hash=model.model_hash,
        seed_manifest={"model": 7, "inputs": seed},
        repository_root=Path(__file__).resolve().parents[2],
    )
    report_path = output / "certificate.json"
    report.write(report_path)
    write_json_immutable(
        output / "manifest.json",
        {
            "schema_version": "ToyStudy/v1",
            "model_hash": model.model_hash,
            "data_hash": array_hash(inputs),
            "sample_count": samples,
            "seed": seed,
            "certificate_hash": report.report_hash,
            "note": "synthetic software smoke study; not manuscript evidence",
        },
    )
    return report_path


def reproduce_from_config(config_path: str | Path, output_dir: str | Path) -> Path:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("study") != "toy":
        raise ValueError("this release supports the 'toy' smoke study only")
    return run_toy_study(
        output_dir,
        samples=int(config.get("samples", 256)),
        seed=int(config.get("seed", 11)),
    )

