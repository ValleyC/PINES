from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from transportcert.certificates import CertificateEngine, SemanticsFamily
from transportcert.repair import CertificateDirectedRepair
from transportcert.semantics import ExecutionSemantics, ResetRule


def _source_report(model, inputs):
    reference = ExecutionSemantics()
    target = replace(reference, reset_rule=ResetRule.TO_VALUE)
    return CertificateEngine().build_report(
        model,
        inputs,
        reference,
        SemanticsFamily((target,)),
        delta=0.05,
        decision_budget=0.5,
        dataset_split="audit",
        checkpoint_hash=model.model_hash,
        seed_manifest=[0],
    )


def test_report_write_is_immutable(tmp_path, small_model, event_batch) -> None:
    report = _source_report(small_model, event_batch)
    path = tmp_path / "certificate.json"
    report.write(path)
    with pytest.raises(FileExistsError, match="immutable"):
        report.write(path)


def test_repair_enforces_disjoint_sample_ids(small_model, event_batch) -> None:
    report = _source_report(small_model, event_batch[16:])
    with pytest.raises(ValueError, match="overlap"):
        CertificateDirectedRepair().repair(
            small_model,
            event_batch[:16],
            event_batch[16:],
            [f"x-{i}" for i in range(16)],
            ["x-0"] + [f"y-{i}" for i in range(15)],
            ExecutionSemantics(),
            ExecutionSemantics(reset_rule=ResetRule.TO_VALUE),
            report,
        )


def test_repair_is_label_free_and_recertifies(small_model, event_batch) -> None:
    calibration = event_batch[:16]
    audit = event_batch[16:]
    source = _source_report(small_model, audit)
    outcome = CertificateDirectedRepair().repair(
        small_model,
        calibration,
        audit,
        [f"cal-{i}" for i in range(len(calibration))],
        [f"audit-{i}" for i in range(len(audit))],
        ExecutionSemantics(),
        ExecutionSemantics(reset_rule=ResetRule.TO_VALUE),
        source,
    )
    assert outcome.report.label_budget == 0
    assert outcome.report.post_repair_objective <= outcome.report.pre_repair_objective
    assert outcome.report.post_repair_certificate_hash == outcome.post_repair_certificate.report_hash

