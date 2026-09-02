from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import config_description, write_json


@dataclass(frozen=True)
class CertificateReport:
    schema_version: str
    model_description: str
    data_description: str
    dataset_split: str
    checkpoint_file: str
    reference_semantics: str
    target_semantics: tuple[str, ...]
    static_family: str
    sample_count: int
    confidence_level: float
    certified_input_fraction: float
    semantic_disagreement_count: int
    semantic_bound: float
    conformance_disagreement_count: int | None
    conformance_bound: float | None
    total_bound: float
    decision_budget: float
    budget_verdict: str
    conditional_on_emulator: bool
    code_revision: str
    random_seed: str
    firmware_version: str | None = None
    bitstream_file: str | None = None
    assumptions: tuple[str, ...] = ()
    member_bounds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "CertificateReport/v1":
            raise ValueError("unsupported certificate report schema")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        for value in (
            self.confidence_level,
            self.certified_input_fraction,
            self.semantic_bound,
            self.total_bound,
            self.decision_budget,
        ):
            if not 0 <= value <= 1:
                raise ValueError("report probabilities and budgets must be in [0,1]")
        if self.budget_verdict not in {"accept", "reject"}:
            raise ValueError("budget_verdict must be accept or reject")
        if self.conditional_on_emulator != (self.conformance_bound is None):
            raise ValueError("conditional flag must match conformance evidence")

    @property
    def report_description(self) -> str:
        return config_description(asdict(self))

    def write(self, path: str | Path) -> Path:
        return write_json(path, asdict(self))


@dataclass(frozen=True)
class RepairReport:
    schema_version: str
    source_certificate: str
    permitted_parameter_changes: tuple[str, ...]
    calibration_split: str
    audit_split: str
    data_budget: int
    label_budget: int
    optimization_evaluations: int
    optimization_seconds: float
    original_model: str
    repaired_model: str
    pre_repair_objective: float
    post_repair_objective: float
    post_repair_certificate: str
    selected_changes: tuple[dict[str, Any], ...]
    code_revision: str

    def __post_init__(self) -> None:
        if self.schema_version != "RepairReport/v1":
            raise ValueError("unsupported repair report schema")
        if self.label_budget < 0 or self.data_budget <= 0:
            raise ValueError("invalid data or label budget")
        if self.calibration_split == self.audit_split:
            raise ValueError("calibration and audit artifacts must differ")

    @property
    def report_description(self) -> str:
        return config_description(asdict(self))

    def write(self, path: str | Path) -> Path:
        return write_json(path, asdict(self))
