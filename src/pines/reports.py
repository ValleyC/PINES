from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_json, write_json_immutable


@dataclass(frozen=True)
class CertificateReport:
    schema_version: str
    model_hash: str
    data_hash: str
    dataset_split: str
    checkpoint_hash: str
    reference_semantics_hash: str
    target_semantics_hashes: tuple[str, ...]
    static_family_hash: str
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
    random_seed_hash: str
    firmware_hash: str | None = None
    bitstream_hash: str | None = None
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
    def report_hash(self) -> str:
        return sha256_json(asdict(self))

    def write(self, path: str | Path) -> Path:
        return write_json_immutable(path, asdict(self))


@dataclass(frozen=True)
class RepairReport:
    schema_version: str
    source_certificate_hash: str
    permitted_parameter_changes: tuple[str, ...]
    calibration_data_hash: str
    audit_data_hash: str
    data_budget: int
    label_budget: int
    optimization_evaluations: int
    optimization_seconds: float
    original_model_hash: str
    repaired_model_hash: str
    pre_repair_objective: float
    post_repair_objective: float
    post_repair_certificate_hash: str
    selected_changes: tuple[dict[str, Any], ...]
    code_revision: str

    def __post_init__(self) -> None:
        if self.schema_version != "RepairReport/v1":
            raise ValueError("unsupported repair report schema")
        if self.label_budget < 0 or self.data_budget <= 0:
            raise ValueError("invalid data or label budget")
        if self.calibration_data_hash == self.audit_data_hash:
            raise ValueError("calibration and audit artifacts must differ")

    @property
    def report_hash(self) -> str:
        return sha256_json(asdict(self))

    def write(self, path: str | Path) -> Path:
        return write_json_immutable(path, asdict(self))
