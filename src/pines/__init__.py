"""Prediction-preservation certificates for finite-horizon digital SNNs."""

from .abstract import IntervalFamilyCertifier, SemanticsBox
from .certificates import CertificateEngine, SemanticsFamily
from .emulator import ExecutionTrace, ScalarInterpreter, VectorizedEmulator
from .exact_boundary import ExactThresholdBoundaryOracle
from .models import DenseRecurrentSNN
from .parameter_batch import TorchParameterBatchEmulator
from .reports import CertificateReport, RepairReport
from .semantics import ExecutionSemantics, NumericFormat

__all__ = [
    "CertificateEngine",
    "CertificateReport",
    "DenseRecurrentSNN",
    "ExecutionSemantics",
    "ExecutionTrace",
    "ExactThresholdBoundaryOracle",
    "IntervalFamilyCertifier",
    "NumericFormat",
    "RepairReport",
    "ScalarInterpreter",
    "SemanticsFamily",
    "SemanticsBox",
    "TorchParameterBatchEmulator",
    "VectorizedEmulator",
]

__version__ = "0.1.0"
