"""Transport certificates for finite-horizon digital SNNs."""

from .abstract import IntervalFamilyCertifier, SemanticsBox
from .certificates import CertificateEngine, SemanticsFamily
from .emulator import ExecutionTrace, ScalarInterpreter, VectorizedEmulator
from .models import DenseRecurrentSNN
from .reports import CertificateReport, RepairReport
from .semantics import ExecutionSemantics, NumericFormat

__all__ = [
    "CertificateEngine",
    "CertificateReport",
    "DenseRecurrentSNN",
    "ExecutionSemantics",
    "ExecutionTrace",
    "IntervalFamilyCertifier",
    "NumericFormat",
    "RepairReport",
    "ScalarInterpreter",
    "SemanticsFamily",
    "SemanticsBox",
    "VectorizedEmulator",
]

__version__ = "0.1.0"
