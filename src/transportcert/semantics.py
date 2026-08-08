from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import sha256_json


class IntegrationRule(str, Enum):
    FORWARD_EULER = "forward_euler"
    EXPONENTIAL_EULER = "exponential_euler"


class ThresholdTiming(str, Enum):
    PRE_INTEGRATION = "pre_integration"
    POST_INTEGRATION = "post_integration"


class UpdateOrdering(str, Enum):
    THRESHOLD_RESET_INTEGRATE = "threshold>reset>integrate"
    INTEGRATE_THRESHOLD_RESET = "integrate>threshold>reset"


class ResetRule(str, Enum):
    SUBTRACTIVE = "subtractive"
    TO_VALUE = "to_value"


class RoundingMode(str, Enum):
    NEAREST_EVEN = "nearest_even"
    FLOOR = "floor"
    TRUNCATE = "truncate"
    STOCHASTIC = "stochastic"


class OverflowMode(str, Enum):
    SATURATE = "saturate"
    WRAP = "wrap"


@dataclass(frozen=True)
class NumericFormat:
    kind: str = "float64"
    total_bits: int | None = None
    fractional_bits: int | None = None
    rounding: RoundingMode = RoundingMode.NEAREST_EVEN
    overflow: OverflowMode = OverflowMode.SATURATE

    def __post_init__(self) -> None:
        if self.kind not in {"float32", "float64", "fixed"}:
            raise ValueError("numeric kind must be 'float32', 'float64', or 'fixed'")
        if self.kind == "fixed":
            if self.total_bits is None or self.fractional_bits is None:
                raise ValueError("fixed format requires total_bits and fractional_bits")
            if self.total_bits < 2:
                raise ValueError("total_bits must include sign and magnitude")
            if not 0 <= self.fractional_bits < self.total_bits:
                raise ValueError("fractional_bits must be in [0, total_bits)")

    @property
    def is_fixed(self) -> bool:
        return self.kind == "fixed"

    def quantize(
        self, value: float | np.ndarray, rng: np.random.Generator | None = None
    ) -> float | np.ndarray:
        dtype = np.float32 if self.kind == "float32" else np.float64
        values = np.asarray(value, dtype=dtype)
        if not self.is_fixed:
            result = values
        else:
            assert self.fractional_bits is not None
            assert self.total_bits is not None
            scale = float(1 << self.fractional_bits)
            scaled = values * scale
            if self.rounding is RoundingMode.NEAREST_EVEN:
                integers = np.rint(scaled)
            elif self.rounding is RoundingMode.FLOOR:
                integers = np.floor(scaled)
            elif self.rounding is RoundingMode.TRUNCATE:
                integers = np.trunc(scaled)
            else:
                if rng is None:
                    raise ValueError("stochastic rounding requires an RNG")
                lower = np.floor(scaled)
                integers = lower + (rng.random(scaled.shape) < (scaled - lower))
            minimum = -(1 << (self.total_bits - 1))
            maximum = (1 << (self.total_bits - 1)) - 1
            if self.overflow is OverflowMode.SATURATE:
                integers = np.clip(integers, minimum, maximum)
            else:
                modulus = 1 << self.total_bits
                integers = ((integers - minimum) % modulus) + minimum
            result = integers / scale
        if np.isscalar(value):
            return float(result)
        return result


@dataclass(frozen=True)
class Randomness:
    deterministic: bool = True
    seed: int = 0


@dataclass(frozen=True)
class ExecutionSemantics:
    version: str = "1.0"
    timestep: float = 1.0
    integration_rule: IntegrationRule = IntegrationRule.FORWARD_EULER
    threshold_timing: ThresholdTiming = ThresholdTiming.POST_INTEGRATION
    update_ordering: UpdateOrdering = UpdateOrdering.INTEGRATE_THRESHOLD_RESET
    reset_rule: ResetRule = ResetRule.SUBTRACTIVE
    state_format: NumericFormat = field(default_factory=NumericFormat)
    weight_format: NumericFormat = field(default_factory=NumericFormat)
    synaptic_delay_steps: int = 0
    output_delay_steps: int = 0
    randomness: Randomness = field(default_factory=Randomness)

    def __post_init__(self) -> None:
        if self.version != "1.0":
            raise ValueError(f"unsupported ExecutionSemantics version: {self.version}")
        if not math.isfinite(self.timestep) or self.timestep <= 0:
            raise ValueError("timestep must be finite and positive")
        if self.synaptic_delay_steps < 0 or self.output_delay_steps < 0:
            raise ValueError("delivery delays must be non-negative")
        consistent = {
            ThresholdTiming.PRE_INTEGRATION: UpdateOrdering.THRESHOLD_RESET_INTEGRATE,
            ThresholdTiming.POST_INTEGRATION: UpdateOrdering.INTEGRATE_THRESHOLD_RESET,
        }
        if consistent[self.threshold_timing] is not self.update_ordering:
            raise ValueError("threshold_timing and update_ordering describe different orders")
        formats = (self.state_format, self.weight_format)
        if any(f.rounding is RoundingMode.STOCHASTIC for f in formats):
            if self.randomness.deterministic:
                raise ValueError("stochastic rounding requires deterministic=false")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))

    @property
    def semantics_hash(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionSemantics":
        state = data.get("state_format", {})
        weight = data.get("weight_format", {})
        random = data.get("randomness", {})

        def numeric(raw: dict[str, Any]) -> NumericFormat:
            return NumericFormat(
                kind=raw.get("kind", "float64"),
                total_bits=raw.get("total_bits"),
                fractional_bits=raw.get("fractional_bits"),
                rounding=RoundingMode(raw.get("rounding", "nearest_even")),
                overflow=OverflowMode(raw.get("overflow", "saturate")),
            )

        return cls(
            version=data.get("version", "1.0"),
            timestep=float(data.get("timestep", 1.0)),
            integration_rule=IntegrationRule(
                data.get("integration_rule", "forward_euler")
            ),
            threshold_timing=ThresholdTiming(
                data.get("threshold_timing", "post_integration")
            ),
            update_ordering=UpdateOrdering(
                data.get("update_ordering", "integrate>threshold>reset")
            ),
            reset_rule=ResetRule(data.get("reset_rule", "subtractive")),
            state_format=numeric(state),
            weight_format=numeric(weight),
            synaptic_delay_steps=int(data.get("synaptic_delay_steps", 0)),
            output_delay_steps=int(data.get("output_delay_steps", 0)),
            randomness=Randomness(
                deterministic=bool(random.get("deterministic", True)),
                seed=int(random.get("seed", 0)),
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExecutionSemantics":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
