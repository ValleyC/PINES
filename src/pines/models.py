from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import array_hash, sha256_json


def _as_vector(value: float | np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim == 0:
        vector = np.full(size, float(vector), dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    return vector


@dataclass(frozen=True)
class DenseRecurrentSNN:
    input_weights: np.ndarray
    recurrent_weights: np.ndarray
    output_weights: np.ndarray
    bias: np.ndarray
    threshold: np.ndarray
    tau_mem: np.ndarray
    reset_value: np.ndarray
    name: str = "dense-recurrent-snn"

    def __post_init__(self) -> None:
        input_weights = np.asarray(self.input_weights, dtype=np.float64)
        recurrent_weights = np.asarray(self.recurrent_weights, dtype=np.float64)
        output_weights = np.asarray(self.output_weights, dtype=np.float64)
        if input_weights.ndim != 2:
            raise ValueError("input_weights must be [inputs, neurons]")
        neurons = input_weights.shape[1]
        if recurrent_weights.shape != (neurons, neurons):
            raise ValueError("recurrent_weights must be [neurons, neurons]")
        if output_weights.ndim != 2 or output_weights.shape[0] != neurons:
            raise ValueError("output_weights must be [neurons, classes]")
        object.__setattr__(self, "input_weights", input_weights)
        object.__setattr__(self, "recurrent_weights", recurrent_weights)
        object.__setattr__(self, "output_weights", output_weights)
        object.__setattr__(self, "bias", _as_vector(self.bias, neurons, "bias"))
        object.__setattr__(
            self, "threshold", _as_vector(self.threshold, neurons, "threshold")
        )
        object.__setattr__(self, "tau_mem", _as_vector(self.tau_mem, neurons, "tau_mem"))
        object.__setattr__(
            self, "reset_value", _as_vector(self.reset_value, neurons, "reset_value")
        )
        if np.any(self.threshold <= 0):
            raise ValueError("thresholds must be positive")
        if np.any(self.tau_mem <= 0):
            raise ValueError("tau_mem values must be positive")
        for array in self.arrays.values():
            array.setflags(write=False)

    @property
    def input_size(self) -> int:
        return self.input_weights.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.input_weights.shape[1]

    @property
    def output_size(self) -> int:
        return self.output_weights.shape[1]

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "input_weights": self.input_weights,
            "recurrent_weights": self.recurrent_weights,
            "output_weights": self.output_weights,
            "bias": self.bias,
            "threshold": self.threshold,
            "tau_mem": self.tau_mem,
            "reset_value": self.reset_value,
        }

    @property
    def model_hash(self) -> str:
        return sha256_json(
            {
                "type": "DenseRecurrentSNN/v1",
                "name": self.name,
                "arrays": {name: array_hash(value) for name, value in self.arrays.items()},
            }
        )

    def with_parameters(self, **updates: Any) -> "DenseRecurrentSNN":
        return replace(self, **updates)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"model artifact already exists: {destination}")
        with destination.open("xb") as handle:
            np.savez_compressed(
                handle,
                **self.arrays,
                metadata=np.asarray(
                    json.dumps({"type": "DenseRecurrentSNN/v1", "name": self.name})
                ),
            )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DenseRecurrentSNN":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("type") != "DenseRecurrentSNN/v1":
                raise ValueError("unsupported model artifact")
            return cls(
                input_weights=data["input_weights"],
                recurrent_weights=data["recurrent_weights"],
                output_weights=data["output_weights"],
                bias=data["bias"],
                threshold=data["threshold"],
                tau_mem=data["tau_mem"],
                reset_value=data["reset_value"],
                name=metadata.get("name", "dense-recurrent-snn"),
            )

