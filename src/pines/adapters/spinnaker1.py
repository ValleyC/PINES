"""Explicit mapping to sPyNNaker's IF_curr_delta neuron.

The floating-point executor below is an ideal mapped emulator, not a
bit-accurate simulation of SpiNNaker. Fixed-point and packet effects belong
in the measured emulator-device conformance term.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import DenseRecurrentSNN


@dataclass(frozen=True)
class SpiNNaker1Mapping:
    timestep_ms: float = 1.0
    integration: str = "native_exponential"

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestep_ms) or self.timestep_ms <= 0:
            raise ValueError("timestep_ms must be positive and finite")
        if self.integration not in {"native_exponential", "match_source_euler"}:
            raise ValueError("unknown integration mapping")

    def neuron_parameters(self, model: DenseRecurrentSNN) -> dict:
        tau = np.array(model.tau_mem, copy=True)
        if self.integration == "match_source_euler":
            ratio = self.timestep_ms / tau
            if np.any(ratio >= 1):
                raise ValueError("Euler leak matching requires 0 < timestep/tau < 1")
            tau = -self.timestep_ms / np.log1p(-ratio)
        # R=tau/cm=1. Delta input is divided by timestep on the device.
        return {
            "tau_m": tau,
            "cm": tau.copy(),
            "v_rest": np.zeros(model.hidden_size),
            "v_reset": np.array(model.reset_value, copy=True),
            "v_thresh": np.array(model.threshold, copy=True),
            "tau_refrac": np.zeros(model.hidden_size),
            "i_offset": np.array(model.bias, copy=True),
        }

    def projection_weights(self, model: DenseRecurrentSNN) -> tuple[np.ndarray, np.ndarray]:
        return (model.input_weights * self.timestep_ms,
                model.recurrent_weights * self.timestep_ms)

    def contract(self) -> dict:
        return {
            "version": "SpiNNaker1Mapping/v1",
            "neuron": "IF_curr_delta",
            "integration_mapping": self.integration,
            "timestep_ms": self.timestep_ms,
            "reset": "to_value",
            "threshold": "post-integration >=",
            "refractory_ms": 0.0,
            "projection_delay_steps": 1,
            "weight_orientation": "source,destination",
            "signed_weights": "positive excitatory, absolute negative inhibitory",
            "voltage_recording": "pre-update",
            "emulator_precision": "float64 ideal mapped dynamics, not bit-accurate",
            "readout": "host sum of recorded hidden spikes times original output weights",
        }

    def emulate(self, model: DenseRecurrentSNN, inputs: np.ndarray) -> dict[str, np.ndarray]:
        events = np.asarray(inputs, dtype=np.float64)
        if events.ndim != 3 or events.shape[2] != model.input_size:
            raise ValueError("inputs must have shape [sample,time,input]")
        if np.any((events != 0) & (events != 1)):
            raise ValueError("SpikeSourceArray mapping requires binary events")
        params = self.neuron_parameters(model)
        alpha = np.exp(-self.timestep_ms / params["tau_m"])
        voltage = np.zeros((len(events), model.hidden_size))
        previous = np.zeros_like(voltage)
        membranes, spikes = [], []
        for t in range(events.shape[1]):
            current = events[:, t] @ model.input_weights + previous @ model.recurrent_weights + model.bias
            voltage = current - alpha * (current - voltage)
            previous = (voltage >= model.threshold).astype(np.float64)
            voltage = np.where(previous != 0, model.reset_value, voltage)
            membranes.append(voltage.copy())
            spikes.append(previous.copy())
        spike_array = np.stack(spikes, axis=1)
        logits = spike_array.sum(axis=1) @ model.output_weights
        return {"membrane": np.stack(membranes, axis=1), "spikes": spike_array,
                "logits": logits, "predictions": logits.argmax(axis=1)}


def signed_connections(weights: np.ndarray, delay_ms: float) -> dict[str, list[tuple]]:
    """PyNN FromListConnector rows, preserving source/destination orientation."""
    matrix = np.asarray(weights)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("weights must be a finite matrix")
    result = {}
    for name, mask in (("excitatory", matrix > 0), ("inhibitory", matrix < 0)):
        src, dst = np.nonzero(mask)
        result[name] = [(int(i), int(j), float(abs(matrix[i, j])), float(delay_ms))
                        for i, j in zip(src, dst)]
    return result
