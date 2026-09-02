from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .semantics import NumericFormat


LEAK_FRACTIONAL_BITS = 24


def fixed_integer(values: np.ndarray, numeric: NumericFormat) -> np.ndarray:
    """Quantize values and return their signed fixed-point integers."""

    if not numeric.is_fixed:
        raise ValueError("fixed_integer requires a fixed-point NumericFormat")
    assert numeric.fractional_bits is not None
    assert numeric.total_bits is not None
    quantized = np.asarray(numeric.quantize(values), dtype=np.float64)
    integers = np.rint(quantized * (1 << numeric.fractional_bits)).astype(
        np.int64
    )
    minimum = -(1 << (numeric.total_bits - 1))
    maximum = (1 << (numeric.total_bits - 1)) - 1
    if np.any(integers < minimum) or np.any(integers > maximum):
        raise AssertionError("quantized integer lies outside its declared format")
    if numeric.total_bits <= 8:
        return integers.astype(np.int8)
    if numeric.total_bits <= 16:
        return integers.astype(np.int16)
    if numeric.total_bits <= 32:
        return integers.astype(np.int32)
    return integers


def destination_major(matrix: np.ndarray) -> np.ndarray:
    """Flatten a canonical [source, destination] matrix destination first."""

    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("destination_major requires a two-dimensional matrix")
    return np.ascontiguousarray(values.T).reshape(-1)


def twos_complement_hex(values: np.ndarray, bits: int) -> list[str]:
    """Encode signed integers as fixed-width two's-complement hexadecimal."""

    if bits <= 0 or bits % 4:
        raise ValueError("hex output requires a positive multiple-of-four width")
    integers = np.asarray(values, dtype=np.int64).reshape(-1)
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if np.any(integers < minimum) or np.any(integers > maximum):
        raise ValueError(f"value does not fit signed {bits}-bit representation")
    mask = (1 << bits) - 1
    digits = bits // 4
    return [f"{int(value) & mask:0{digits}x}" for value in integers]


def write_mem(path: str | Path, values: np.ndarray, bits: int) -> Path:
    """Write one fixed-width hexadecimal word per line."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(twos_complement_hex(values, bits)) + "\n"
    with destination.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(payload)
    return destination


def pack_spike_words(frames: np.ndarray, channels: int) -> list[str]:
    """Encode [sample,time,channel] spikes as one channel-bit vector per line.

    Channel zero is bit zero and therefore the least significant bit of the
    rightmost hexadecimal digit.
    """

    spikes = np.asarray(frames, dtype=np.uint8)
    if spikes.ndim != 3 or spikes.shape[2] != channels:
        raise ValueError("frames must have shape [sample,time,channels]")
    if np.any((spikes != 0) & (spikes != 1)):
        raise ValueError("spike frames must be binary")
    digits = (channels + 3) // 4
    rows: list[str] = []
    for sample in spikes:
        for frame in sample:
            word = 0
            for channel in np.flatnonzero(frame):
                word |= 1 << int(channel)
            rows.append(f"{word:0{digits}x}")
    return rows


def write_spike_mem(
    path: str | Path, frames: np.ndarray, channels: int
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(pack_spike_words(frames, channels)) + "\n")
    return destination


def write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Write a compressed NumPy bundle."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return destination


def json_scalar(value: object) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def leak_reciprocal_q24(tau_mem: np.ndarray) -> np.ndarray:
    """Map positive time constants to unsigned floor-rounded Q0.24 leaks."""

    tau = np.asarray(tau_mem, dtype=np.float64)
    if np.any(~np.isfinite(tau)) or np.any(tau <= 1.0):
        raise ValueError("Q0.24 leak mapping requires finite tau_mem greater than one")
    scaled = np.floor((1.0 / tau) * (1 << LEAK_FRACTIONAL_BITS))
    if np.any(scaled <= 0) or np.any(scaled >= (1 << LEAK_FRACTIONAL_BITS)):
        raise ValueError("leak reciprocal lies outside unsigned Q0.24")
    return scaled.astype(np.uint32)


def run_floor_q8q16_integer(
    frames: np.ndarray,
    *,
    input_weights_q8: np.ndarray,
    recurrent_weights_q8: np.ndarray,
    output_weights_q8: np.ndarray,
    bias_q16: np.ndarray,
    threshold_q16: np.ndarray,
    leak_reciprocal_q24_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact integer oracle for the bundle's floor-rounded hardware target."""

    events = np.asarray(frames, dtype=np.int64)
    w_in = np.asarray(input_weights_q8, dtype=np.int64)
    w_rec = np.asarray(recurrent_weights_q8, dtype=np.int64)
    w_out = np.asarray(output_weights_q8, dtype=np.int64)
    bias = np.asarray(bias_q16, dtype=np.int64)
    threshold = np.asarray(threshold_q16, dtype=np.int64)
    leak = np.asarray(leak_reciprocal_q24_values, dtype=np.int64)
    if events.ndim != 3 or events.shape[2] != w_in.shape[0]:
        raise ValueError("event shape and input weight shape disagree")
    hidden = w_in.shape[1]
    if w_rec.shape != (hidden, hidden):
        raise ValueError("recurrent weight shape is invalid")
    if w_out.shape[0] != hidden:
        raise ValueError("output weight shape is invalid")
    if bias.shape != (hidden,) or threshold.shape != (hidden,):
        raise ValueError("hidden parameter shape is invalid")
    if leak.shape != (hidden,):
        raise ValueError("leak parameter shape is invalid")

    batch, horizon, _ = events.shape
    voltage = np.zeros((batch, hidden), dtype=np.int64)
    previous_spikes = np.zeros_like(voltage)
    logits = np.zeros((batch, w_out.shape[1]), dtype=np.int64)
    voltage_history: list[np.ndarray] = []
    spike_history: list[np.ndarray] = []
    logit_history: list[np.ndarray] = []
    for step in range(horizon):
        weight_sum = events[:, step] @ w_in + previous_spikes @ w_rec
        current = np.clip((weight_sum << 2) + bias, -32768, 32767)
        difference = current - voltage
        increment = np.floor_divide(
            difference * leak, 1 << LEAK_FRACTIONAL_BITS
        )
        integrated = np.clip(voltage + increment, -32768, 32767)
        spikes = (integrated >= threshold).astype(np.int64)
        voltage = np.clip(integrated - spikes * threshold, -32768, 32767)
        contribution = np.clip((spikes @ w_out) << 2, -32768, 32767)
        logits = np.clip(logits + contribution, -32768, 32767)
        previous_spikes = spikes
        voltage_history.append(voltage.astype(np.int16))
        spike_history.append(spikes.astype(np.uint8))
        logit_history.append(logits.astype(np.int16))
    return {
        "membrane_q16": np.stack(voltage_history, axis=1),
        "hidden_spikes": np.stack(spike_history, axis=1),
        "logits_over_time_q16": np.stack(logit_history, axis=1),
        "final_logits_q16": logits.astype(np.int16),
        "predictions": np.argmax(logits, axis=1).astype(np.int16),
    }
