"""Frozen DVS FPGA arithmetic, independent of SpiNNaker's native neuron model.

MACs use exactly representable integer-valued FP32 operations (TF32 disabled,
absolute sums < 2**24). All neuron, leak, saturation and logit updates are int64.
The independent NumPy implementation is used for differential trace checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .hardware_bundle import fixed_integer, leak_reciprocal_q24
from .semantics import NumericFormat, OverflowMode, RoundingMode

WEIGHTS = ("conv1", "conv2", "hidden_input", "recurrent", "readout")
BIASES = ("conv1", "conv2", "hidden_input")
TRACE_FIELDS = ("conv1_membrane_q16", "conv1_spikes", "conv2_membrane_q16",
                "conv2_spikes", "hidden_membrane_q16", "hidden_spikes",
                "logits_over_time_q16")


def load_model(path: str | Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as data:
        return ({key: data[key] for key in data.files if key != "metadata"},
                json.loads(str(data["metadata"])))


def load_parameters(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != "metadata"}


def quantize_model(state: dict, meta: dict) -> dict[str, np.ndarray]:
    qw = NumericFormat("fixed", 8, 6, RoundingMode.FLOOR, OverflowMode.SATURATE)
    qs = NumericFormat("fixed", 16, 8, RoundingMode.FLOOR, OverflowMode.SATURATE)
    parameters = {name + "_weights_q8": fixed_integer(state[name + ".weight"], qw)
                  for name in WEIGHTS}
    parameters.update({name + "_bias_q16": fixed_integer(state[name + ".bias"], qs)
                       for name in BIASES})
    parameters["threshold_q16"] = fixed_integer(np.asarray([meta["threshold"]]), qs)
    parameters["leak_reciprocal_q24"] = leak_reciprocal_q24(np.asarray([meta["tau_mem"]]))
    return parameters


def mapping_contract() -> dict:
    return dict(
        version="DVSFPGA-Q8Q16-Q024/v1", timestep=1.0,
        integration="v + floor((current-v)*leak_q24 / 2**24)",
        leak="floor((1/tau_mem)*2**24), unsigned 32-bit storage",
        threshold="post-integration >=", reset="subtractive",
        weight_format="signed 8 bits, 6 fractional bits, floor, saturate",
        state_format="signed 16 bits, 8 fractional bits, floor, saturate",
        weight_to_state_left_shift=2,
        current="saturate16(4*sum(spike*weight_q8) + bias_q16)",
        accumulator="signed 32-bit MAC; signed 64-bit leak product; no intermediate MAC saturation",
        layer_order="input[t] -> conv1[t] -> conv2[t] -> hidden[t] -> readout[t]",
        recurrence="hidden[t-1], initially zero", synaptic_delay_steps=0,
        output_delay_steps=0, reset_between_windows=True,
        convolution="cross-correlation, stride 2, padding 0, no kernel rotation",
        memory_order="OIHW convolution; destination,source dense; CHW flattened states",
        readout="saturate16(logits + saturate16(4*hidden_spikes @ W_out.T)) every timestep",
        aggregation="host float64 mean of four softmax(window_logits_q16/256/0.5)",
        tie_rule="lowest class index", labels_included=False,
        note="Explicit Q0.24 FPGA leak mapping. Not SpiNNaker and not the original unquantized 1/tau software executor.",
    )


def unpack_inputs(packed: np.ndarray) -> np.ndarray:
    """[sample,window,time,256 bytes] -> [sample,window,time,2,32,32]."""
    if packed.ndim != 4 or packed.shape[-1] != 256:
        raise ValueError("DVS input must be [sample,window,time,256 packed bytes]")
    return np.unpackbits(packed, axis=-1, bitorder="little").reshape(*packed.shape[:-1], 2, 32, 32)


def aggregate_q16(window_logits: np.ndarray, temperature: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Host postprocessing. Four windows are ONE statistical observation."""
    z = np.asarray(window_logits, dtype=np.float64) / (256.0 * temperature)
    z -= z.max(axis=-1, keepdims=True)
    probabilities = np.exp(z)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    scores = probabilities.mean(axis=-2)
    return scores.argmax(axis=-1).astype(np.int16), scores


def numpy_step(voltage, current, threshold, leak):
    integrated = np.clip(voltage + np.floor_divide((current - voltage) * int(leak), 1 << 24), -32768, 32767)
    spikes = (integrated >= int(threshold)).astype(np.int64)
    return np.clip(integrated - spikes * int(threshold), -32768, 32767), spikes


def _conv_numpy(events, weights):
    kh, kw = weights.shape[-2:]
    patches = np.lib.stride_tricks.sliding_window_view(events, (kh, kw), axis=(-2, -1))
    patches = patches[:, :, ::2, ::2]
    return np.einsum("biyxkl,oikl->boyx", patches, weights, optimize=True)


def run_numpy(events: np.ndarray, parameters: dict) -> dict[str, np.ndarray]:
    """Independent int64 oracle. Intended for development traces, not fast batches."""
    x = np.asarray(events, dtype=np.int64)
    p = {key: np.asarray(value, dtype=np.int64) for key, value in parameters.items()}
    w1, w2 = p["conv1_weights_q8"], p["conv2_weights_q8"]
    shape1 = (x.shape[0], w1.shape[0], (x.shape[-2]-w1.shape[-2])//2+1, (x.shape[-1]-w1.shape[-1])//2+1)
    shape2 = (x.shape[0], w2.shape[0], (shape1[-2]-w2.shape[-2])//2+1, (shape1[-1]-w2.shape[-1])//2+1)
    v1, v2 = np.zeros(shape1, dtype=np.int64), np.zeros(shape2, dtype=np.int64)
    vh = np.zeros((x.shape[0], p["recurrent_weights_q8"].shape[0]), dtype=np.int64)
    previous = np.zeros_like(vh)
    logits = np.zeros((x.shape[0], p["readout_weights_q8"].shape[0]), dtype=np.int64)
    traces = {key: [] for key in TRACE_FIELDS}
    sat = lambda value: np.clip(value, -32768, 32767)
    step = lambda voltage, current: numpy_step(voltage, current, p["threshold_q16"].item(), p["leak_reciprocal_q24"].item())
    for t in range(x.shape[1]):
        v1, q1 = step(v1, sat(4*_conv_numpy(x[:, t], w1) + p["conv1_bias_q16"][None, :, None, None]))
        v2, q2 = step(v2, sat(4*_conv_numpy(q1, w2) + p["conv2_bias_q16"][None, :, None, None]))
        mac = q2.reshape(len(x), -1) @ p["hidden_input_weights_q8"].T + previous @ p["recurrent_weights_q8"].T
        vh, qh = step(vh, sat(4*mac + p["hidden_input_bias_q16"]))
        logits = sat(logits + sat(4*(qh @ p["readout_weights_q8"].T)))
        previous = qh
        for key, value in zip(TRACE_FIELDS, (v1, q1, v2, q2, vh, qh, logits)):
            traces[key].append(value.astype(np.uint8 if key.endswith("spikes") else np.int16))
    return {**{key: np.stack(value, axis=1) for key, value in traces.items()},
            "window_logits_q16": logits.astype(np.int16)}


def run_torch(events: np.ndarray, parameters: dict, *, device="cpu", record=False) -> dict[str, np.ndarray]:
    """Bit-exact integer-valued MACs and integer state evolution for DVS windows."""
    import torch
    import torch.nn.functional as F

    x = np.asarray(events)
    if x.ndim != 5 or np.any((x != 0) & (x != 1)):
        raise ValueError("events must be binary [window,time,channel,height,width]")
    # Each partial sum is bounded by the sum of absolute weights. Integer-valued
    # FP32 MACs then remain exact regardless of reduction order, without TF32.
    bounds = {}
    for name in WEIGHTS:
        w = np.asarray(parameters[name + "_weights_q8"], dtype=np.int64)
        bounds[name] = int(np.abs(w).reshape(w.shape[0], -1).sum(axis=1).max())
    if max(*bounds.values(), bounds["hidden_input"] + bounds["recurrent"]) >= 1 << 24:
        raise ValueError("MAC exceeds exact integer-valued FP32 range")
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False, benchmark=False, deterministic=True), torch.autocast(device_type=torch.device(device).type, enabled=False):
            spikes = torch.as_tensor(x, dtype=torch.float32, device=device)
            weights = {name: torch.as_tensor(parameters[name+"_weights_q8"], dtype=torch.float32, device=device) for name in WEIGHTS}
            biases = {name: torch.as_tensor(parameters[name+"_bias_q16"], dtype=torch.int64, device=device) for name in BIASES}
            threshold = int(parameters["threshold_q16"].item())
            leak = int(parameters["leak_reciprocal_q24"].item())
            def conv(value, weight):
                kh, kw = weight.shape[-2:]
                oh, ow = (value.shape[-2]-kh)//2+1, (value.shape[-1]-kw)//2+1
                # Explicit patches avoid Winograd/FFT convolution transforms.
                patches = F.unfold(value, (kh, kw), stride=2)
                return torch.matmul(weight.flatten(1), patches).reshape(len(value), weight.shape[0], oh, ow)

            shape1 = conv(spikes[:, 0], weights["conv1"]).shape
            shape2 = conv(torch.zeros(shape1, device=device), weights["conv2"]).shape
            v1 = torch.zeros(shape1, dtype=torch.int64, device=device)
            v2 = torch.zeros(shape2, dtype=torch.int64, device=device)
            vh = torch.zeros((len(x), weights["recurrent"].shape[0]), dtype=torch.int64, device=device)
            previous = torch.zeros_like(vh, dtype=torch.float32)
            logits = torch.zeros((len(x), weights["readout"].shape[0]), dtype=torch.int64, device=device)
            traces = {key: [] for key in TRACE_FIELDS}
            sat = lambda value: value.clamp(-32768, 32767)

            def step(voltage, current):
                integrated = sat(voltage + torch.div((current-voltage)*leak, 1 << 24, rounding_mode="floor"))
                q = (integrated >= threshold).to(torch.int64)
                return sat(integrated-q*threshold), q.to(torch.float32)

            for t in range(x.shape[1]):
                mac1 = conv(spikes[:, t], weights["conv1"]).to(torch.int64)
                v1, q1 = step(v1, sat(4*mac1 + biases["conv1"][None, :, None, None]))
                mac2 = conv(q1, weights["conv2"]).to(torch.int64)
                v2, q2 = step(v2, sat(4*mac2 + biases["conv2"][None, :, None, None]))
                mac = F.linear(q2.flatten(1), weights["hidden_input"]).to(torch.int64) + F.linear(previous, weights["recurrent"]).to(torch.int64)
                vh, qh = step(vh, sat(4*mac + biases["hidden_input"]))
                logits = sat(logits + sat(4*F.linear(qh, weights["readout"]).to(torch.int64)))
                previous = qh
                if record:
                    for key, value in zip(TRACE_FIELDS, (v1, q1, v2, q2, vh, qh, logits)):
                        traces[key].append(value.to(torch.uint8 if key.endswith("spikes") else torch.int16))
            result = {"window_logits_q16": logits.to(torch.int16).cpu().numpy()}
            if record:
                result.update({key: torch.stack(value, dim=1).cpu().numpy() for key, value in traces.items()})
            return result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul


def run_recordings(packed, parameters, *, device="cpu", batch_size=8, record=False):
    outputs = {}
    for start in range(0, len(packed), batch_size):
        frames = unpack_inputs(packed[start:start+batch_size])
        count, windows = frames.shape[:2]
        result = run_torch(frames.reshape(count*windows, *frames.shape[2:]), parameters, device=device, record=record)
        for key, value in result.items():
            outputs.setdefault(key, []).append(value.reshape(count, windows, *value.shape[1:]))
    result = {key: np.concatenate(values) for key, values in outputs.items()}
    result["predictions"], result["scores"] = aggregate_q16(result["window_logits_q16"])
    return result
