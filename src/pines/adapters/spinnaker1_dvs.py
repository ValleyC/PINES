"""DVS convolution topology and ideal mapping for physical SpiNNaker-1.

Both convolutional layers and the recurrent layer are represented as device
populations. Only readout and four-window aggregation are host computations.
This target uses native LIF state arithmetic, not the FPGA Q16 state format.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def conv2d_connections(weights, input_shape, stride=2, delay=1.0):
    """PyNN [source, destination, signed weight, delay] rows in CHW order.

Implements the unpadded PyTorch cross-correlation operation, without rotating
the kernel. Rows are NumPy arrays to avoid millions of Python tuple objects.
"""
    weights = np.asarray(weights)
    channels, height, width = input_shape
    outputs, weight_channels, kh, kw = weights.shape
    if channels != weight_channels:
        raise ValueError("convolution input channels do not match weights")
    oh, ow = (height-kh)//stride+1, (width-kw)//stride+1
    if min(oh, ow) < 1 or stride < 1:
        raise ValueError("kernel and stride must fit the input")
    output_y, output_x = np.indices((oh, ow))
    output_offsets = np.arange(oh*ow)
    rows = []
    for oc, ic, ky, kx in zip(*np.nonzero(weights)):
        source = ic*height*width + (output_y.ravel()*stride+ky)*width + output_x.ravel()*stride+kx
        target = oc*oh*ow + output_offsets
        block = np.empty((oh*ow, 4), dtype=np.float64)
        block[:, 0], block[:, 1] = source, target
        block[:, 2], block[:, 3] = weights[oc, ic, ky, kx], delay
        rows.append(block)
    connections = np.concatenate(rows) if rows else np.empty((0,4))
    return connections, (outputs, oh, ow)


def dense_connections(weights, delay=1.0):
    """Convert PyTorch [destination, source] weights to signed PyNN rows."""
    target, source = np.nonzero(weights)
    return np.column_stack((source, target, np.asarray(weights)[target, source],
                            np.full(len(source), delay))).astype(np.float64)


def split_signed_rows(rows):
    """Preserve neuron indices and delays while separating receptor signs."""
    result = {}
    for name, sign in (("excitatory", 1), ("inhibitory", -1)):
        selected = rows[rows[:,2]*sign > 0].copy()
        selected[:,2] *= sign
        if len(selected):
            result[name] = selected
    return result


def bias_connections(bias, spatial_shape=(), delay=1.0):
    values = np.repeat(np.asarray(bias), int(np.prod(spatial_shape)))
    target = np.flatnonzero(values)
    return np.column_stack((np.zeros(len(target)), target, values[target],
                            np.full(len(target), delay))).astype(np.float64)


def aggregate_windows(logits, temperature):
    logits = np.asarray(logits, dtype=np.float64)
    if temperature > 0:
        scaled = logits / temperature
        exp = np.exp(scaled-scaled.max(axis=-1, keepdims=True))
        logits = exp / exp.sum(axis=-1, keepdims=True)
    return logits.mean(axis=-2)


@dataclass(frozen=True)
class SpiNNaker1DVSMapping:
    timestep_ms: float = 1.0
    weight_fractional_bits: int = 6
    bias_fractional_bits: int = 8

    def neuron_parameters(self, tau, threshold):
        ratio = self.timestep_ms/tau
        if not 0 < ratio < 1:
            raise ValueError("Euler matching requires 0 < timestep/tau < 1")
        effective_tau = -self.timestep_ms/math.log1p(-ratio)
        return dict(tau_m=effective_tau, cm=effective_tau, v_rest=0., v_reset=0.,
                    v_thresh=threshold, tau_refrac=0., i_offset=0.)

    def prepare_parameters(self, state):
        """Realize floor-rounded weights and clocked biases before projection."""
        result = {}
        for name, value in state.items():
            bits, fractional = (8, self.weight_fractional_bits) if name.endswith("weight") else (16, self.bias_fractional_bits)
            integer = np.floor(np.asarray(value, dtype=np.float64) * 2**fractional)
            result[name] = np.clip(integer, -(2**(bits-1)), 2**(bits-1)-1) / 2**fractional
        return result

    def contract(self):
        return dict(version="SpiNNaker1DVSMapping/v1", neuron="IF_curr_delta",
            timestep_ms=self.timestep_ms, integration="Euler-matched exponential leak",
            reset="to_value_zero", threshold="post-integration >=", refractory_ms=0,
            weights="signed 8-bit floor-rounded, 6 fractional bits",
            clocked_bias="signed 16-bit floor-rounded, 8 fractional bits",
            state="native device arithmetic, ideal float64 emulator",
            source_offset_steps=2, conv1_latency_steps=1, conv2_latency_steps=2,
            hidden_latency_steps=3, recurrent_delay_steps=1,
            readout="host linear readout using the same quantized output weights",
            aggregation="original checkpoint mean-window softmax at its saved temperature",
            scope="This is not the FPGA floor-rounded Q16-state executor. Its semantic and conformance terms must be measured separately.")

    def emulate(self, state, events, tau, threshold, *, device="cpu", record=False):
        import torch
        from torch.nn import functional as F

        params = {name: torch.as_tensor(value, dtype=torch.float64, device=device)
                  for name, value in self.prepare_parameters(state).items()}
        x = torch.as_tensor(events, dtype=torch.float64, device=device)
        if x.ndim != 5:
            raise ValueError("events must have shape [window,time,channel,height,width]")
        batch, steps, _, height, width = x.shape
        c1, c2 = params["conv1.weight"].shape[0], params["conv2.weight"].shape[0]
        h1, w1 = (height-5)//2+1, (width-5)//2+1
        h2, w2 = (h1-3)//2+1, (w1-3)//2+1
        hidden = params["recurrent.weight"].shape[0]
        v1 = torch.zeros((batch,c1,h1,w1), dtype=x.dtype, device=device)
        v2 = torch.zeros((batch,c2,h2,w2), dtype=x.dtype, device=device)
        vh = torch.zeros((batch,hidden), dtype=x.dtype, device=device)
        qh = torch.zeros_like(vh)
        sums = torch.zeros_like(vh)
        trace = {name: [] for name in ("conv1", "conv2", "hidden")}
        alpha = 1-self.timestep_ms/tau
        if not 0 < alpha < 1:
            raise ValueError("Euler matching requires 0 < timestep/tau < 1")

        def step(v, current):
            v = current-alpha*(current-v)
            q = (v >= threshold).to(x.dtype)
            return torch.where(q != 0, torch.zeros_like(v), v), q

        with torch.no_grad():
            for t in range(steps):
                v1, q1 = step(v1, F.conv2d(x[:,t], params["conv1.weight"], params["conv1.bias"], stride=2))
                v2, q2 = step(v2, F.conv2d(q1, params["conv2.weight"], params["conv2.bias"], stride=2))
                drive = F.linear(q2.flatten(1), params["hidden_input.weight"], params["hidden_input.bias"])
                drive = drive+F.linear(qh, params["recurrent.weight"])
                vh, qh = step(vh, drive)
                sums += qh
                if record:
                    for name, q in (("conv1",q1), ("conv2",q2), ("hidden",qh)):
                        trace[name].append(q.flatten(1).cpu().numpy().astype(np.uint8))
            logits = F.linear(sums, params["readout.weight"]).cpu().numpy()
        result = dict(window_logits=logits)
        if record:
            result.update({name: np.stack(values,axis=1) for name,values in trace.items()})
        return result
