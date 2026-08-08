from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .emulator import ExecutionTrace
from .models import DenseRecurrentSNN
from .semantics import (
    ExecutionSemantics,
    IntegrationRule,
    OverflowMode,
    ResetRule,
    RoundingMode,
    ThresholdTiming,
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("TorchEmulator requires transportcert[torch]") from error
    return torch


@dataclass(frozen=True)
class TorchExecutionTrace:
    membrane: Any
    spikes: Any
    logits_over_time: Any
    final_logits: Any
    predictions: Any

    def numpy(self) -> ExecutionTrace:
        def convert(value: Any) -> np.ndarray:
            return value.detach().cpu().numpy()

        return ExecutionTrace(
            membrane=convert(self.membrane),
            spikes=convert(self.spikes),
            logits_over_time=convert(self.logits_over_time),
            final_logits=convert(self.final_logits),
            predictions=convert(self.predictions),
        )


class TorchEmulator:
    """Vectorized PyTorch executor matching the scalar operational semantics."""

    def __init__(self, device: str = "cpu", dtype: Any | None = None) -> None:
        torch = _torch()
        self.device = torch.device(device)
        self.dtype = dtype or torch.float64

    def _tensor(self, value: np.ndarray) -> Any:
        torch = _torch()
        return torch.tensor(
            np.array(value, copy=True), dtype=self.dtype, device=self.device
        )

    def _quantize(self, value: Any, numeric: Any, generator: Any) -> Any:
        torch = _torch()
        if numeric.kind == "float32":
            # Operational semantics use binary64 arithmetic with explicit
            # float32 storage/cast points, matching NumericFormat.quantize and
            # the scalar/vector reference interpreters.  Cast back to the
            # engine dtype so intervening operations do not silently become
            # all-float32 arithmetic.
            return value.to(torch.float32).to(self.dtype)
        if numeric.kind == "float64":
            return value
        scale = float(1 << numeric.fractional_bits)
        scaled = value * scale
        if numeric.rounding is RoundingMode.NEAREST_EVEN:
            integers = torch.round(scaled)
        elif numeric.rounding is RoundingMode.FLOOR:
            integers = torch.floor(scaled)
        elif numeric.rounding is RoundingMode.TRUNCATE:
            integers = torch.trunc(scaled)
        else:
            lower = torch.floor(scaled)
            random = torch.rand(
                scaled.shape,
                dtype=scaled.dtype,
                device=scaled.device,
                generator=generator,
            )
            integers = lower + (random < (scaled - lower)).to(scaled.dtype)
        minimum = -(1 << (numeric.total_bits - 1))
        maximum = (1 << (numeric.total_bits - 1)) - 1
        if numeric.overflow is OverflowMode.SATURATE:
            integers = torch.clamp(integers, minimum, maximum)
        else:
            modulus = 1 << numeric.total_bits
            integers = torch.remainder(integers - minimum, modulus) + minimum
        return integers / scale

    def run(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray | Any,
        semantics: ExecutionSemantics,
    ) -> TorchExecutionTrace:
        torch = _torch()
        events = torch.as_tensor(inputs, dtype=self.dtype, device=self.device)
        if events.ndim == 2:
            events = events.unsqueeze(0)
        if events.ndim != 3 or events.shape[2] != model.input_size:
            raise ValueError(
                f"inputs must be [batch,time,{model.input_size}], got {tuple(events.shape)}"
            )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(semantics.randomness.seed)
        qs = semantics.state_format
        qw = semantics.weight_format
        w_in = self._quantize(self._tensor(model.input_weights), qw, generator)
        w_rec = self._quantize(self._tensor(model.recurrent_weights), qw, generator)
        w_out = self._quantize(self._tensor(model.output_weights), qw, generator)
        bias = self._quantize(self._tensor(model.bias), qs, generator)
        threshold = self._tensor(model.threshold)
        tau = self._tensor(model.tau_mem)
        reset_value = self._tensor(model.reset_value)
        batch, horizon, _ = events.shape
        voltage = torch.zeros((batch, model.hidden_size), dtype=self.dtype, device=self.device)
        previous_spikes = torch.zeros_like(voltage)
        logits = torch.zeros((batch, model.output_size), dtype=self.dtype, device=self.device)
        current_queue = [torch.zeros_like(voltage) for _ in range(semantics.synaptic_delay_steps)]
        output_queue = [torch.zeros_like(logits) for _ in range(semantics.output_delay_steps)]
        voltages = []
        spikes_history = []
        logits_history = []

        def integrate(state: Any, current: Any) -> Any:
            if semantics.integration_rule is IntegrationRule.FORWARD_EULER:
                return state + semantics.timestep * (-state + current) / tau
            alpha = torch.exp(-semantics.timestep / tau)
            return alpha * state + (1.0 - alpha) * current

        def reset(state: Any, spikes: Any) -> Any:
            if semantics.reset_rule is ResetRule.SUBTRACTIVE:
                return state - spikes * threshold
            return torch.where(spikes > 0, reset_value, state)

        for step in range(horizon):
            raw_current = events[:, step, :] @ w_in + previous_spikes @ w_rec + bias
            raw_current = self._quantize(raw_current, qs, generator)
            if current_queue:
                current_queue.append(raw_current)
                current = current_queue.pop(0)
            else:
                current = raw_current
            if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                spikes = (voltage >= threshold).to(self.dtype)
                voltage = integrate(reset(voltage, spikes), current)
            else:
                voltage = self._quantize(integrate(voltage, current), qs, generator)
                spikes = (voltage >= threshold).to(self.dtype)
                voltage = reset(voltage, spikes)
            voltage = self._quantize(voltage, qs, generator)
            contribution = self._quantize(spikes @ w_out, qs, generator)
            if output_queue:
                output_queue.append(contribution)
                delivered = output_queue.pop(0)
            else:
                delivered = contribution
            logits = self._quantize(logits + delivered, qs, generator)
            previous_spikes = spikes
            voltages.append(voltage)
            spikes_history.append(spikes)
            logits_history.append(logits)
        return TorchExecutionTrace(
            membrane=torch.stack(voltages, dim=1),
            spikes=torch.stack(spikes_history, dim=1),
            logits_over_time=torch.stack(logits_history, dim=1),
            final_logits=logits,
            predictions=torch.argmax(logits, dim=1),
        )
