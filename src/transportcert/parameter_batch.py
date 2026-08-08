from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .emulator import _validate_inputs
from .models import DenseRecurrentSNN
from .semantics import (
    ExecutionSemantics,
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
)
from .torch_emulator import TorchEmulator, _torch


@dataclass(frozen=True)
class ParameterBatchResult:
    final_logits: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True)
class CartesianParameterBatchResult:
    """Outputs indexed by input first and parameter point second."""

    final_logits: np.ndarray
    predictions: np.ndarray


class TorchParameterBatchEmulator:
    """Execute one event sample at many timestep/threshold parameter points."""

    def __init__(self, device: str = "cpu", dtype: Any | None = None) -> None:
        torch = _torch()
        self.device = torch.device(device)
        self.dtype = dtype or torch.float64

    def run(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        semantics: ExecutionSemantics,
        timesteps: np.ndarray,
        threshold_scales: np.ndarray,
        *,
        batch_size: int = 1024,
    ) -> ParameterBatchResult:
        torch = _torch()
        events = _validate_inputs(model, inputs)
        if len(events) != 1:
            raise ValueError("parameter-batch execution accepts one event sample")
        timestep_values = np.asarray(timesteps, dtype=np.float64)
        threshold_values = np.asarray(threshold_scales, dtype=np.float64)
        if (
            timestep_values.ndim != 1
            or threshold_values.shape != timestep_values.shape
        ):
            raise ValueError("timesteps and threshold scales must be aligned vectors")
        if len(timestep_values) < 1 or batch_size < 1:
            raise ValueError("parameter points and batch size must be positive")
        if (
            np.any(~np.isfinite(timestep_values))
            or np.any(timestep_values <= 0.0)
            or np.any(~np.isfinite(threshold_values))
            or np.any(threshold_values <= 0.0)
        ):
            raise ValueError("parameter values must be finite and positive")
        if not semantics.randomness.deterministic:
            raise NotImplementedError("parameter batches require determinism")

        engine = TorchEmulator(device=str(self.device), dtype=self.dtype)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(semantics.randomness.seed)
        qs = semantics.state_format
        qw = semantics.weight_format
        w_in = engine._quantize(engine._tensor(model.input_weights), qw, generator)
        w_rec = engine._quantize(
            engine._tensor(model.recurrent_weights), qw, generator
        )
        w_out = engine._quantize(engine._tensor(model.output_weights), qw, generator)
        bias = engine._quantize(engine._tensor(model.bias), qs, generator)
        tau = engine._tensor(model.tau_mem)
        reset_value = engine._tensor(model.reset_value)
        base_threshold = engine._tensor(model.threshold)
        event_tensor = torch.as_tensor(
            events[0], dtype=self.dtype, device=self.device
        )
        base_drive = event_tensor @ w_in + bias
        logits_parts = []
        prediction_parts = []

        with torch.no_grad():
            for start in range(0, len(timestep_values), batch_size):
                stop = min(start + batch_size, len(timestep_values))
                dt = torch.as_tensor(
                    timestep_values[start:stop],
                    dtype=self.dtype,
                    device=self.device,
                )[:, None]
                threshold_scale = torch.as_tensor(
                    threshold_values[start:stop],
                    dtype=self.dtype,
                    device=self.device,
                )[:, None]
                threshold = base_threshold[None, :] * threshold_scale
                batch = stop - start
                voltage = torch.zeros(
                    (batch, model.hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                )
                previous_spikes = torch.zeros_like(voltage)
                logits = torch.zeros(
                    (batch, model.output_size),
                    dtype=self.dtype,
                    device=self.device,
                )
                current_queue = [
                    torch.zeros_like(voltage)
                    for _ in range(semantics.synaptic_delay_steps)
                ]
                output_queue = [
                    torch.zeros_like(logits)
                    for _ in range(semantics.output_delay_steps)
                ]

                def integrate(state: Any, current: Any) -> Any:
                    if semantics.integration_rule is IntegrationRule.FORWARD_EULER:
                        return state + dt * (-state + current) / tau
                    alpha = torch.exp(-dt / tau)
                    return alpha * state + (1.0 - alpha) * current

                def reset(state: Any, spikes: Any) -> Any:
                    if semantics.reset_rule is ResetRule.SUBTRACTIVE:
                        return state - spikes * threshold
                    return torch.where(spikes > 0.0, reset_value, state)

                for step in range(event_tensor.shape[0]):
                    raw_current = base_drive[step][None, :] + previous_spikes @ w_rec
                    raw_current = engine._quantize(raw_current, qs, generator)
                    if current_queue:
                        current_queue.append(raw_current)
                        current = current_queue.pop(0)
                    else:
                        current = raw_current
                    if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                        spikes = (voltage >= threshold).to(self.dtype)
                        voltage = integrate(reset(voltage, spikes), current)
                    else:
                        voltage = engine._quantize(
                            integrate(voltage, current), qs, generator
                        )
                        spikes = (voltage >= threshold).to(self.dtype)
                        voltage = reset(voltage, spikes)
                    voltage = engine._quantize(voltage, qs, generator)
                    contribution = engine._quantize(
                        spikes @ w_out, qs, generator
                    )
                    if output_queue:
                        output_queue.append(contribution)
                        delivered = output_queue.pop(0)
                    else:
                        delivered = contribution
                    logits = engine._quantize(logits + delivered, qs, generator)
                    previous_spikes = spikes
                logits_parts.append(logits.detach().cpu().numpy())
                prediction_parts.append(
                    torch.argmax(logits, dim=1).detach().cpu().numpy()
                )
        return ParameterBatchResult(
            final_logits=np.concatenate(logits_parts),
            predictions=np.concatenate(prediction_parts),
        )

    def run_cartesian(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        semantics: ExecutionSemantics,
        timesteps: np.ndarray,
        threshold_scales: np.ndarray,
        *,
        input_batch_size: int = 256,
        parameter_batch_size: int = 256,
    ) -> CartesianParameterBatchResult:
        """Execute every input at every aligned timestep/threshold point."""

        torch = _torch()
        events = _validate_inputs(model, inputs)
        timestep_values = np.asarray(timesteps, dtype=np.float64)
        threshold_values = np.asarray(threshold_scales, dtype=np.float64)
        if (
            timestep_values.ndim != 1
            or threshold_values.shape != timestep_values.shape
        ):
            raise ValueError("timesteps and threshold scales must be aligned vectors")
        if (
            len(events) < 1
            or len(timestep_values) < 1
            or input_batch_size < 1
            or parameter_batch_size < 1
        ):
            raise ValueError("input, parameter, and batch counts must be positive")
        if (
            np.any(~np.isfinite(timestep_values))
            or np.any(timestep_values <= 0.0)
            or np.any(~np.isfinite(threshold_values))
            or np.any(threshold_values <= 0.0)
        ):
            raise ValueError("parameter values must be finite and positive")
        if not semantics.randomness.deterministic:
            raise NotImplementedError("parameter batches require determinism")

        engine = TorchEmulator(device=str(self.device), dtype=self.dtype)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(semantics.randomness.seed)
        qs = semantics.state_format
        qw = semantics.weight_format
        w_in = engine._quantize(engine._tensor(model.input_weights), qw, generator)
        w_rec = engine._quantize(
            engine._tensor(model.recurrent_weights), qw, generator
        )
        w_out = engine._quantize(engine._tensor(model.output_weights), qw, generator)
        bias = engine._quantize(engine._tensor(model.bias), qs, generator)
        tau = engine._tensor(model.tau_mem)
        reset_value = engine._tensor(model.reset_value)
        base_threshold = engine._tensor(model.threshold)
        all_logits = []
        all_predictions = []

        with torch.no_grad():
            for input_start in range(0, len(events), input_batch_size):
                input_stop = min(input_start + input_batch_size, len(events))
                event_tensor = torch.as_tensor(
                    events[input_start:input_stop],
                    dtype=self.dtype,
                    device=self.device,
                )
                base_drive = event_tensor @ w_in + bias
                logits_by_parameter = []
                predictions_by_parameter = []
                for parameter_start in range(
                    0, len(timestep_values), parameter_batch_size
                ):
                    parameter_stop = min(
                        parameter_start + parameter_batch_size,
                        len(timestep_values),
                    )
                    dt = torch.as_tensor(
                        timestep_values[parameter_start:parameter_stop],
                        dtype=self.dtype,
                        device=self.device,
                    )[None, :, None]
                    threshold_scale = torch.as_tensor(
                        threshold_values[parameter_start:parameter_stop],
                        dtype=self.dtype,
                        device=self.device,
                    )[None, :, None]
                    threshold = base_threshold[None, None, :] * threshold_scale
                    input_count = input_stop - input_start
                    parameter_count = parameter_stop - parameter_start
                    voltage = torch.zeros(
                        (input_count, parameter_count, model.hidden_size),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    previous_spikes = torch.zeros_like(voltage)
                    logits = torch.zeros(
                        (input_count, parameter_count, model.output_size),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    current_queue = [
                        torch.zeros_like(voltage)
                        for _ in range(semantics.synaptic_delay_steps)
                    ]
                    output_queue = [
                        torch.zeros_like(logits)
                        for _ in range(semantics.output_delay_steps)
                    ]

                    def integrate(state: Any, current: Any) -> Any:
                        if semantics.integration_rule is IntegrationRule.FORWARD_EULER:
                            return state + dt * (-state + current) / tau
                        alpha = torch.exp(-dt / tau)
                        return alpha * state + (1.0 - alpha) * current

                    def reset(state: Any, spikes: Any) -> Any:
                        if semantics.reset_rule is ResetRule.SUBTRACTIVE:
                            return state - spikes * threshold
                        return torch.where(spikes > 0.0, reset_value, state)

                    for step in range(event_tensor.shape[1]):
                        raw_current = (
                            base_drive[:, step, None, :] + previous_spikes @ w_rec
                        )
                        raw_current = engine._quantize(raw_current, qs, generator)
                        if current_queue:
                            current_queue.append(raw_current)
                            current = current_queue.pop(0)
                        else:
                            current = raw_current
                        if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                            spikes = (voltage >= threshold).to(self.dtype)
                            voltage = integrate(reset(voltage, spikes), current)
                        else:
                            voltage = engine._quantize(
                                integrate(voltage, current), qs, generator
                            )
                            spikes = (voltage >= threshold).to(self.dtype)
                            voltage = reset(voltage, spikes)
                        voltage = engine._quantize(voltage, qs, generator)
                        contribution = engine._quantize(
                            spikes @ w_out, qs, generator
                        )
                        if output_queue:
                            output_queue.append(contribution)
                            delivered = output_queue.pop(0)
                        else:
                            delivered = contribution
                        logits = engine._quantize(logits + delivered, qs, generator)
                        previous_spikes = spikes
                    logits_by_parameter.append(logits.detach().cpu().numpy())
                    predictions_by_parameter.append(
                        torch.argmax(logits, dim=2).detach().cpu().numpy()
                    )
                all_logits.append(np.concatenate(logits_by_parameter, axis=1))
                all_predictions.append(
                    np.concatenate(predictions_by_parameter, axis=1)
                )
        return CartesianParameterBatchResult(
            final_logits=np.concatenate(all_logits, axis=0),
            predictions=np.concatenate(all_predictions, axis=0),
        )
