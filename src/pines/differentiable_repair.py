from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .emulator import VectorizedEmulator
from .models import DenseRecurrentSNN
from .semantics import (
    ExecutionSemantics,
    IntegrationRule,
    OverflowMode,
    ResetRule,
    RoundingMode,
    ThresholdTiming,
)


@dataclass(frozen=True)
class DifferentiableRepairCandidate:
    model: DenseRecurrentSNN
    loss_history: tuple[float, ...]
    steps: int


class DifferentiableSurrogateRepair:
    """Optimize a smooth surrogate of uncertified mass without labels."""

    def calibrate(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        reference: ExecutionSemantics,
        target: ExecutionSemantics,
        *,
        steps: int = 50,
        learning_rate: float = 0.03,
        spike_temperature: float = 0.15,
    ) -> DifferentiableRepairCandidate:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("differentiable repair requires pines-snn[torch]") from error
        if steps <= 0:
            return DifferentiableRepairCandidate(model, (), 0)
        if not target.randomness.deterministic:
            raise ValueError("differentiable repair requires deterministic target semantics")
        torch.manual_seed(target.randomness.seed)
        dtype = torch.float64
        events = torch.as_tensor(np.asarray(inputs), dtype=dtype)
        if events.ndim == 2:
            events = events.unsqueeze(0)
        reference_trace = VectorizedEmulator().run(model, np.asarray(inputs), reference)
        reference_logits = torch.as_tensor(reference_trace.final_logits, dtype=dtype)
        reference_states = torch.as_tensor(reference_trace.membrane, dtype=dtype)
        reference_spikes = torch.as_tensor(reference_trace.spikes, dtype=dtype)
        reference_prediction = torch.argmax(reference_logits, dim=1)
        input_weights = torch.tensor(model.input_weights.copy(), dtype=dtype)
        recurrent_weights = torch.tensor(model.recurrent_weights.copy(), dtype=dtype)
        output_weights = torch.tensor(model.output_weights.copy(), dtype=dtype)
        reset_value = torch.tensor(model.reset_value.copy(), dtype=dtype)
        threshold = torch.nn.Parameter(torch.as_tensor(model.threshold.copy(), dtype=dtype))
        log_tau = torch.nn.Parameter(torch.log(torch.as_tensor(model.tau_mem.copy(), dtype=dtype)))
        bias = torch.nn.Parameter(torch.as_tensor(model.bias.copy(), dtype=dtype))
        log_scale = torch.nn.Parameter(torch.zeros(model.hidden_size, dtype=dtype))
        optimizer = torch.optim.Adam(
            (threshold, log_tau, bias, log_scale), lr=learning_rate
        )
        history: list[float] = []

        def quantize_ste(value, numeric):
            if numeric.kind == "float32":
                hard = value.to(torch.float32).to(value.dtype)
                return value + (hard - value).detach()
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
                integers = torch.round(scaled)
            minimum = -(1 << (numeric.total_bits - 1))
            maximum = (1 << (numeric.total_bits - 1)) - 1
            if numeric.overflow is OverflowMode.SATURATE:
                integers = torch.clamp(integers, minimum, maximum)
            else:
                modulus = 1 << numeric.total_bits
                integers = torch.remainder(integers - minimum, modulus) + minimum
            hard = integers / scale
            return value + (hard - value).detach()

        def soft_run():
            tau = torch.exp(log_tau).clamp(min=1e-4)
            incoming_scale = torch.exp(log_scale).clamp(0.25, 4.0)
            w_in = quantize_ste(
                input_weights * incoming_scale.unsqueeze(0), target.weight_format
            )
            w_rec = quantize_ste(
                recurrent_weights * incoming_scale.unsqueeze(0), target.weight_format
            )
            w_out = quantize_ste(output_weights, target.weight_format)
            effective_bias = quantize_ste(bias, target.state_format)
            voltage = torch.zeros((events.shape[0], model.hidden_size), dtype=dtype)
            previous_spikes = torch.zeros_like(voltage)
            logits = torch.zeros((events.shape[0], model.output_size), dtype=dtype)
            current_queue = [
                torch.zeros_like(voltage) for _ in range(target.synaptic_delay_steps)
            ]
            output_queue = [
                torch.zeros_like(logits) for _ in range(target.output_delay_steps)
            ]
            states = []
            spike_traces = []

            def integrate(state, current):
                if target.integration_rule is IntegrationRule.FORWARD_EULER:
                    return state + target.timestep * (-state + current) / tau
                alpha = torch.exp(-target.timestep / tau)
                return alpha * state + (1.0 - alpha) * current

            def reset(state, spikes):
                if target.reset_rule is ResetRule.SUBTRACTIVE:
                    return state - spikes * threshold
                return (1.0 - spikes) * state + spikes * reset_value

            for time in range(events.shape[1]):
                current = events[:, time, :] @ w_in + previous_spikes @ w_rec + effective_bias
                current = quantize_ste(current, target.state_format)
                if current_queue:
                    current_queue.append(current)
                    current = current_queue.pop(0)
                if target.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                    spikes = torch.sigmoid((voltage - threshold) / spike_temperature)
                    voltage = integrate(reset(voltage, spikes), current)
                else:
                    voltage = quantize_ste(
                        integrate(voltage, current), target.state_format
                    )
                    spikes = torch.sigmoid((voltage - threshold) / spike_temperature)
                    voltage = reset(voltage, spikes)
                voltage = quantize_ste(voltage, target.state_format)
                contribution = quantize_ste(spikes @ w_out, target.state_format)
                if output_queue:
                    output_queue.append(contribution)
                    contribution = output_queue.pop(0)
                logits = quantize_ste(logits + contribution, target.state_format)
                previous_spikes = spikes
                states.append(voltage)
                spike_traces.append(spikes)
            return logits, torch.stack(states, dim=1), torch.stack(spike_traces, dim=1)

        for _ in range(steps):
            optimizer.zero_grad()
            logits, states, spikes = soft_run()
            rows = torch.arange(events.shape[0])
            selected = logits[rows, reference_prediction]
            masked = logits.clone()
            masked[rows, reference_prediction] = -torch.inf
            margin = selected - torch.max(masked, dim=1).values
            uncertified_mass = torch.nn.functional.softplus(-margin).mean()
            state_loss = torch.mean(torch.abs(states - reference_states)) / (
                torch.mean(torch.abs(reference_states)) + 1e-9
            )
            spike_loss = torch.mean(torch.abs(spikes - reference_spikes))
            logit_loss = torch.mean(torch.abs(logits - reference_logits)) / (
                torch.mean(torch.abs(reference_logits)) + 1e-9
            )
            loss = uncertified_mass + 0.02 * state_loss + 0.10 * spike_loss + 0.02 * logit_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                threshold.clamp_(min=1e-4)
                log_tau.clamp_(min=np.log(1e-4), max=np.log(1e4))
                log_scale.clamp_(min=np.log(0.25), max=np.log(4.0))
            history.append(float(loss.detach()))

        scale = torch.exp(log_scale).detach().numpy()
        repaired_input = model.input_weights * scale[None, :]
        repaired_recurrent = model.recurrent_weights * scale[None, :]
        repaired = model.with_parameters(
            input_weights=repaired_input,
            recurrent_weights=repaired_recurrent,
            threshold=threshold.detach().numpy(),
            tau_mem=torch.exp(log_tau).detach().numpy(),
            bias=bias.detach().numpy(),
        )
        return DifferentiableRepairCandidate(repaired, tuple(history), steps)


def reference_margin_deficit(
    logits,
    reference_logits,
    reference_predictions,
    *,
    reduction: str = "mean",
):
    """Return a scale-normalized deficit in the source decision margin.

    The loss is zero once the target execution matches or exceeds the source
    margin.  Unlike hard pseudo-label cross-entropy, it does not force
    low-confidence source decisions toward arbitrarily high confidence.
    """

    import torch

    if logits.shape != reference_logits.shape:
        raise ValueError("target and reference logits must have the same shape")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if reference_predictions.shape != (logits.shape[0],):
        raise ValueError("reference_predictions must have shape [batch]")
    rows = torch.arange(logits.shape[0], device=logits.device)

    target_selected = logits[rows, reference_predictions]
    target_others = logits.clone()
    target_others[rows, reference_predictions] = -torch.inf
    target_margin = target_selected - torch.max(target_others, dim=1).values

    reference_selected = reference_logits[rows, reference_predictions]
    reference_others = reference_logits.clone()
    reference_others[rows, reference_predictions] = -torch.inf
    reference_margin = (
        reference_selected - torch.max(reference_others, dim=1).values
    ).detach()
    normalized = torch.relu(reference_margin - target_margin) / (
        1.0 + torch.abs(reference_margin)
    )
    losses = normalized.square()
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")
