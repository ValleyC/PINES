from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import DenseRecurrentSNN
from .semantics import (
    ExecutionSemantics,
    IntegrationRule,
    ResetRule,
    ThresholdTiming,
)


@dataclass(frozen=True)
class ExecutionTrace:
    membrane: np.ndarray
    spikes: np.ndarray
    logits_over_time: np.ndarray
    final_logits: np.ndarray
    predictions: np.ndarray


def _validate_inputs(model: DenseRecurrentSNN, inputs: np.ndarray) -> np.ndarray:
    events = np.asarray(inputs, dtype=np.float64)
    if events.ndim == 2:
        events = events[None, ...]
    if events.ndim != 3 or events.shape[2] != model.input_size:
        raise ValueError(
            f"inputs must be [batch,time,{model.input_size}], got {events.shape}"
        )
    if not np.all(np.isfinite(events)):
        raise ValueError("inputs must be finite")
    return events


def _decay_and_drive(
    voltage: np.ndarray | float,
    current: np.ndarray | float,
    tau: np.ndarray | float,
    semantics: ExecutionSemantics,
) -> np.ndarray | float:
    dt = semantics.timestep
    if semantics.integration_rule is IntegrationRule.FORWARD_EULER:
        return voltage + dt * (-voltage + current) / tau
    alpha = np.exp(-dt / tau)
    return alpha * voltage + (1.0 - alpha) * current


def _reset_vector(
    voltage: np.ndarray,
    spikes: np.ndarray,
    model: DenseRecurrentSNN,
    semantics: ExecutionSemantics,
) -> np.ndarray:
    if semantics.reset_rule is ResetRule.SUBTRACTIVE:
        return voltage - spikes * model.threshold
    return np.where(spikes > 0, model.reset_value, voltage)


class VectorizedEmulator:
    """Batch-vectorized operational interpreter for dense recurrent LIF SNNs."""

    def run(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        semantics: ExecutionSemantics,
    ) -> ExecutionTrace:
        events = _validate_inputs(model, inputs)
        batch, horizon, _ = events.shape
        neurons = model.hidden_size
        classes = model.output_size
        rng = np.random.default_rng(semantics.randomness.seed)
        q_state = semantics.state_format
        q_weight = semantics.weight_format
        w_in = q_weight.quantize(model.input_weights, rng)
        w_rec = q_weight.quantize(model.recurrent_weights, rng)
        w_out = q_weight.quantize(model.output_weights, rng)
        bias = q_state.quantize(model.bias, rng)
        voltage = np.zeros((batch, neurons), dtype=np.float64)
        previous_spikes = np.zeros_like(voltage)
        logits = np.zeros((batch, classes), dtype=np.float64)
        current_queue = [np.zeros_like(voltage) for _ in range(semantics.synaptic_delay_steps)]
        output_queue = [np.zeros_like(logits) for _ in range(semantics.output_delay_steps)]
        voltages: list[np.ndarray] = []
        spikes_history: list[np.ndarray] = []
        logits_history: list[np.ndarray] = []

        for step in range(horizon):
            raw_current = events[:, step, :] @ w_in + previous_spikes @ w_rec + bias
            raw_current = q_state.quantize(raw_current, rng)
            if current_queue:
                current_queue.append(raw_current)
                current = current_queue.pop(0)
            else:
                current = raw_current

            if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                spikes = (voltage >= model.threshold).astype(np.float64)
                voltage = _reset_vector(voltage, spikes, model, semantics)
                voltage = _decay_and_drive(voltage, current, model.tau_mem, semantics)
            else:
                voltage = _decay_and_drive(voltage, current, model.tau_mem, semantics)
                voltage = q_state.quantize(voltage, rng)
                spikes = (voltage >= model.threshold).astype(np.float64)
                voltage = _reset_vector(voltage, spikes, model, semantics)
            voltage = q_state.quantize(voltage, rng)
            contribution = q_state.quantize(spikes @ w_out, rng)
            if output_queue:
                output_queue.append(contribution)
                delivered = output_queue.pop(0)
            else:
                delivered = contribution
            logits = q_state.quantize(logits + delivered, rng)
            previous_spikes = spikes
            voltages.append(voltage.copy())
            spikes_history.append(spikes.copy())
            logits_history.append(logits.copy())

        final_logits = logits.copy()
        predictions = np.argmax(final_logits, axis=1)
        return ExecutionTrace(
            membrane=np.stack(voltages, axis=1),
            spikes=np.stack(spikes_history, axis=1),
            logits_over_time=np.stack(logits_history, axis=1),
            final_logits=final_logits,
            predictions=predictions,
        )


class ScalarInterpreter:
    """Readable scalar implementation used as the semantic oracle."""

    def run(
        self,
        model: DenseRecurrentSNN,
        inputs: np.ndarray,
        semantics: ExecutionSemantics,
    ) -> ExecutionTrace:
        events = _validate_inputs(model, inputs)
        batch_traces = [self._run_one(model, sample, semantics, index) for index, sample in enumerate(events)]
        return ExecutionTrace(
            membrane=np.stack([trace.membrane[0] for trace in batch_traces]),
            spikes=np.stack([trace.spikes[0] for trace in batch_traces]),
            logits_over_time=np.stack([trace.logits_over_time[0] for trace in batch_traces]),
            final_logits=np.stack([trace.final_logits[0] for trace in batch_traces]),
            predictions=np.asarray([trace.predictions[0] for trace in batch_traces]),
        )

    def _run_one(
        self,
        model: DenseRecurrentSNN,
        events: np.ndarray,
        semantics: ExecutionSemantics,
        sample_index: int,
    ) -> ExecutionTrace:
        rng = np.random.default_rng(semantics.randomness.seed + sample_index)
        qs = semantics.state_format
        qw = semantics.weight_format
        w_in = np.asarray(qw.quantize(model.input_weights, rng))
        w_rec = np.asarray(qw.quantize(model.recurrent_weights, rng))
        w_out = np.asarray(qw.quantize(model.output_weights, rng))
        bias = np.asarray(qs.quantize(model.bias, rng))
        voltage = [0.0] * model.hidden_size
        previous_spikes = [0.0] * model.hidden_size
        logits = [0.0] * model.output_size
        current_queue = [
            [0.0] * model.hidden_size for _ in range(semantics.synaptic_delay_steps)
        ]
        output_queue = [
            [0.0] * model.output_size for _ in range(semantics.output_delay_steps)
        ]
        voltages: list[list[float]] = []
        all_spikes: list[list[float]] = []
        all_logits: list[list[float]] = []

        for step in range(events.shape[0]):
            raw_current = []
            for neuron in range(model.hidden_size):
                drive = float(bias[neuron])
                for source in range(model.input_size):
                    drive += float(events[step, source]) * float(w_in[source, neuron])
                for source in range(model.hidden_size):
                    drive += previous_spikes[source] * float(w_rec[source, neuron])
                raw_current.append(float(qs.quantize(drive, rng)))
            if current_queue:
                current_queue.append(raw_current)
                current = current_queue.pop(0)
            else:
                current = raw_current

            spikes = [0.0] * model.hidden_size
            for neuron in range(model.hidden_size):
                v = voltage[neuron]
                threshold = float(model.threshold[neuron])
                if semantics.threshold_timing is ThresholdTiming.PRE_INTEGRATION:
                    spike = float(v >= threshold)
                    if spike:
                        if semantics.reset_rule is ResetRule.SUBTRACTIVE:
                            v -= threshold
                        else:
                            v = float(model.reset_value[neuron])
                    v = float(
                        _decay_and_drive(v, current[neuron], model.tau_mem[neuron], semantics)
                    )
                else:
                    v = float(
                        _decay_and_drive(v, current[neuron], model.tau_mem[neuron], semantics)
                    )
                    v = float(qs.quantize(v, rng))
                    spike = float(v >= threshold)
                    if spike:
                        if semantics.reset_rule is ResetRule.SUBTRACTIVE:
                            v -= threshold
                        else:
                            v = float(model.reset_value[neuron])
                voltage[neuron] = float(qs.quantize(v, rng))
                spikes[neuron] = spike

            contribution = []
            for output in range(model.output_size):
                value = sum(
                    spikes[neuron] * float(w_out[neuron, output])
                    for neuron in range(model.hidden_size)
                )
                contribution.append(float(qs.quantize(value, rng)))
            if output_queue:
                output_queue.append(contribution)
                delivered = output_queue.pop(0)
            else:
                delivered = contribution
            logits = [
                float(qs.quantize(logits[index] + delivered[index], rng))
                for index in range(model.output_size)
            ]
            previous_spikes = spikes
            voltages.append(voltage.copy())
            all_spikes.append(spikes.copy())
            all_logits.append(logits.copy())

        final = np.asarray(logits, dtype=np.float64)[None, :]
        return ExecutionTrace(
            membrane=np.asarray(voltages, dtype=np.float64)[None, :, :],
            spikes=np.asarray(all_spikes, dtype=np.float64)[None, :, :],
            logits_over_time=np.asarray(all_logits, dtype=np.float64)[None, :, :],
            final_logits=final,
            predictions=np.argmax(final, axis=1),
        )
