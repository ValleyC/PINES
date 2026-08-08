from __future__ import annotations

from dataclasses import dataclass


def signed_clip_or_wrap(value: int, bits: int, saturate: bool) -> int:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if saturate:
        return min(maximum, max(minimum, value))
    modulus = 1 << bits
    return ((value - minimum) % modulus) + minimum


@dataclass(frozen=True)
class RTLNeuronConfig:
    state_bits: int = 16
    fractional_bits: int = 8
    integration_mode: str = "forward_euler"
    tau_steps: int = 2
    alpha_q: int = 155
    threshold_timing: str = "post_integration"
    reset_mode: str = "subtractive"
    saturate: bool = True


def rtl_lif_step(
    membrane: int,
    current: int,
    threshold: int,
    reset_value: int,
    config: RTLNeuronConfig,
) -> tuple[int, int]:
    """Bit-accurate oracle for ``rtl/lif_neuron.sv`` integer transitions."""

    if config.integration_mode == "forward_euler":
        def integrate(voltage: int) -> int:
            return voltage + (current - voltage) // config.tau_steps
    elif config.integration_mode == "exponential_euler":
        one_q = 1 << config.fractional_bits

        def integrate(voltage: int) -> int:
            return (
                config.alpha_q * voltage + (one_q - config.alpha_q) * current
            ) >> config.fractional_bits
    else:
        raise ValueError("unsupported integration mode")

    def reset(voltage: int) -> int:
        if config.reset_mode == "subtractive":
            return voltage - threshold
        if config.reset_mode == "to_value":
            return reset_value
        raise ValueError("unsupported reset mode")

    if config.threshold_timing == "pre_integration":
        spike = int(membrane >= threshold)
        base = reset(membrane) if spike else membrane
        updated = integrate(base)
    elif config.threshold_timing == "post_integration":
        integrated = integrate(membrane)
        spike = int(integrated >= threshold)
        updated = reset(integrated) if spike else integrated
    else:
        raise ValueError("unsupported threshold timing")
    return signed_clip_or_wrap(updated, config.state_bits, config.saturate), spike

