"""Execution padding for tests of loaded-network reuse on SpiNNaker-1."""


def align_run_steps(minimum_steps: int, colour_bits: int) -> int:
    """End at a packet-colour wrap without changing the observed prefix.

    Hardware repeat tests must establish reset fidelity separately. This
    arithmetic alone does not establish independence or state restoration.
    """
    if minimum_steps < 1 or colour_bits < 0:
        raise ValueError("Run length must be positive and colour width nonnegative")
    period = 1 << colour_bits
    return ((minimum_steps + period - 1) // period) * period
