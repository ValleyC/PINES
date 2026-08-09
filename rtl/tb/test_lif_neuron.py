from __future__ import annotations

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from pines.rtl_reference import RTLNeuronConfig, rtl_lif_step


@cocotb.test()
async def matches_bit_accurate_oracle(dut):
    config = RTLNeuronConfig(
        state_bits=int(os.getenv("STATE_BITS", "16")),
        fractional_bits=int(os.getenv("FRAC_BITS", "8")),
        integration_mode=os.getenv("INTEGRATION", "forward_euler"),
        tau_steps=int(os.getenv("TAU_STEPS", "2")),
        threshold_timing=os.getenv("THRESHOLD_TIMING", "post_integration"),
        reset_mode=os.getenv("RESET_MODE", "subtractive"),
        saturate=bool(int(os.getenv("SATURATE", "1"))),
    )
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst.value = 1
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    threshold = 192
    reset_value = 0
    dut.threshold.value = threshold
    dut.reset_value.value = reset_value
    expected_membrane = 0
    for current in (0, 128, 256, 256, -64, 512, 0):
        expected_membrane, expected_spike = rtl_lif_step(
            expected_membrane, current, threshold, reset_value, config
        )
        dut.synaptic_current.value = current
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        assert dut.valid_out.value == 1
        assert dut.spike.value.integer == expected_spike
        assert dut.membrane.value.signed_integer == expected_membrane

