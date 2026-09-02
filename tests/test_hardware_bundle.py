from __future__ import annotations

import numpy as np

from pines.hardware_bundle import (
    destination_major,
    fixed_integer,
    leak_reciprocal_q24,
    pack_spike_words,
    run_floor_q8q16_integer,
    twos_complement_hex,
)
from pines.semantics import NumericFormat, OverflowMode, RoundingMode


def test_fixed_integer_and_hex_encoding() -> None:
    numeric = NumericFormat(
        "fixed", 8, 6, RoundingMode.FLOOR, OverflowMode.SATURATE
    )
    values = np.asarray([-3.0, -0.01, 0.0, 0.999, 4.0])
    integers = fixed_integer(values, numeric)
    np.testing.assert_array_equal(integers, [-128, -1, 0, 63, 127])
    assert twos_complement_hex(integers, 8) == ["80", "ff", "00", "3f", "7f"]


def test_destination_major_layout() -> None:
    matrix = np.asarray([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(destination_major(matrix), [1, 4, 2, 5, 3, 6])


def test_spike_word_bit_order() -> None:
    frames = np.zeros((1, 2, 9), dtype=np.uint8)
    frames[0, 0, [0, 4, 8]] = 1
    frames[0, 1, [1, 3]] = 1
    assert pack_spike_words(frames, 9) == ["111", "00a"]


def test_integer_oracle_uses_previous_recurrent_spike() -> None:
    frames = np.asarray([[[1], [0], [0]]], dtype=np.uint8)
    trace = run_floor_q8q16_integer(
        frames,
        input_weights_q8=np.asarray([[64]], dtype=np.int8),
        recurrent_weights_q8=np.asarray([[64]], dtype=np.int8),
        output_weights_q8=np.asarray([[64]], dtype=np.int8),
        bias_q16=np.asarray([0], dtype=np.int16),
        threshold_q16=np.asarray([32], dtype=np.int16),
        leak_reciprocal_q24_values=np.asarray([1 << 23], dtype=np.uint32),
    )
    np.testing.assert_array_equal(trace["hidden_spikes"], [[[1], [1], [1]]])
    np.testing.assert_array_equal(trace["predictions"], [0])


def test_leak_mapping_is_floor_q024() -> None:
    observed = leak_reciprocal_q24(np.asarray([5.0, 8.0]))
    np.testing.assert_array_equal(observed, [3355443, 2097152])
