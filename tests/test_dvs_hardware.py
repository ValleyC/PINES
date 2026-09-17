import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pines.dvs_hardware import (
    TRACE_FIELDS, aggregate_q16, numpy_step, quantize_model, run_numpy, run_torch,
    unpack_inputs,
)


def small_parameters(seed):
    rng = np.random.default_rng(seed)
    shapes = {"conv1": (2, 2, 5, 5), "conv2": (2, 2, 3, 3),
              "hidden_input": (3, 2), "recurrent": (3, 3), "readout": (2, 3)}
    p = {name+"_weights_q8": rng.integers(-128, 128, shape, dtype=np.int16).astype(np.int8)
         for name, shape in shapes.items()}
    p.update({name+"_bias_q16": rng.integers(-600, 900, shapes[name][0], dtype=np.int16)
              for name in ("conv1", "conv2", "hidden_input")})
    p["threshold_q16"] = np.array([64], dtype=np.int16)
    p["leak_reciprocal_q24"] = np.array([5592405], dtype=np.uint32)
    return p


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_all_state_traces_match_independent_integer_oracle(seed, device):
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, (2, 7, 2, 9, 9), dtype=np.uint8)
    p = small_parameters(seed)
    reference = run_numpy(x, p)
    observed = run_torch(x, p, device=device, record=True)
    for key in (*TRACE_FIELDS, "window_logits_q16"):
        np.testing.assert_array_equal(observed[key], reference[key], err_msg=key)


def test_floor_negative_increment_and_subtractive_reset():
    voltage, spike = numpy_step(np.array([0, 0, 32767]), np.array([-1, 768, 32767]), 256, 5592405)
    # Floor Q0.24 maps 768/3 just below 256, hence no spike in entry 1.
    np.testing.assert_array_equal(voltage, [-1, 255, 32511])
    np.testing.assert_array_equal(spike, [0, 0, 1])


def test_saturation_precedes_threshold_and_reset():
    voltage, spike = numpy_step(np.array([32767, -32768]), np.array([32767, -32768]), 256, 5592405)
    np.testing.assert_array_equal(voltage, [32511, -32768])
    np.testing.assert_array_equal(spike, [1, 0])


def test_input_bit_order_and_window_shape():
    packed = np.zeros((1, 4, 60, 256), dtype=np.uint8)
    packed[0, 2, 4, 0] = 1
    packed[0, 2, 4, 128] = 2
    x = unpack_inputs(packed)
    assert x.shape == (1, 4, 60, 2, 32, 32)
    assert x[0, 2, 4, 0, 0, 0] == 1
    assert x[0, 2, 4, 1, 0, 1] == 1
    assert x.sum() == 2


def test_host_aggregation_is_not_average_logits_or_window_votes():
    logits = np.array([[[25600, 0], [0, 256], [0, 256], [0, 256]]], dtype=np.int16)
    prediction, scores = aggregate_q16(logits)
    assert logits.mean(axis=1).argmax(axis=1).item() == 0
    assert prediction.item() == 1
    np.testing.assert_allclose(scores.sum(axis=1), 1)
    assert aggregate_q16(np.zeros((1, 4, 11), dtype=np.int16))[0].item() == 0


def test_parameter_quantization_and_memory_orientation():
    p = small_parameters(4)
    state = {name+".weight": p[name+"_weights_q8"].astype(np.float32)/64
             for name in ("conv1", "conv2", "hidden_input", "recurrent", "readout")}
    state.update({name+".bias": p[name+"_bias_q16"].astype(np.float32)/256
                  for name in ("conv1", "conv2", "hidden_input")})
    result = quantize_model(state, {"tau_mem": 3., "threshold": .25})
    for key in p:
        np.testing.assert_array_equal(result[key], p[key])


def test_batch_and_window_independence():
    p = small_parameters(12)
    x = np.random.default_rng(2).integers(0, 2, (2, 5, 2, 9, 9), dtype=np.uint8)
    both = run_torch(x, p)["window_logits_q16"]
    separate = np.concatenate([run_torch(x[i:i+1], p)["window_logits_q16"] for i in range(2)])
    np.testing.assert_array_equal(both, separate)


def test_nonbinary_input_is_rejected():
    x = np.full((1, 1, 2, 9, 9), 2, dtype=np.uint8)
    with pytest.raises(ValueError, match="binary"):
        run_torch(x, small_parameters(0))


def test_current_and_logit_saturation_in_full_oracle():
    p = small_parameters(3)
    for key in p:
        if key.endswith("weights_q8"):
            p[key][:] = 127
        elif key.endswith("bias_q16"):
            p[key][:] = 32767
    p["leak_reciprocal_q24"][:] = 1 << 23
    x = np.ones((1, 25, 2, 9, 9), dtype=np.uint8)
    reference, observed = run_numpy(x,p), run_torch(x,p,record=True)
    for key in reference:
        np.testing.assert_array_equal(reference[key], observed[key])
    assert np.all(observed["window_logits_q16"] == 32767)
    assert np.all(observed["conv1_membrane_q16"][:,0] == 16319)
