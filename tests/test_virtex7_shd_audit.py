import importlib.util
from pathlib import Path

import numpy as np

from pines.hardware_bundle import run_floor_q8q16_integer

script = Path(__file__).resolve().parents[1] / "experiments/prepare_virtex7_shd_audit.py"
spec = importlib.util.spec_from_file_location("virtex_shd_audit", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_batched_predictions_preserve_exact_integer_outputs_and_order():
    rng = np.random.default_rng(29)
    frames = rng.integers(0, 2, size=(5, 8, 3), dtype=np.uint8)
    parameters = dict(input_weights_q8=rng.integers(-128, 128, size=(3, 2)),
        recurrent_weights_q8=rng.integers(-128, 128, size=(2, 2)),
        output_weights_q8=rng.integers(-128, 128, size=(2, 2)),
        bias_q16=np.array([20, -10]), threshold_q16=np.array([70, 80]),
        leak_reciprocal_q24=np.array([2**22, 2**23]))
    oracle_args = dict(parameters)
    oracle_args["leak_reciprocal_q24_values"] = oracle_args.pop("leak_reciprocal_q24")
    expected = run_floor_q8q16_integer(frames, **oracle_args)
    for batch in (1, 2, 5):
        predictions, logits = module.integer_predictions(frames, parameters, batch)
        np.testing.assert_array_equal(predictions, expected["predictions"])
        np.testing.assert_array_equal(logits, expected["final_logits_q16"])
