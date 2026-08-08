from __future__ import annotations

import numpy as np
import pytest

from transportcert.models import DenseRecurrentSNN


@pytest.fixture
def small_model() -> DenseRecurrentSNN:
    return DenseRecurrentSNN(
        input_weights=np.asarray([[1.0, 0.4], [0.2, 0.9]]),
        recurrent_weights=np.asarray([[0.0, 0.25], [-0.15, 0.0]]),
        output_weights=np.asarray([[1.0, -0.5], [-0.2, 1.0]]),
        bias=np.asarray([0.0, 0.05]),
        threshold=np.asarray([0.75, 0.8]),
        tau_mem=np.asarray([2.0, 2.5]),
        reset_value=np.zeros(2),
        name="small-test-srnn",
    )


@pytest.fixture
def event_batch() -> np.ndarray:
    rng = np.random.default_rng(41)
    return (rng.random((32, 12, 2)) < 0.3).astype(np.float64)

