from __future__ import annotations

import numpy as np
import pytest

from transportcert.benchmarks.dvs_gesture import (
    DVSGestureTrainConfig,
    build_dvs_conv_srnn,
)
from transportcert.semantics import ExecutionSemantics, NumericFormat


torch = pytest.importorskip("torch")


def test_dvs_reference_semantics_matches_training_forward() -> None:
    torch.manual_seed(4)
    config = DVSGestureTrainConfig(
        conv1_channels=2,
        conv2_channels=3,
        hidden_size=5,
        epochs=1,
        batch_size=2,
    )
    model = build_dvs_conv_srnn(16, 16, config, output_size=4)
    events = torch.as_tensor(
        (np.random.default_rng(2).random((2, 5, 2, 16, 16)) < 0.05).astype(
            np.float32
        )
    )
    float32 = NumericFormat("float32")
    semantics = ExecutionSemantics(state_format=float32, weight_format=float32)
    model.eval()
    with torch.no_grad():
        training_forward = model(events)
        exact_forward = model(events, semantics)
    torch.testing.assert_close(training_forward, exact_forward, rtol=0, atol=0)


def test_dvs_fixed_semantics_produces_finite_logits() -> None:
    config = DVSGestureTrainConfig(
        conv1_channels=2,
        conv2_channels=2,
        hidden_size=4,
        epochs=1,
        batch_size=1,
    )
    model = build_dvs_conv_srnn(16, 16, config, output_size=3)
    events = torch.zeros((1, 4, 2, 16, 16), dtype=torch.float32)
    fixed = NumericFormat("fixed", 12, 6)
    semantics = ExecutionSemantics(state_format=fixed, weight_format=fixed)
    with torch.no_grad():
        logits = model(events, semantics)
    assert logits.shape == (1, 3)
    assert torch.all(torch.isfinite(logits))
