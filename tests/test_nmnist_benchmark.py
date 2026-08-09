from __future__ import annotations

import torch

from pines.benchmarks.nmnist import (
    NMNISTPreprocessConfig,
    build_feedforward_snn,
)


def test_nmnist_channel_count() -> None:
    assert NMNISTPreprocessConfig().input_channels == 34 * 34 * 2


def test_feedforward_snn_has_no_recurrent_parameter() -> None:
    model = build_feedforward_snn(12, 7, 3, tau_mem=5.0, threshold=1.0)
    logits = model(torch.zeros((4, 6, 12)))
    assert logits.shape == (4, 3)
    assert all("recurrent" not in name for name, _ in model.named_parameters())
