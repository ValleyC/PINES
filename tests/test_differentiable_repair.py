from __future__ import annotations

import pytest
import torch

from pines.differentiable_repair import reference_margin_deficit


def test_reference_margin_deficit_is_directional() -> None:
    reference = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    predictions = torch.tensor([0, 1])
    assert reference_margin_deficit(reference, reference, predictions).item() == 0.0

    stronger = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    assert reference_margin_deficit(stronger, reference, predictions).item() == 0.0

    weaker = torch.tensor([[1.5, 1.0, 0.0], [0.0, 1.25, 1.0]])
    assert reference_margin_deficit(weaker, reference, predictions).item() > 0.0


def test_reference_margin_deficit_reductions_agree() -> None:
    reference = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    predictions = torch.tensor([0, 1])
    values = reference_margin_deficit(
        target, reference, predictions, reduction="none"
    )
    torch.testing.assert_close(
        values.sum(),
        reference_margin_deficit(target, reference, predictions, reduction="sum"),
    )
    torch.testing.assert_close(
        values.mean(),
        reference_margin_deficit(target, reference, predictions, reduction="mean"),
    )


def test_reference_margin_deficit_validates_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        reference_margin_deficit(
            torch.zeros((2, 3)), torch.zeros((2, 2)), torch.zeros(2, dtype=torch.long)
        )
    with pytest.raises(ValueError, match="reference_predictions"):
        reference_margin_deficit(
            torch.zeros((2, 3)), torch.zeros((2, 3)), torch.zeros(1, dtype=torch.long)
        )
