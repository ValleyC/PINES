import numpy as np

from transportcert.abstract import _linear_interval


def test_linear_interval_zero_weights_preserves_batch_shape() -> None:
    lower = np.zeros((3, 4))
    upper = np.ones((3, 4))
    weights = np.zeros((4, 5))

    result_lower, result_upper = _linear_interval(lower, upper, weights)

    assert result_lower.shape == (3, 5)
    assert result_upper.shape == (3, 5)
    np.testing.assert_array_equal(result_lower, 0.0)
    np.testing.assert_array_equal(result_upper, 0.0)
