from __future__ import annotations

import numpy as np
import pytest

from transportcert.statistics import (
    bonferroni_alpha,
    clopper_pearson_upper,
    compose_physical_bound,
    disagreement_bound,
    regularized_incomplete_beta,
    zero_error_sample_size,
)


def test_clopper_pearson_zero_errors_closed_form() -> None:
    expected = 1.0 - 0.05 ** (1.0 / 100)
    assert clopper_pearson_upper(0, 100, 0.05) == pytest.approx(expected)


def test_clopper_pearson_known_value() -> None:
    assert clopper_pearson_upper(5, 10, 0.05) == pytest.approx(0.7775589, abs=1e-6)
    assert clopper_pearson_upper(10, 10, 0.05) == 1.0


def test_regularized_beta_symmetry() -> None:
    left = regularized_incomplete_beta(0.37, 2.5, 4.2)
    right = 1.0 - regularized_incomplete_beta(0.63, 4.2, 2.5)
    assert left == pytest.approx(right, abs=1e-12)


def test_disagreement_and_composition() -> None:
    first = np.asarray([0, 0, 1, 1, 2])
    second = np.asarray([0, 1, 1, 0, 2])
    result = disagreement_bound(first, second, 0.05)
    assert result.errors == 2
    assert result.empirical_rate == 0.4
    assert result.upper_bound >= result.empirical_rate
    assert compose_physical_bound(0.7, 0.4) == 1.0


def test_bonferroni_and_invalid_counts() -> None:
    assert bonferroni_alpha(0.05, 5) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        clopper_pearson_upper(3, 2, 0.05)


def test_zero_error_sample_size_is_minimal() -> None:
    alpha = bonferroni_alpha(0.05, 10)
    for budget in (0.01, 0.02, 0.05):
        samples = zero_error_sample_size(budget, alpha)
        assert clopper_pearson_upper(0, samples, alpha) <= budget
        assert clopper_pearson_upper(0, samples - 1, alpha) > budget
