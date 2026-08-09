from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("beta parameters must be positive")
    if not 0 <= x <= 1:
        raise ValueError("x must be in [0,1]")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def beta_quantile(probability: float, a: float, b: float) -> float:
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if regularized_incomplete_beta(middle, a, b) < probability:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def clopper_pearson_upper(errors: int, samples: int, alpha: float) -> float:
    """Exact one-sided binomial upper confidence limit.

    With probability at least ``1-alpha``, the true Bernoulli error rate is no
    greater than this value under independent identically distributed draws.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if errors < 0 or errors > samples:
        raise ValueError("errors must be between zero and samples")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if errors == samples:
        return 1.0
    if errors == 0:
        return 1.0 - alpha ** (1.0 / samples)
    return beta_quantile(1.0 - alpha, errors + 1.0, samples - errors)


def bonferroni_alpha(total_alpha: float, comparisons: int) -> float:
    if not 0 < total_alpha < 1:
        raise ValueError("total_alpha must be in (0,1)")
    if comparisons <= 0:
        raise ValueError("comparisons must be positive")
    return total_alpha / comparisons


def zero_error_sample_size(budget: float, alpha: float) -> int:
    """Minimum paired samples whose zero-error upper limit is within budget."""

    if not 0 < budget < 1:
        raise ValueError("budget must be in (0,1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    samples = max(1, math.ceil(math.log(alpha) / math.log1p(-budget)))
    while clopper_pearson_upper(0, samples, alpha) > budget:
        samples += 1
    while samples > 1 and clopper_pearson_upper(0, samples - 1, alpha) <= budget:
        samples -= 1
    return samples


@dataclass(frozen=True)
class DisagreementBound:
    errors: int
    samples: int
    empirical_rate: float
    upper_bound: float
    alpha: float


@dataclass(frozen=True)
class DisagreementTightnessWitness:
    disagreement_rate: float
    reference_favoring_labels: np.ndarray
    target_favoring_labels: np.ndarray
    reference_minus_target_accuracy: float
    target_minus_reference_accuracy: float


def disagreement_tightness_witness(
    reference: np.ndarray,
    target: np.ndarray,
) -> DisagreementTightnessWitness:
    """Construct labelings that attain both signs of the disagreement bound."""

    reference_predictions = np.asarray(reference)
    target_predictions = np.asarray(target)
    if (
        reference_predictions.ndim != 1
        or target_predictions.shape != reference_predictions.shape
    ):
        raise ValueError("prediction arrays must be one-dimensional and shape-matched")
    if reference_predictions.size == 0:
        raise ValueError("at least one paired prediction is required")
    reference_labels = np.array(reference_predictions, copy=True)
    target_labels = np.array(target_predictions, copy=True)
    disagreement = float(np.mean(reference_predictions != target_predictions))
    reference_change = float(
        np.mean(reference_predictions == reference_labels)
        - np.mean(target_predictions == reference_labels)
    )
    target_change = float(
        np.mean(target_predictions == target_labels)
        - np.mean(reference_predictions == target_labels)
    )
    reference_labels.setflags(write=False)
    target_labels.setflags(write=False)
    return DisagreementTightnessWitness(
        disagreement_rate=disagreement,
        reference_favoring_labels=reference_labels,
        target_favoring_labels=target_labels,
        reference_minus_target_accuracy=reference_change,
        target_minus_reference_accuracy=target_change,
    )


def disagreement_bound(
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
) -> DisagreementBound:
    left = np.asarray(first)
    right = np.asarray(second)
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("prediction arrays must be one-dimensional and shape-matched")
    errors = int(np.count_nonzero(left != right))
    samples = int(left.size)
    if samples == 0:
        raise ValueError("at least one paired prediction is required")
    return DisagreementBound(
        errors=errors,
        samples=samples,
        empirical_rate=errors / samples,
        upper_bound=clopper_pearson_upper(errors, samples, alpha),
        alpha=alpha,
    )


def compose_physical_bound(semantic: float, conformance: float) -> float:
    if not 0 <= semantic <= 1 or not 0 <= conformance <= 1:
        raise ValueError("component bounds must be in [0,1]")
    return min(1.0, semantic + conformance)
