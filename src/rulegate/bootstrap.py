"""Paired request-level bootstrap comparison for RuleGate H1."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

from .metrics import (
    DEFAULT_POLICY_IDS,
    _policy_evaluation_normalized,
    normalize_inputs,
)


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Percentile interval for ours minus full-sequential mean policy F1."""

    point_estimate: float
    confidence_low: float
    confidence_high: float
    confidence_level: float
    n_resamples: int
    seed: int
    request_count: int


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linearly interpolated percentile (the common type-7 convention)."""

    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def paired_bootstrap_delta_f1(
    gold: Sequence[Iterable[int]],
    ours: Sequence[Iterable[int]],
    full_sequential: Sequence[Iterable[int]],
    *,
    policy_ids: Iterable[int] = DEFAULT_POLICY_IDS,
    n_resamples: int = 10_000,
    seed: int = 484654,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Bootstrap the paired H1 difference on request indices.

    Each replicate draws request indices once and applies that same draw to both
    systems. All configured policies remain in every replicate; unsupported
    policies therefore receive F1 zero through the metric's documented rule.
    """

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    normalized_gold, normalized_ours, policies = normalize_inputs(
        gold, ours, policy_ids=policy_ids
    )
    baseline_gold, normalized_full, baseline_policies = normalize_inputs(
        gold, full_sequential, policy_ids=policies
    )
    if normalized_gold != baseline_gold or policies != baseline_policies:
        raise AssertionError("internal normalization mismatch")
    request_count = len(normalized_gold)
    if request_count == 0:
        raise ValueError("at least one request is required")

    ours_point = _policy_evaluation_normalized(
        normalized_gold, normalized_ours, policies
    ).mean_f1
    full_point = _policy_evaluation_normalized(
        normalized_gold, normalized_full, policies
    ).mean_f1

    generator = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        indices = [generator.randrange(request_count) for _ in range(request_count)]
        sampled_gold = tuple(normalized_gold[index] for index in indices)
        sampled_ours = tuple(normalized_ours[index] for index in indices)
        sampled_full = tuple(normalized_full[index] for index in indices)
        ours_f1 = _policy_evaluation_normalized(
            sampled_gold, sampled_ours, policies
        ).mean_f1
        full_f1 = _policy_evaluation_normalized(
            sampled_gold, sampled_full, policies
        ).mean_f1
        deltas.append(ours_f1 - full_f1)

    deltas.sort()
    alpha = 1.0 - confidence_level
    return PairedBootstrapResult(
        point_estimate=ours_point - full_point,
        confidence_low=_percentile(deltas, alpha / 2.0),
        confidence_high=_percentile(deltas, 1.0 - (alpha / 2.0)),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
        request_count=request_count,
    )
