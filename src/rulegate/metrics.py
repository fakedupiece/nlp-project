"""Evaluation metrics for the Cisneros-Velarde policy benchmark.

The paper's Appendix D metric is deliberately implemented here rather than via
``sklearn``.  In particular, a prediction that contains both the gold policy and
an additional policy contributes both ``TP`` and ``FN*`` for the gold policy.
That makes this metric different from ordinary one-vs-rest F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


DEFAULT_POLICY_IDS: tuple[int, ...] = tuple(range(1, 10))


@dataclass(frozen=True)
class PolicyCounts:
    """Appendix D counts for one policy."""

    tp: int = 0
    fp1: int = 0
    fp2: int = 0
    fn: int = 0
    fn_star: int = 0


@dataclass(frozen=True)
class PolicyMetric:
    policy_id: int
    support: int
    counts: PolicyCounts
    f1: float


@dataclass(frozen=True)
class PolicyEvaluation:
    policies: tuple[PolicyMetric, ...]
    mean_f1: float

    def for_policy(self, policy_id: int) -> PolicyMetric:
        for metric in self.policies:
            if metric.policy_id == policy_id:
                return metric
        raise KeyError(policy_id)


@dataclass(frozen=True)
class GlobalCounts:
    """Request-level categories used by the paper's aggregate analysis.

    ``tp`` and ``fn_star`` can overlap for a multi-ID prediction: when the gold
    policy is present alongside a wrong policy, the request contributes to both.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    fn_star: int = 0
    tn: int = 0


@dataclass(frozen=True)
class RequestMetrics:
    request_count: int
    exact_matches: int
    exact_match_accuracy: float
    compliant_count: int
    compliant_correct: int
    compliant_recall: float
    violating_count: int
    violating_detected: int
    violating_request_detection_recall: float
    gold_policy_detected: int
    exact_violated_policy_recall: float
    global_counts: GlobalCounts


@dataclass(frozen=True)
class EvaluationReport:
    policy: PolicyEvaluation
    requests: RequestMetrics


def _normalize_policy_ids(policy_ids: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(policy_ids)
    if not normalized:
        raise ValueError("policy_ids must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in normalized):
        raise TypeError("policy IDs must be integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("policy_ids must be unique")
    if normalized != DEFAULT_POLICY_IDS:
        raise ValueError(
            "the primary metric requires the fixed ordered policy IDs 1 through 9"
        )
    return normalized


def _normalize_label_set(
    values: Iterable[int], *, allowed: frozenset[int], name: str
) -> frozenset[int]:
    try:
        normalized = frozenset(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integer policy IDs") from exc
    if any(isinstance(item, bool) or not isinstance(item, int) for item in normalized):
        raise TypeError(f"{name} must contain only integer policy IDs")
    invalid = normalized - allowed
    if invalid:
        raise ValueError(f"{name} contains unknown policy IDs: {sorted(invalid)}")
    return normalized


def normalize_inputs(
    gold: Sequence[Iterable[int]],
    predictions: Sequence[Iterable[int]],
    *,
    policy_ids: Iterable[int] = DEFAULT_POLICY_IDS,
) -> tuple[tuple[frozenset[int], ...], tuple[frozenset[int], ...], tuple[int, ...]]:
    """Validate and normalize evaluation inputs.

    The released benchmark has zero or one gold policy per request. Predictions
    may contain any subset of the configured policies.
    """

    policies = _normalize_policy_ids(policy_ids)
    allowed = frozenset(policies)
    if len(gold) != len(predictions):
        raise ValueError("gold and predictions must have the same length")

    normalized_gold: list[frozenset[int]] = []
    normalized_predictions: list[frozenset[int]] = []
    for index, (gold_ids, predicted_ids) in enumerate(zip(gold, predictions, strict=True)):
        gold_set = _normalize_label_set(
            gold_ids, allowed=allowed, name=f"gold[{index}]"
        )
        if len(gold_set) > 1:
            raise ValueError(
                f"gold[{index}] has multiple policies; the benchmark contract permits at most one"
            )
        predicted_set = _normalize_label_set(
            predicted_ids, allowed=allowed, name=f"predictions[{index}]"
        )
        normalized_gold.append(gold_set)
        normalized_predictions.append(predicted_set)
    return tuple(normalized_gold), tuple(normalized_predictions), policies


def _policy_evaluation_normalized(
    gold: Sequence[frozenset[int]],
    predictions: Sequence[frozenset[int]],
    policy_ids: tuple[int, ...],
) -> PolicyEvaluation:
    metrics: list[PolicyMetric] = []
    for policy_id in policy_ids:
        tp = fp1 = fp2 = fn = fn_star = support = 0
        for gold_ids, predicted_ids in zip(gold, predictions, strict=True):
            if policy_id in gold_ids:
                support += 1
                if policy_id in predicted_ids:
                    tp += 1
                if not predicted_ids:
                    fn += 1
                elif any(other_id != policy_id for other_id in predicted_ids):
                    # Appendix D permits FN* in addition to TP.
                    fn_star += 1
            elif policy_id in predicted_ids:
                if not gold_ids:
                    fp1 += 1
                else:
                    fp2 += 1

        counts = PolicyCounts(tp=tp, fp1=fp1, fp2=fp2, fn=fn, fn_star=fn_star)
        denominator = (2 * tp) + fp1 + fp2 + fn + fn_star
        f1 = (2 * tp / denominator) if denominator else 0.0
        metrics.append(
            PolicyMetric(policy_id=policy_id, support=support, counts=counts, f1=f1)
        )

    return PolicyEvaluation(
        policies=tuple(metrics),
        mean_f1=sum(metric.f1 for metric in metrics) / len(policy_ids),
    )


def policy_evaluation(
    gold: Sequence[Iterable[int]],
    predictions: Sequence[Iterable[int]],
    *,
    policy_ids: Iterable[int] = DEFAULT_POLICY_IDS,
) -> PolicyEvaluation:
    """Compute all nine Appendix D policy metrics and their fixed mean."""

    normalized_gold, normalized_predictions, policies = normalize_inputs(
        gold, predictions, policy_ids=policy_ids
    )
    return _policy_evaluation_normalized(normalized_gold, normalized_predictions, policies)


def mean_policy_f1(
    gold: Sequence[Iterable[int]],
    predictions: Sequence[Iterable[int]],
    *,
    policy_ids: Iterable[int] = DEFAULT_POLICY_IDS,
) -> float:
    return policy_evaluation(gold, predictions, policy_ids=policy_ids).mean_f1


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _request_metrics_normalized(
    gold: Sequence[frozenset[int]], predictions: Sequence[frozenset[int]]
) -> RequestMetrics:
    exact_matches = compliant_count = compliant_correct = 0
    violating_count = violating_detected = gold_policy_detected = 0
    tp = fp = fn = fn_star = tn = 0

    for gold_ids, predicted_ids in zip(gold, predictions, strict=True):
        if gold_ids == predicted_ids:
            exact_matches += 1

        if not gold_ids:
            compliant_count += 1
            if not predicted_ids:
                compliant_correct += 1
                tn += 1
            else:
                fp += 1
            continue

        violating_count += 1
        if predicted_ids:
            violating_detected += 1
        else:
            fn += 1
        if gold_ids.issubset(predicted_ids):
            gold_policy_detected += 1
            tp += 1
        if predicted_ids and bool(predicted_ids - gold_ids):
            fn_star += 1

    request_count = len(gold)
    return RequestMetrics(
        request_count=request_count,
        exact_matches=exact_matches,
        exact_match_accuracy=_safe_rate(exact_matches, request_count),
        compliant_count=compliant_count,
        compliant_correct=compliant_correct,
        compliant_recall=_safe_rate(compliant_correct, compliant_count),
        violating_count=violating_count,
        violating_detected=violating_detected,
        violating_request_detection_recall=_safe_rate(
            violating_detected, violating_count
        ),
        gold_policy_detected=gold_policy_detected,
        exact_violated_policy_recall=_safe_rate(gold_policy_detected, violating_count),
        global_counts=GlobalCounts(tp=tp, fp=fp, fn=fn, fn_star=fn_star, tn=tn),
    )


def request_metrics(
    gold: Sequence[Iterable[int]],
    predictions: Sequence[Iterable[int]],
    *,
    policy_ids: Iterable[int] = DEFAULT_POLICY_IDS,
) -> RequestMetrics:
    normalized_gold, normalized_predictions, _ = normalize_inputs(
        gold, predictions, policy_ids=policy_ids
    )
    return _request_metrics_normalized(normalized_gold, normalized_predictions)


def evaluate(
    gold: Sequence[Iterable[int]],
    predictions: Sequence[Iterable[int]],
    *,
    policy_ids: Iterable[int] = DEFAULT_POLICY_IDS,
) -> EvaluationReport:
    normalized_gold, normalized_predictions, policies = normalize_inputs(
        gold, predictions, policy_ids=policy_ids
    )
    return EvaluationReport(
        policy=_policy_evaluation_normalized(
            normalized_gold, normalized_predictions, policies
        ),
        requests=_request_metrics_normalized(normalized_gold, normalized_predictions),
    )
