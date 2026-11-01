import math

import pytest

from rulegate.metrics import evaluate, mean_policy_f1, policy_evaluation


def test_correct_policy_uses_fixed_nine_policy_mean() -> None:
    report = evaluate([[2]], [[2]])

    policy_2 = report.policy.for_policy(2)
    assert policy_2.support == 1
    assert policy_2.counts.tp == 1
    assert policy_2.f1 == 1.0
    assert report.policy.mean_f1 == pytest.approx(1.0 / 9.0)
    assert report.requests.exact_match_accuracy == 1.0
    assert report.requests.global_counts.tp == 1


def test_compliant_false_positive_is_fp1() -> None:
    report = evaluate([[]], [[2]])

    policy_2 = report.policy.for_policy(2)
    assert policy_2.counts.fp1 == 1
    assert policy_2.counts.fp2 == 0
    assert report.requests.compliant_recall == 0.0
    assert report.requests.global_counts.fp == 1


def test_empty_prediction_for_violation_is_fn() -> None:
    report = evaluate([[2]], [[]])

    assert report.policy.for_policy(2).counts.fn == 1
    assert report.policy.for_policy(2).counts.fn_star == 0
    assert report.requests.global_counts.fn == 1
    assert report.requests.violating_request_detection_recall == 0.0


def test_wrong_policy_is_fn_star_for_gold_and_fp2_for_wrong_policy() -> None:
    report = evaluate([[2]], [[7]])

    assert report.policy.for_policy(2).counts.fn_star == 1
    assert report.policy.for_policy(2).counts.fn == 0
    assert report.policy.for_policy(7).counts.fp2 == 1
    assert report.requests.global_counts.fn_star == 1
    assert report.requests.global_counts.tp == 0
    assert report.requests.violating_request_detection_recall == 1.0
    assert report.requests.exact_violated_policy_recall == 0.0


def test_multi_id_prediction_counts_tp_and_fn_star_for_gold_policy() -> None:
    report = evaluate([[2]], [[2, 7]])

    policy_2 = report.policy.for_policy(2)
    assert policy_2.counts.tp == 1
    assert policy_2.counts.fn_star == 1
    assert policy_2.f1 == pytest.approx(2.0 / 3.0)
    assert report.policy.for_policy(7).counts.fp2 == 1
    assert report.requests.global_counts.tp == 1
    assert report.requests.global_counts.fn_star == 1
    assert report.requests.exact_match_accuracy == 0.0
    assert report.requests.exact_violated_policy_recall == 1.0


def test_all_zero_denominators_return_zero() -> None:
    result = policy_evaluation([[]], [[]])

    assert result.mean_f1 == 0.0
    assert all(metric.f1 == 0.0 for metric in result.policies)


def test_mixed_request_level_metrics() -> None:
    report = evaluate(
        [[], [], [1], [2], [3]],
        [[], [4], [1], [7], []],
    )

    requests = report.requests
    assert requests.request_count == 5
    assert requests.exact_matches == 2
    assert requests.exact_match_accuracy == pytest.approx(0.4)
    assert requests.compliant_recall == pytest.approx(0.5)
    assert requests.violating_request_detection_recall == pytest.approx(2.0 / 3.0)
    assert requests.exact_violated_policy_recall == pytest.approx(1.0 / 3.0)
    assert requests.global_counts.tp == 1
    assert requests.global_counts.fp == 1
    assert requests.global_counts.fn == 1
    assert requests.global_counts.fn_star == 1
    assert requests.global_counts.tn == 1


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        mean_policy_f1([[1]], [])
    with pytest.raises(ValueError, match="unknown policy"):
        mean_policy_f1([[10]], [[1]])
    with pytest.raises(ValueError, match="multiple policies"):
        mean_policy_f1([[1, 2]], [[1]])
    with pytest.raises(TypeError, match="integer"):
        mean_policy_f1([[True]], [[1]])
    with pytest.raises(ValueError, match="fixed ordered policy IDs"):
        mean_policy_f1([[1]], [[1]], policy_ids=(1,))
