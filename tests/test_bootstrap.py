import pytest

from rulegate.bootstrap import paired_bootstrap_delta_f1
from rulegate.decisions import DecisionStatus, decide_h1, decide_h1_bootstrap, decide_h2


def test_synthetic_inconclusive_bootstrap_is_deterministic() -> None:
    gold = [[], [1], [1], [2], [2], []]
    full = [[], [1], [1], [2], [2], []]
    ours = [[], [1], [], [2], [7], []]

    first = paired_bootstrap_delta_f1(
        gold, ours, full, seed=2026
    )
    second = paired_bootstrap_delta_f1(
        gold, ours, full, seed=2026
    )

    assert first == second
    assert first.point_estimate < 0.0
    assert first.confidence_low <= first.confidence_high
    assert first.request_count == 6
    assert decide_h1_bootstrap(first).status is DecisionStatus.INCONCLUSIVE


def test_default_ten_thousand_resamples_and_go_fixture() -> None:
    result = paired_bootstrap_delta_f1([[1]], [[1]], [[1]])

    assert result.n_resamples == 10_000
    assert result.point_estimate == 0.0
    assert result.confidence_low == 0.0
    assert result.confidence_high == 0.0
    assert decide_h1_bootstrap(result).status is DecisionStatus.GO


def test_synthetic_fail_fixture() -> None:
    result = paired_bootstrap_delta_f1([[1]], [[]], [[1]], seed=7)

    assert result.point_estimate == pytest.approx(-1.0 / 9.0)
    assert result.confidence_low == pytest.approx(-1.0 / 9.0)
    assert result.confidence_high == pytest.approx(-1.0 / 9.0)
    assert decide_h1_bootstrap(result).status is DecisionStatus.FAIL


def test_synthetic_h1_decision_go_inconclusive_and_fail() -> None:
    go = decide_h1(0.0, -0.049, 0.02)
    inconclusive = decide_h1(-0.04, -0.08, 0.01)
    fail = decide_h1(-0.08, -0.12, -0.05)

    assert go.status is DecisionStatus.GO
    assert inconclusive.status is DecisionStatus.INCONCLUSIVE
    assert fail.status is DecisionStatus.FAIL


def test_h1_boundary_is_strict() -> None:
    decision = decide_h1(-0.01, -0.05, 0.01)

    assert decision.status is DecisionStatus.INCONCLUSIVE


def test_h2_threshold_is_inclusive() -> None:
    assert decide_h2(0.50).status is DecisionStatus.GO
    assert decide_h2(0.499).status is DecisionStatus.FAIL


def test_bootstrap_rejects_empty_or_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="at least one request"):
        paired_bootstrap_delta_f1([], [], [], n_resamples=10)
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_delta_f1([[1]], [], [[1]], n_resamples=10)
