"""Predeclared Stage 0/1 hypothesis decision rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .bootstrap import PairedBootstrapResult


class DecisionStatus(str, Enum):
    GO = "GO"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAIL = "FAIL"


@dataclass(frozen=True)
class H1Decision:
    status: DecisionStatus
    point_estimate: float
    confidence_low: float
    confidence_high: float
    noninferiority_margin: float


@dataclass(frozen=True)
class H2Decision:
    status: DecisionStatus
    call_reduction: float
    required_reduction: float


def decide_h1(
    point_estimate: float,
    confidence_low: float,
    confidence_high: float,
    *,
    noninferiority_margin: float = 0.05,
) -> H1Decision:
    """Apply the corrected H1 rule: CI lower must be strictly above -0.05."""

    values = (point_estimate, confidence_low, confidence_high, noninferiority_margin)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("H1 inputs must be finite")
    if noninferiority_margin < 0.0:
        raise ValueError("noninferiority_margin must be nonnegative")
    if confidence_low > confidence_high:
        raise ValueError("confidence_low must not exceed confidence_high")

    boundary = -noninferiority_margin
    if confidence_low > boundary:
        status = DecisionStatus.GO
    elif confidence_high <= boundary:
        status = DecisionStatus.FAIL
    else:
        status = DecisionStatus.INCONCLUSIVE
    return H1Decision(
        status=status,
        point_estimate=point_estimate,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        noninferiority_margin=noninferiority_margin,
    )


def decide_h1_bootstrap(
    result: PairedBootstrapResult, *, noninferiority_margin: float = 0.05
) -> H1Decision:
    return decide_h1(
        result.point_estimate,
        result.confidence_low,
        result.confidence_high,
        noninferiority_margin=noninferiority_margin,
    )


def decide_h2(
    call_reduction: float, *, required_reduction: float = 0.50
) -> H2Decision:
    if not math.isfinite(call_reduction):
        raise ValueError("call_reduction must be finite")
    if not 0.0 <= required_reduction <= 1.0:
        raise ValueError("required_reduction must be between zero and one")
    status = (
        DecisionStatus.GO
        if call_reduction >= required_reduction
        else DecisionStatus.FAIL
    )
    return H2Decision(
        status=status,
        call_reduction=call_reduction,
        required_reduction=required_reduction,
    )
