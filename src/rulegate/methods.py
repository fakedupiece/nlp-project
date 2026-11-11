"""Typed method contracts for the frozen RuleGate experiment design.

The Stage 2 baseline prompts are documented reimplementations because the
official sources do not publish author prompt strings. Screening prompts remain
disabled until their own versioned reimplementations pass design-set review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from .schemas import InferenceRequest


class MethodName(str, Enum):
    """Names used in call logs and result artifacts."""

    SINGLE_PROMPTING = "single_prompting"
    FULL_SEQUENTIAL_BOTH = "full_sequential_both"
    PLAIN_SCREENING = "plain_screening"
    ACTION_AWARE_SCREENING = "action_aware_screening"
    REQUEST_PLAN = "request_plan"
    OTHER_AUXILIARY = "other_auxiliary"


InferenceInput = InferenceRequest


@dataclass(frozen=True, slots=True)
class MethodContract:
    name: MethodName
    calls_per_request: str
    prompt_status: str


METHOD_CONTRACTS: tuple[MethodContract, ...] = (
    MethodContract(
        MethodName.SINGLE_PROMPTING,
        "one call containing all nine policies",
        "single-prompting-v2-reimplementation",
    ),
    MethodContract(
        MethodName.FULL_SEQUENTIAL_BOTH,
        "one unchanged verifier call per selected policy",
        "sequential-both-v3-reimplementation",
    ),
    MethodContract(
        MethodName.PLAIN_SCREENING,
        "one screening call, before verification",
        "plain-screening-v1 (design development; freeze before locked test)",
    ),
    MethodContract(
        MethodName.ACTION_AWARE_SCREENING,
        "one screening call, before verification",
        "action-aware-screening-v1 (design development; freeze before locked test)",
    ),
)


class SinglePromptingMethod(Protocol):
    def predict(self, request: InferenceInput) -> Sequence[int]: ...


class FullSequentialBothMethod(Protocol):
    def verify_full(self, request: InferenceInput) -> Sequence[int]: ...


class ScreeningMethod(Protocol):
    def screen(self, request: InferenceInput) -> Sequence[int]: ...
