"""Strict conservative-screen routing and safe full-verification fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from .methods import InferenceInput, MethodName
from .model_backend import BackendCallError, CallRequest, InstrumentedBackend


POLICY_IDS = tuple(range(1, 10))


class ScreenLabel(str, Enum):
    POSSIBLE = "POSSIBLE"
    UNCERTAIN = "UNCERTAIN"
    EXCLUDE = "EXCLUDE"


class ScreeningParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PolicyRoute:
    policy_id: int
    label: ScreenLabel


@dataclass(frozen=True, slots=True)
class ScreeningDecision:
    routes: tuple[PolicyRoute, ...]
    candidate_ids: tuple[int, ...]
    valid: bool
    fallback_used: bool
    fallback_type: str | None
    screening_failure_reason: str | None
    call_ids: tuple[int, ...]


def parse_screen_response(text: str) -> tuple[PolicyRoute, ...]:
    """Parse the frozen Stage 0/1 JSON shape without accepting partial output."""

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ScreeningParseError("response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"labels"}:
        raise ScreeningParseError("top-level object must contain only 'labels'")
    labels = payload["labels"]
    if not isinstance(labels, list) or len(labels) != len(POLICY_IDS):
        raise ScreeningParseError("labels must contain exactly nine entries")

    routes: list[PolicyRoute] = []
    seen: set[int] = set()
    for item in labels:
        if not isinstance(item, dict) or set(item) != {"policy_id", "label"}:
            raise ScreeningParseError("each label entry must contain policy_id and label")
        policy_id = item["policy_id"]
        if isinstance(policy_id, bool) or not isinstance(policy_id, int):
            raise ScreeningParseError("policy_id must be an integer")
        if policy_id not in POLICY_IDS:
            raise ScreeningParseError(f"unexpected policy_id {policy_id}")
        if policy_id in seen:
            raise ScreeningParseError(f"duplicate policy_id {policy_id}")
        try:
            label = ScreenLabel(item["label"])
        except (ValueError, TypeError) as exc:
            raise ScreeningParseError(f"invalid label for policy {policy_id}") from exc
        routes.append(PolicyRoute(policy_id, label))
        seen.add(policy_id)

    if seen != set(POLICY_IDS):
        raise ScreeningParseError("missing or extra policy IDs")
    return tuple(sorted(routes, key=lambda route: route.policy_id))


def run_screening(
    *,
    backend: InstrumentedBackend,
    request: InferenceInput,
    prompt: str,
    prompt_template: str,
    method: MethodName,
    max_attempts: int = 1,
) -> ScreeningDecision:
    """Call a plain/action-aware screener, retry if configured, then fail safe."""

    if method not in {MethodName.PLAIN_SCREENING, MethodName.ACTION_AWARE_SCREENING}:
        raise ValueError("method must be a screening method")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    call_ids: list[int] = []
    last_reason = "screening did not run"
    for attempt in range(1, max_attempts + 1):
        try:
            completed = backend.complete(
                CallRequest(
                    request_id=request.request_id,
                    method=method,
                    prompt=prompt,
                    prompt_template=prompt_template,
                    attempt=attempt,
                )
            )
        except BackendCallError as exc:
            call_ids.append(exc.call_id)
            last_reason = f"backend_error:{type(exc.cause).__name__}"
            continue

        call_ids.append(completed.call_id)
        try:
            routes = parse_screen_response(completed.response.text)
        except ScreeningParseError as exc:
            last_reason = f"invalid_screen:{exc}"
            backend.ledger.mark_parse(
                completed.call_id,
                "invalid_screen",
                failure_reason=last_reason,
            )
            continue

        backend.ledger.mark_parse(
            completed.call_id,
            "valid_screen",
            parsed_output=";".join(
                f"{route.policy_id}:{route.label.value}" for route in routes
            ),
        )
        candidates = tuple(
            route.policy_id
            for route in routes
            if route.label in {ScreenLabel.POSSIBLE, ScreenLabel.UNCERTAIN}
        )
        return ScreeningDecision(
            routes=routes,
            candidate_ids=candidates,
            valid=True,
            fallback_used=False,
            fallback_type=None,
            screening_failure_reason=None,
            call_ids=tuple(call_ids),
        )

    fallback_type = "full_nine_policy_verification"
    for call_id in call_ids:
        backend.ledger.mark_fallback(call_id, fallback_type)
    return ScreeningDecision(
        routes=(),
        candidate_ids=POLICY_IDS,
        valid=False,
        fallback_used=True,
        fallback_type=fallback_type,
        screening_failure_reason=last_reason,
        call_ids=tuple(call_ids),
    )
