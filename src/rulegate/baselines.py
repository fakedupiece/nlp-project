"""Stage 2 baseline runners sharing the instrumented backend boundary."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .methods import InferenceInput, MethodName
from .model_backend import BackendCallError, CallRequest, InstrumentedBackend
from .prompts import (
    SEQUENTIAL_BOTH_SYSTEM_PROMPT,
    SEQUENTIAL_BOTH_TEMPLATE_ID,
    SINGLE_PROMPT_SYSTEM_PROMPT,
    SINGLE_PROMPT_TEMPLATE_ID,
    make_sequential_both_prompt_builder,
    render_single_prompt,
)
from .schemas import POLICY_IDS, Policy
from .jsonreply import strip_single_code_fence
from .verification import SequentialBothVerifier


class SinglePromptingParseError(ValueError):
    """Raised when a Single Prompting response violates the frozen JSON schema."""


class SinglePromptingError(RuntimeError):
    """Raised when all Single Prompting attempts fail or remain invalid."""


def parse_single_prompt_response(text: str) -> tuple[int, ...]:
    """Parse exactly ``{"violated_policy_ids": [...]}`` and normalize ID order."""

    try:
        payload = json.loads(strip_single_code_fence(text))
    except (json.JSONDecodeError, TypeError) as exc:
        raise SinglePromptingParseError("response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"violated_policy_ids"}:
        raise SinglePromptingParseError(
            "response must contain only 'violated_policy_ids'"
        )
    values = payload["violated_policy_ids"]
    if not isinstance(values, list):
        raise SinglePromptingParseError("violated_policy_ids must be a list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise SinglePromptingParseError("policy IDs must be integers")
    if len(values) != len(set(values)):
        raise SinglePromptingParseError("policy IDs must be unique")
    invalid = sorted(set(values) - set(POLICY_IDS))
    if invalid:
        raise SinglePromptingParseError(
            f"policy IDs must be in 1..9; got {invalid!r}"
        )
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class SinglePromptingResult:
    predicted_policy_ids: tuple[int, ...]
    call_ids: tuple[int, ...]


class SinglePromptingBaseline:
    """One-call all-policy baseline with explicit retry and parse accounting."""

    def __init__(
        self,
        *,
        backend: InstrumentedBackend,
        policies: Sequence[Policy],
    ) -> None:
        self.backend = backend
        self.policies = tuple(policies)
        # Render-time validation ensures exactly the official ordered policies.
        render_single_prompt("validation request", self.policies)

    def predict(
        self,
        request: InferenceInput,
        *,
        max_attempts: int = 1,
    ) -> tuple[int, ...]:
        return self.predict_with_trace(
            request,
            max_attempts=max_attempts,
        ).predicted_policy_ids

    def predict_with_trace(
        self,
        request: InferenceInput,
        *,
        max_attempts: int = 1,
    ) -> SinglePromptingResult:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        prompt = render_single_prompt(request.text, self.policies)
        call_ids: list[int] = []
        last_reason = "single prompting did not run"

        for attempt in range(1, max_attempts + 1):
            try:
                completed = self.backend.complete(
                    CallRequest(
                        request_id=request.request_id,
                        method=MethodName.SINGLE_PROMPTING,
                        prompt=prompt,
                        prompt_template=SINGLE_PROMPT_TEMPLATE_ID,
                        system_prompt=SINGLE_PROMPT_SYSTEM_PROMPT,
                        attempt=attempt,
                    )
                )
            except BackendCallError as exc:
                call_ids.append(exc.call_id)
                last_reason = f"backend_error:{type(exc.cause).__name__}"
                continue

            call_ids.append(completed.call_id)
            try:
                policy_ids = parse_single_prompt_response(completed.response.text)
            except SinglePromptingParseError as exc:
                last_reason = f"invalid_single_response:{exc}"
                self.backend.ledger.mark_parse(
                    completed.call_id,
                    "invalid_single_response",
                    failure_reason=last_reason,
                )
                continue

            self.backend.ledger.mark_parse(
                completed.call_id,
                "valid_single_response",
                parsed_output=json.dumps(list(policy_ids), separators=(",", ":")),
            )
            return SinglePromptingResult(
                predicted_policy_ids=policy_ids,
                call_ids=tuple(call_ids),
            )

        raise SinglePromptingError(
            f"request could not be classified after {max_attempts} attempt(s): "
            f"{last_reason}"
        )


def make_sequential_both_verifier(
    *,
    backend: InstrumentedBackend,
    policies: Sequence[Policy],
) -> SequentialBothVerifier:
    """Create the one shared verifier used by full and screened conditions."""

    return SequentialBothVerifier(
        backend=backend,
        prompt_builder=make_sequential_both_prompt_builder(policies),
        prompt_template=SEQUENTIAL_BOTH_TEMPLATE_ID,
        system_prompt=SEQUENTIAL_BOTH_SYSTEM_PROMPT,
    )
