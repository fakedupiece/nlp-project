"""One unchanged per-policy verifier shared by full and screened conditions."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .methods import InferenceInput, MethodName
from .model_backend import BackendCallError, CallRequest, InstrumentedBackend
from .screening import POLICY_IDS, ScreeningDecision


class VerificationParseError(ValueError):
    pass


class VerificationError(RuntimeError):
    pass


_SINGLE_FENCE = re.compile(r"^```[A-Za-z]*[ \t]*\r?\n?(?P<body>.*?)\r?\n?```$", re.DOTALL)


def strip_single_code_fence(text: str) -> str:
    """Remove one Markdown code fence wrapping the whole reply, if present.

    Parser revision for the v2 prompts: a reply that is exactly one JSON object
    inside one fence (for example a ```json block) is accepted. Any other prose
    around or inside the fence still fails JSON parsing.
    """

    if not isinstance(text, str):
        return text
    stripped = text.strip()
    match = _SINGLE_FENCE.fullmatch(stripped)
    return match.group("body").strip() if match else stripped


def parse_json_verdict(text: str) -> bool:
    try:
        payload = json.loads(strip_single_code_fence(text))
    except (json.JSONDecodeError, TypeError) as exc:
        raise VerificationParseError("verdict is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"violated"}:
        raise VerificationParseError("verdict must contain only 'violated'")
    if not isinstance(payload["violated"], bool):
        raise VerificationParseError("violated must be a boolean")
    return payload["violated"]


PromptBuilder = Callable[[str, int], str]
VerdictParser = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    predicted_policy_ids: tuple[int, ...]
    checked_policy_ids: tuple[int, ...]
    call_ids: tuple[int, ...]


class SequentialBothVerifier:
    """Verifier whose prompt builder/parser are identical in both routes.

    Stage 0/1 callers must inject a documented prompt builder.  This avoids
    presenting a newly invented prompt as the paper's exact template.
    """

    def __init__(
        self,
        *,
        backend: InstrumentedBackend,
        prompt_builder: PromptBuilder,
        prompt_template: str,
        verdict_parser: VerdictParser = parse_json_verdict,
        system_prompt: str | None = None,
    ) -> None:
        self.backend = backend
        self.prompt_builder = prompt_builder
        self.prompt_template = prompt_template
        self.verdict_parser = verdict_parser
        self.system_prompt = system_prompt

    def verify_full(
        self,
        request: InferenceInput,
        *,
        max_attempts: int = 1,
    ) -> VerificationResult:
        return self.verify_candidates(
            request,
            POLICY_IDS,
            max_attempts=max_attempts,
            fallback_used=False,
            fallback_type=None,
        )

    def verify_screened(
        self,
        request: InferenceInput,
        decision: ScreeningDecision,
        *,
        max_attempts: int = 1,
    ) -> VerificationResult:
        return self.verify_candidates(
            request,
            decision.candidate_ids,
            max_attempts=max_attempts,
            fallback_used=decision.fallback_used,
            fallback_type=decision.fallback_type,
        )

    def verify_candidates(
        self,
        request: InferenceInput,
        policy_ids: Sequence[int],
        *,
        max_attempts: int = 1,
        fallback_used: bool = False,
        fallback_type: str | None = None,
    ) -> VerificationResult:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        checked = tuple(policy_ids)
        if len(checked) != len(set(checked)) or any(pid not in POLICY_IDS for pid in checked):
            raise ValueError("policy_ids must be unique values from 1 through 9")

        predicted: list[int] = []
        call_ids: list[int] = []
        for policy_id in checked:
            prompt = self.prompt_builder(request.text, policy_id)
            verdict: bool | None = None
            last_reason = "verification did not run"
            for attempt in range(1, max_attempts + 1):
                try:
                    completed = self.backend.complete(
                        CallRequest(
                            request_id=request.request_id,
                            method=MethodName.FULL_SEQUENTIAL_BOTH,
                            prompt=prompt,
                            prompt_template=self.prompt_template,
                            system_prompt=self.system_prompt,
                            policy_id=policy_id,
                            attempt=attempt,
                            fallback_used=fallback_used,
                            fallback_type=fallback_type,
                        )
                    )
                except BackendCallError as exc:
                    call_ids.append(exc.call_id)
                    last_reason = f"backend_error:{type(exc.cause).__name__}"
                    continue

                call_ids.append(completed.call_id)
                try:
                    verdict = self.verdict_parser(completed.response.text)
                except VerificationParseError as exc:
                    last_reason = f"invalid_verdict:{exc}"
                    self.backend.ledger.mark_parse(
                        completed.call_id,
                        "invalid_verdict",
                        failure_reason=last_reason,
                    )
                    continue
                self.backend.ledger.mark_parse(
                    completed.call_id,
                    "valid_verdict",
                    parsed_output=str(verdict).lower(),
                )
                break

            if verdict is None:
                raise VerificationError(
                    f"policy {policy_id} could not be verified after {max_attempts} "
                    f"attempt(s): {last_reason}"
                )
            if verdict:
                predicted.append(policy_id)

        return VerificationResult(
            predicted_policy_ids=tuple(predicted),
            checked_policy_ids=checked,
            call_ids=tuple(call_ids),
        )
