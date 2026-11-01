"""Shared LLM backend boundary with complete attempted-call instrumentation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol, TypeAlias

from .methods import MethodName


@dataclass(frozen=True, slots=True)
class ModelSettings:
    model_id: str
    revision: str | None = None
    quantization: str | None = None
    temperature: float = 0.0
    seed: int | None = None
    decoding_settings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CallRequest:
    request_id: str
    method: MethodName | str
    prompt: str
    prompt_template: str
    system_prompt: str | None = None
    policy_id: int | None = None
    attempt: int = 1
    fallback_used: bool = False
    fallback_type: str | None = None
    raw_response_path: str | None = None


@dataclass(frozen=True, slots=True)
class RawBackendResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class RawBackend(Protocol):
    def complete(self, request: CallRequest) -> RawBackendResponse: ...


@dataclass(slots=True)
class CallRecord:
    call_id: int
    request_id: str
    method: str
    policy_id: int | None
    prompt_template: str
    prompt_hash: str
    model_id: str
    model_revision: str | None
    quantization: str | None
    temperature: float
    seed: int | None
    decoding_settings: tuple[tuple[str, str], ...]
    attempt: int
    fallback_used: bool
    fallback_type: str | None
    success: bool
    error_type: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_seconds: float
    parse_status: str
    parsed_output: str | None
    failure_reason: str | None
    raw_response_path: str | None


class CallLedger:
    """Append-only call records plus explicit post-call parse annotations."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(self._records)

    def append(self, record: CallRecord) -> int:
        expected_id = len(self._records) + 1
        if record.call_id != expected_id:
            raise ValueError(f"call_id must be {expected_id}, got {record.call_id}")
        self._records.append(record)
        return record.call_id

    def next_call_id(self) -> int:
        return len(self._records) + 1

    def write_jsonl(self, path: str | Path) -> None:
        """Persist the complete ledger without embedding raw model responses."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(record), sort_keys=True) for record in self._records]
        destination.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    def mark_parse(
        self,
        call_id: int,
        status: str,
        parsed_output: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        if not status:
            raise ValueError("parse status must be non-empty")
        try:
            record = self._records[call_id - 1]
        except IndexError as exc:
            raise KeyError(f"unknown call_id {call_id}") from exc
        if record.call_id != call_id:
            raise KeyError(f"unknown call_id {call_id}")
        record.parse_status = status
        record.parsed_output = parsed_output
        record.failure_reason = failure_reason

    def mark_fallback(self, call_id: int, fallback_type: str) -> None:
        if not fallback_type:
            raise ValueError("fallback_type must be non-empty")
        try:
            record = self._records[call_id - 1]
        except IndexError as exc:
            raise KeyError(f"unknown call_id {call_id}") from exc
        if record.call_id != call_id:
            raise KeyError(f"unknown call_id {call_id}")
        record.fallback_used = True
        record.fallback_type = fallback_type


@dataclass(frozen=True, slots=True)
class CompletedCall:
    call_id: int
    response: RawBackendResponse


class BackendCallError(RuntimeError):
    """Backend failure carrying the ledger ID of the failed attempt."""

    def __init__(self, call_id: int, cause: Exception) -> None:
        super().__init__(f"backend call {call_id} failed: {type(cause).__name__}: {cause}")
        self.call_id = call_id
        self.cause = cause


class InstrumentedBackend:
    """The only model-call boundary used by runners.

    A record is appended for every attempt, including exceptions.  Downstream
    parsers annotate the resulting record rather than estimating calls from the
    expected control flow.
    """

    def __init__(
        self,
        backend: RawBackend,
        settings: ModelSettings,
        ledger: CallLedger | None = None,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.ledger = ledger if ledger is not None else CallLedger()

    def complete(self, request: CallRequest) -> CompletedCall:
        call_id = self.ledger.next_call_id()
        started = perf_counter()
        try:
            response = self.backend.complete(request)
        except Exception as exc:
            elapsed = perf_counter() - started
            self.ledger.append(
                self._record(
                    call_id=call_id,
                    request=request,
                    success=False,
                    error_type=type(exc).__name__,
                    input_tokens=None,
                    output_tokens=None,
                    elapsed=elapsed,
                    parse_status="backend_error",
                )
            )
            raise BackendCallError(call_id, exc) from exc

        elapsed = perf_counter() - started
        self.ledger.append(
            self._record(
                call_id=call_id,
                request=request,
                success=True,
                error_type=None,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                elapsed=elapsed,
                parse_status="not_parsed",
            )
        )
        return CompletedCall(call_id=call_id, response=response)

    def _record(
        self,
        *,
        call_id: int,
        request: CallRequest,
        success: bool,
        error_type: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        elapsed: float,
        parse_status: str,
    ) -> CallRecord:
        method = request.method.value if isinstance(request.method, MethodName) else request.method
        return CallRecord(
            call_id=call_id,
            request_id=request.request_id,
            method=method,
            policy_id=request.policy_id,
            prompt_template=request.prompt_template,
            prompt_hash=sha256(
                json.dumps(
                    {"system": request.system_prompt, "user": request.prompt},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            model_id=self.settings.model_id,
            model_revision=self.settings.revision,
            quantization=self.settings.quantization,
            temperature=self.settings.temperature,
            seed=self.settings.seed,
            decoding_settings=self.settings.decoding_settings,
            attempt=request.attempt,
            fallback_used=request.fallback_used,
            fallback_type=request.fallback_type,
            success=success,
            error_type=error_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed,
            parse_status=parse_status,
            parsed_output=None,
            failure_reason=None,
            raw_response_path=request.raw_response_path,
        )


FakeOutcome: TypeAlias = (
    RawBackendResponse
    | Exception
    | Callable[[CallRequest], RawBackendResponse]
)


class FakeBackend:
    """Deterministic queued backend for CPU-only tests."""

    def __init__(self, outcomes: Iterable[FakeOutcome]) -> None:
        self._outcomes = deque(outcomes)
        self.requests: list[CallRequest] = []

    @property
    def remaining(self) -> int:
        return len(self._outcomes)

    def complete(self, request: CallRequest) -> RawBackendResponse:
        self.requests.append(request)
        if not self._outcomes:
            raise AssertionError("fake backend outcome queue exhausted")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            outcome = outcome(request)
        if not isinstance(outcome, RawBackendResponse):
            raise TypeError("fake outcome must return RawBackendResponse")
        return outcome
