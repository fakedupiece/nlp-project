"""H2 summaries derived only from observed backend-boundary call records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .model_backend import CallLedger


class H2Status(str, Enum):
    GO = "GO"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class CallSummary:
    request_ids: tuple[str, ...]
    calls_by_request: tuple[tuple[str, int], ...]
    total_calls: int
    mean_calls_per_request: float
    input_tokens: int
    output_tokens: int
    calls_with_unknown_token_counts: int
    elapsed_seconds: float
    failed_calls: int


@dataclass(frozen=True, slots=True)
class H2Comparison:
    full: CallSummary
    screened: CallSummary
    call_reduction: float
    threshold: float
    status: H2Status


def summarize_calls(ledger: CallLedger, request_ids: tuple[str, ...]) -> CallSummary:
    """Summarize every actual attempt associated with the selected requests."""

    if not request_ids:
        raise ValueError("request_ids must not be empty")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_ids must be unique")

    selected = set(request_ids)
    counts = {request_id: 0 for request_id in request_ids}
    input_tokens = output_tokens = failed_calls = unknown_token_calls = 0
    elapsed_seconds = 0.0
    for record in ledger.records:
        if record.request_id not in selected:
            continue
        counts[record.request_id] += 1
        if record.input_tokens is None or record.output_tokens is None:
            unknown_token_calls += 1
        else:
            input_tokens += record.input_tokens
            output_tokens += record.output_tokens
        elapsed_seconds += record.elapsed_seconds
        failed_calls += int(not record.success)

    total = sum(counts.values())
    return CallSummary(
        request_ids=request_ids,
        calls_by_request=tuple((request_id, counts[request_id]) for request_id in request_ids),
        total_calls=total,
        mean_calls_per_request=total / len(request_ids),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        calls_with_unknown_token_counts=unknown_token_calls,
        elapsed_seconds=elapsed_seconds,
        failed_calls=failed_calls,
    )


def compare_call_reduction(
    *,
    full_ledger: CallLedger,
    screened_ledger: CallLedger,
    request_ids: tuple[str, ...],
    threshold: float = 0.50,
) -> H2Comparison:
    """Apply H2 to measured calls on the same explicitly listed requests."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    selected = set(request_ids)
    for condition, ledger in (("full", full_ledger), ("screened", screened_ledger)):
        unexpected = sorted(
            {record.request_id for record in ledger.records if record.request_id not in selected}
        )
        if unexpected:
            raise ValueError(
                f"{condition} ledger contains unselected request IDs: {unexpected!r}"
            )
    full = summarize_calls(full_ledger, request_ids)
    screened = summarize_calls(screened_ledger, request_ids)
    if full.total_calls == 0:
        raise ValueError("full sequential baseline has no measured calls")
    if any(count == 0 for _, count in full.calls_by_request):
        raise ValueError("full sequential baseline must contain calls for every request")
    if any(count == 0 for _, count in screened.calls_by_request):
        raise ValueError("screened condition must contain calls for every request")

    reduction = 1.0 - screened.mean_calls_per_request / full.mean_calls_per_request
    return H2Comparison(
        full=full,
        screened=screened,
        call_reduction=reduction,
        threshold=threshold,
        status=H2Status.GO if reduction >= threshold else H2Status.FAIL,
    )
