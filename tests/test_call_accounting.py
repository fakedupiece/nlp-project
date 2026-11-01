import json
from pathlib import Path

import pytest

from rulegate.accounting import H2Status, compare_call_reduction, summarize_calls
from rulegate.methods import InferenceInput, MethodName
from rulegate.model_backend import (
    CallLedger,
    CallRequest,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
)
from rulegate.screening import POLICY_IDS, run_screening
from rulegate.verification import SequentialBothVerifier


def _settings():
    return ModelSettings(
        model_id="fake/model",
        revision="test-revision",
        quantization="none",
        temperature=0.0,
        seed=123,
        decoding_settings=(("max_new_tokens", "64"),),
    )


def _screen_json(candidates):
    return json.dumps(
        {
            "labels": [
                {
                    "policy_id": policy_id,
                    "label": "POSSIBLE" if policy_id in candidates else "EXCLUDE",
                }
                for policy_id in POLICY_IDS
            ]
        }
    )


def _verifier(backend):
    return SequentialBothVerifier(
        backend=backend,
        prompt_builder=lambda text, policy_id: f"policy={policy_id}\n{text}",
        prompt_template="synthetic-sequential-both",
    )


def test_h2_uses_measured_calls_and_reports_tokens_separately():
    request_ids = ("opaque-a", "opaque-b")
    full_raw = FakeBackend(
        [RawBackendResponse('{"violated": false}', 11, 2) for _ in range(18)]
    )
    full = InstrumentedBackend(full_raw, _settings(), CallLedger())
    full_verifier = _verifier(full)
    for request_id in request_ids:
        full_verifier.verify_full(InferenceInput(request_id, "synthetic"))

    screened_outcomes = []
    for _ in request_ids:
        screened_outcomes.append(RawBackendResponse(_screen_json((1, 2, 3)), 90, 20))
        screened_outcomes.extend(
            RawBackendResponse('{"violated": false}', 11, 2) for _ in range(3)
        )
    screened = InstrumentedBackend(FakeBackend(screened_outcomes), _settings(), CallLedger())
    screened_verifier = _verifier(screened)
    for request_id in request_ids:
        request = InferenceInput(request_id, "synthetic")
        decision = run_screening(
            backend=screened,
            request=request,
            prompt="screen all policies",
            prompt_template="synthetic-action-screen",
            method=MethodName.ACTION_AWARE_SCREENING,
        )
        screened_verifier.verify_screened(request, decision)

    result = compare_call_reduction(
        full_ledger=full.ledger,
        screened_ledger=screened.ledger,
        request_ids=request_ids,
    )
    assert result.full.total_calls == 18
    assert result.screened.total_calls == 8
    assert result.call_reduction == pytest.approx(1 - 4 / 9)
    assert result.status is H2Status.GO
    assert result.screened.input_tokens == 2 * (90 + 3 * 11)
    assert result.screened.output_tokens == 2 * (20 + 3 * 2)
    verifier_records = [
        record for record in screened.ledger.records if record.policy_id is not None
    ]
    assert all(record.parsed_output == "false" for record in verifier_records)


def test_backend_errors_and_retries_are_both_counted():
    backend = InstrumentedBackend(
        FakeBackend(
            [
                TimeoutError("synthetic timeout"),
                RawBackendResponse(_screen_json((2,)), input_tokens=10, output_tokens=2),
            ]
        ),
        _settings(),
        CallLedger(),
    )
    decision = run_screening(
        backend=backend,
        request=InferenceInput("opaque-retry", "synthetic"),
        prompt="screen",
        prompt_template="synthetic-screen",
        method=MethodName.PLAIN_SCREENING,
        max_attempts=2,
    )

    summary = summarize_calls(backend.ledger, ("opaque-retry",))
    assert decision.valid
    assert summary.total_calls == 2
    assert summary.failed_calls == 1
    assert summary.calls_with_unknown_token_counts == 1
    assert [record.attempt for record in backend.ledger.records] == [1, 2]
    assert backend.ledger.records[0].parse_status == "backend_error"
    assert backend.ledger.records[1].parsed_output is not None


def test_ledger_jsonl_preserves_auditable_fields_without_raw_response(
    tmp_path: Path,
):
    backend = InstrumentedBackend(
        FakeBackend([RawBackendResponse("sensitive synthetic response", 5, 2)]),
        _settings(),
        CallLedger(),
    )
    completed = backend.complete(
        CallRequest(
            request_id="opaque-log",
            method=MethodName.OTHER_AUXILIARY,
            prompt="synthetic prompt",
            prompt_template="synthetic-v1",
        )
    )
    backend.ledger.mark_parse(completed.call_id, "valid", parsed_output="ok")
    path = tmp_path / "ledger.jsonl"

    backend.ledger.write_jsonl(path)

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["request_id"] == "opaque-log"
    assert row["parsed_output"] == "ok"
    assert row["input_tokens"] == 5
    assert "sensitive synthetic response" not in path.read_text(encoding="utf-8")


def test_auxiliary_calls_are_included_without_formula_assumptions():
    backend = InstrumentedBackend(
        FakeBackend([RawBackendResponse("plan", 5, 3)]), _settings(), CallLedger()
    )
    completed = backend.complete(
        CallRequest(
            request_id="opaque-plan",
            method=MethodName.REQUEST_PLAN,
            prompt="make a plan",
            prompt_template="synthetic-plan",
        )
    )
    backend.ledger.mark_parse(completed.call_id, "valid_plan")

    summary = summarize_calls(backend.ledger, ("opaque-plan",))
    assert summary.total_calls == 1
    assert backend.ledger.records[0].method == "request_plan"
    assert len(backend.ledger.records[0].prompt_hash) == 64


def test_success_without_provider_usage_keeps_token_counts_unknown():
    backend = InstrumentedBackend(
        FakeBackend([RawBackendResponse("ok")]), _settings(), CallLedger()
    )
    backend.complete(
        CallRequest(
            request_id="opaque-no-usage",
            method=MethodName.OTHER_AUXILIARY,
            prompt="call",
            prompt_template="synthetic",
        )
    )

    summary = summarize_calls(backend.ledger, ("opaque-no-usage",))
    assert summary.total_calls == 1
    assert summary.input_tokens == 0
    assert summary.output_tokens == 0
    assert summary.calls_with_unknown_token_counts == 1


def test_h2_fails_below_half_reduction():
    request_ids = ("opaque",)
    full = InstrumentedBackend(
        FakeBackend([RawBackendResponse("ok") for _ in range(9)]), _settings(), CallLedger()
    )
    screened = InstrumentedBackend(
        FakeBackend([RawBackendResponse("ok") for _ in range(6)]), _settings(), CallLedger()
    )
    for backend, count in ((full, 9), (screened, 6)):
        for index in range(count):
            call = backend.complete(
                CallRequest(
                    request_id="opaque",
                    method=MethodName.OTHER_AUXILIARY,
                    prompt=f"call {index}",
                    prompt_template="synthetic",
                )
            )
            backend.ledger.mark_parse(call.call_id, "valid")

    result = compare_call_reduction(
        full_ledger=full.ledger,
        screened_ledger=screened.ledger,
        request_ids=request_ids,
    )
    assert result.call_reduction == pytest.approx(1 / 3)
    assert result.status is H2Status.FAIL


def test_h2_rejects_unselected_calls_in_condition_ledgers():
    full = InstrumentedBackend(
        FakeBackend([RawBackendResponse("ok"), RawBackendResponse("stray")]),
        _settings(),
        CallLedger(),
    )
    screened = InstrumentedBackend(
        FakeBackend([RawBackendResponse("ok")]), _settings(), CallLedger()
    )
    for backend, request_id in (
        (full, "opaque"),
        (full, "unselected"),
        (screened, "opaque"),
    ):
        backend.complete(
            CallRequest(
                request_id=request_id,
                method=MethodName.OTHER_AUXILIARY,
                prompt="call",
                prompt_template="synthetic",
            )
        )

    with pytest.raises(ValueError, match="unselected request IDs"):
        compare_call_reduction(
            full_ledger=full.ledger,
            screened_ledger=screened.ledger,
            request_ids=("opaque",),
        )
