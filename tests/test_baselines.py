import json

import pytest

from rulegate.baselines import (
    SinglePromptingBaseline,
    SinglePromptingError,
    SinglePromptingParseError,
    make_sequential_both_verifier,
    parse_single_prompt_response,
)
from rulegate.methods import InferenceInput, MethodName
from rulegate.model_backend import (
    CallLedger,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
)
from rulegate.schemas import Policy


def _policies():
    return tuple(Policy(i, f"Synthetic policy {i}") for i in range(1, 10))


def _backend(outcomes):
    return InstrumentedBackend(
        FakeBackend(outcomes),
        ModelSettings(model_id="fake/model", revision="synthetic", seed=7),
        CallLedger(),
    )


def test_single_response_parser_accepts_zero_or_multiple_policy_ids():
    assert parse_single_prompt_response('{"violated_policy_ids": []}') == ()
    assert parse_single_prompt_response(
        '{"violated_policy_ids": [7,2]}'
    ) == (2, 7)


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        '{"violated_policy_ids": 2}',
        '{"violated_policy_ids": [2,2]}',
        '{"violated_policy_ids": [true]}',
        '{"violated_policy_ids": [10]}',
        '{"violated_policy_ids": [], "extra": 1}',
    ],
)
def test_single_response_parser_is_strict(response):
    with pytest.raises(SinglePromptingParseError):
        parse_single_prompt_response(response)


def test_single_baseline_retries_invalid_output_and_logs_actual_calls():
    backend = _backend(
        [
            RawBackendResponse("invalid", input_tokens=10, output_tokens=1),
            RawBackendResponse(
                json.dumps({"violated_policy_ids": [7, 2]}),
                input_tokens=10,
                output_tokens=4,
            ),
        ]
    )
    baseline = SinglePromptingBaseline(backend=backend, policies=_policies())

    result = baseline.predict_with_trace(
        InferenceInput("opaque", "synthetic request"),
        max_attempts=2,
    )

    assert result.predicted_policy_ids == (2, 7)
    assert result.call_ids == (1, 2)
    assert [record.attempt for record in backend.ledger.records] == [1, 2]
    assert backend.ledger.records[0].parse_status == "invalid_single_response"
    assert backend.ledger.records[1].parse_status == "valid_single_response"
    assert backend.ledger.records[1].parsed_output == "[2,7]"
    assert all(
        record.method == MethodName.SINGLE_PROMPTING.value
        for record in backend.ledger.records
    )


def test_single_baseline_raises_after_backend_and_parse_failures():
    backend = _backend([TimeoutError("synthetic"), RawBackendResponse("bad")])
    baseline = SinglePromptingBaseline(backend=backend, policies=_policies())

    with pytest.raises(SinglePromptingError, match="after 2 attempt"):
        baseline.predict(
            InferenceInput("opaque", "synthetic request"),
            max_attempts=2,
        )
    assert len(backend.ledger.records) == 2
    assert backend.ledger.records[0].parse_status == "backend_error"
    assert backend.ledger.records[1].failure_reason.startswith(
        "invalid_single_response:"
    )


def test_sequential_both_factory_runs_one_logged_call_per_policy():
    outcomes = [
        RawBackendResponse(
            json.dumps({"violated": policy_id in {2, 7}}),
            input_tokens=12,
            output_tokens=2,
        )
        for policy_id in range(1, 10)
    ]
    backend = _backend(outcomes)
    verifier = make_sequential_both_verifier(
        backend=backend,
        policies=_policies(),
    )

    result = verifier.verify_full(
        InferenceInput("opaque", "synthetic request")
    )

    assert result.predicted_policy_ids == (2, 7)
    assert result.checked_policy_ids == tuple(range(1, 10))
    assert len(result.call_ids) == 9
    assert [record.policy_id for record in backend.ledger.records] == list(
        range(1, 10)
    )
    assert all(record.parse_status == "valid_verdict" for record in backend.ledger.records)
    assert "Synthetic policy 2" in backend.backend.requests[1].prompt
    assert "Synthetic policy 1" not in backend.backend.requests[1].prompt
