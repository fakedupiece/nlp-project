import json

import pytest

from rulegate.methods import InferenceInput, MethodName
from rulegate.model_backend import (
    CallLedger,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
)
from rulegate.screening import (
    POLICY_IDS,
    ScreenLabel,
    ScreeningParseError,
    parse_screen_response,
    run_screening,
)
from rulegate.verification import SequentialBothVerifier


def _screen_json(possible=(2,), uncertain=(7,)):
    return json.dumps(
        {
            "labels": [
                {
                    "policy_id": policy_id,
                    "label": (
                        "POSSIBLE"
                        if policy_id in possible
                        else "UNCERTAIN"
                        if policy_id in uncertain
                        else "EXCLUDE"
                    ),
                }
                for policy_id in POLICY_IDS
            ]
        }
    )


def _instrumented(outcomes):
    return InstrumentedBackend(
        FakeBackend(outcomes),
        ModelSettings(model_id="fake/model", revision="test", seed=7),
        CallLedger(),
    )


def test_valid_screen_retains_possible_and_uncertain_in_policy_order():
    backend = _instrumented([RawBackendResponse(_screen_json(possible=(9, 1), uncertain=(4,)))])
    decision = run_screening(
        backend=backend,
        request=InferenceInput("req-001", "synthetic request"),
        prompt="synthetic screen prompt",
        prompt_template="screen-v0-test",
        method=MethodName.ACTION_AWARE_SCREENING,
    )

    assert decision.valid
    assert decision.candidate_ids == (1, 4, 9)
    assert not decision.fallback_used
    assert backend.ledger.records[0].parse_status == "valid_screen"


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"labels": []}),
        json.dumps(
            {
                "labels": [
                    {"policy_id": 1, "label": "EXCLUDE"}
                    for _ in POLICY_IDS
                ]
            }
        ),
        json.dumps(
            {
                "labels": [
                    {"policy_id": policy_id, "label": "MAYBE"}
                    for policy_id in POLICY_IDS
                ]
            }
        ),
    ],
)
def test_strict_parser_rejects_malformed_missing_duplicate_or_unknown(payload):
    with pytest.raises(ScreeningParseError):
        parse_screen_response(payload)


def test_invalid_screen_uses_full_fallback_and_verifier_records_all_calls():
    outcomes = [RawBackendResponse("malformed")]
    outcomes.extend(RawBackendResponse('{"violated": false}', 10, 2) for _ in POLICY_IDS)
    backend = _instrumented(outcomes)
    request = InferenceInput("req-fallback", "synthetic fallback request")

    decision = run_screening(
        backend=backend,
        request=request,
        prompt="synthetic screen prompt",
        prompt_template="screen-v0-test",
        method=MethodName.PLAIN_SCREENING,
    )
    verifier = SequentialBothVerifier(
        backend=backend,
        prompt_builder=lambda text, policy_id: f"verify {policy_id}: {text}",
        prompt_template="sequential-both-v0-test",
    )
    result = verifier.verify_screened(request, decision)

    assert decision.candidate_ids == POLICY_IDS
    assert decision.fallback_used
    assert result.checked_policy_ids == POLICY_IDS
    assert len(backend.ledger.records) == 10
    assert backend.ledger.records[0].parse_status == "invalid_screen"
    assert backend.ledger.records[0].fallback_used
    assert backend.ledger.records[0].failure_reason.startswith("invalid_screen:")
    assert all(record.fallback_used for record in backend.ledger.records[1:])
    assert all(
        record.fallback_type == "full_nine_policy_verification"
        for record in backend.ledger.records[1:]
    )


def test_valid_zero_candidate_screen_does_not_trigger_fallback():
    backend = _instrumented([RawBackendResponse(_screen_json(possible=(), uncertain=()))])
    decision = run_screening(
        backend=backend,
        request=InferenceInput("req-zero", "synthetic compliant request"),
        prompt="synthetic screen prompt",
        prompt_template="screen-v0-test",
        method=MethodName.ACTION_AWARE_SCREENING,
    )
    assert decision.valid
    assert decision.candidate_ids == ()
    assert not decision.fallback_used
    assert {route.label for route in decision.routes} == {ScreenLabel.EXCLUDE}
