from __future__ import annotations

import pytest

from rulegate.baselines import (
    SinglePromptingBaseline,
    SinglePromptingParseError,
    make_sequential_both_verifier,
    parse_single_prompt_response,
)
from rulegate.model_backend import (
    CallLedger,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
)
from rulegate.prompts import CLASSIFIER_SYSTEM_PROMPT_V2
from rulegate.schemas import InferenceRequest, Policy
from rulegate.verification import VerificationParseError, parse_json_verdict


POLICIES = tuple(Policy(i, f"Synthetic policy {i}") for i in range(1, 10))


def _backend(outcomes):
    fake = FakeBackend(outcomes)
    return fake, InstrumentedBackend(fake, ModelSettings(model_id="fake/model"), CallLedger())


def test_both_baselines_send_the_v2_system_message():
    request = InferenceRequest(request_id="rg-design-x", text="inert request")
    fake, backend = _backend([RawBackendResponse('{"violated_policy_ids": [3]}', 1, 1)])
    assert SinglePromptingBaseline(backend=backend, policies=POLICIES).predict(request) == (3,)
    assert fake.requests[0].system_prompt == CLASSIFIER_SYSTEM_PROMPT_V2

    fake, backend = _backend([RawBackendResponse('{"violated": false}', 1, 1)] * 9)
    verifier = make_sequential_both_verifier(backend=backend, policies=POLICIES)
    assert verifier.verify_full(request).predicted_policy_ids == ()
    assert {r.system_prompt for r in fake.requests} == {CLASSIFIER_SYSTEM_PROMPT_V2}


@pytest.mark.parametrize(
    "text, expected",
    [
        ('```json\n{"violated": true}\n```', True),
        ('```\n{"violated": false}\n```', False),
        ('  {"violated": false}  ', False),
    ],
)
def test_verdict_parser_accepts_one_wrapping_fence(text, expected):
    assert parse_json_verdict(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        'The answer is {"violated": true}',
        '```json\n{"violated": true}\n```\nBecause the request asks for passwords.',
        '{"violated": "yes"}',
    ],
)
def test_verdict_parser_still_rejects_prose_and_wrong_types(text):
    with pytest.raises(VerificationParseError):
        parse_json_verdict(text)


def test_single_parser_accepts_fenced_json_and_rejects_prose():
    assert parse_single_prompt_response('```json\n{"violated_policy_ids": [2, 1]}\n```') == (1, 2)
    with pytest.raises(SinglePromptingParseError):
        parse_single_prompt_response('Violated: {"violated_policy_ids": [1]}')
