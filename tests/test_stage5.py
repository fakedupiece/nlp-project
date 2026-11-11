from __future__ import annotations

import json
from pathlib import Path

import pytest

from rulegate.data import load_benchmark
from rulegate.model_backend import (
    CallLedger,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
)
from rulegate.prompts import SINGLE_PROMPT_TEMPLATE_ID
from rulegate.screening_prompts import (
    ACTION_AWARE_SCREENING_TEMPLATE_ID,
    PLAIN_SCREENING_TEMPLATE_ID,
)
from rulegate.schemas import POLICY_IDS
from rulegate.stage3 import CONDITIONS, write_stage3_artifacts
from rulegate.stage5 import (
    CONDITION_ORDER,
    CONFIRMATORY_NOTE,
    confirmatory_decision,
    run_locked_test,
)


KEY = b"stage5-test-key-material-32bytes!"


def _write_dataset(root: Path) -> Path:
    (root / "policy_list").mkdir(parents=True)
    (root / "testing_requests").mkdir()
    (root / "user_requests").mkdir()
    (root / "policy_list" / "policy_list.txt").write_text(
        "".join(f"{i}- Synthetic policy {i}.\n" for i in range(1, 10)), encoding="utf-8"
    )
    (root / "testing_requests" / "test_user_requests.txt").write_text(
        ">>! design compliant\n[]\n", encoding="utf-8"
    )
    for pid in range(1, 10):
        (root / "user_requests" / f"part_{pid}.txt").write_text(
            f">>! locked compliant {pid}\n[]\n>>! locked violation {pid}\n[{pid}]\n",
            encoding="utf-8",
        )
    return root


def _factory(dataset):
    gold = {r.request_id: r.gold_violated_policy_ids for r in dataset.test_records}

    def respond(request):
        g = gold[request.request_id]
        if request.prompt_template in {PLAIN_SCREENING_TEMPLATE_ID,
                                       ACTION_AWARE_SCREENING_TEMPLATE_ID}:
            # Screen keeps the gold policy (if any) plus policy 1.
            keep = set(g) | {1}
            labels = [{"policy_id": p, "label": "POSSIBLE" if p in keep else "EXCLUDE"}
                      for p in POLICY_IDS]
            payload = {"labels": labels}
            if request.prompt_template == ACTION_AWARE_SCREENING_TEMPLATE_ID:
                payload = {"action": "a", "target": "t", "content": "c", **payload}
            return RawBackendResponse(json.dumps(payload), 800, 150)
        if request.prompt_template == SINGLE_PROMPT_TEMPLATE_ID:
            return RawBackendResponse(json.dumps({"violated_policy_ids": list(g)}), 400, 9)
        # Verifier: right on the gold policy, false positive on policy 4.
        verdict = request.policy_id in g or request.policy_id == 4
        return RawBackendResponse(json.dumps({"violated": verdict}), 300, 7)

    return lambda: InstrumentedBackend(FakeBackend([respond] * 400),
                                       ModelSettings(model_id="fake/m"), CallLedger())


def test_locked_test_runs_all_conditions_on_test_records_and_checkpoints(tmp_path):
    dataset = load_benchmark(_write_dataset(tmp_path / "d"), strict_counts=False, id_key=KEY)
    runs = run_locked_test(dataset, _factory(dataset), expected_requests=18,
                           partial_dir=tmp_path / "partial", progress=lambda m: None)
    assert [r.name for r in runs] == [
        "full_sequential_both", "single_prompting", "plain_screening", "action_aware_screening",
    ]
    test_ids = {r.request_id for r in dataset.test_records}
    for run in runs:
        assert set(run.request_ids) == test_ids
        assert all(rec.request_id in test_ids for rec in run.ledger.records)
        assert (tmp_path / "partial" / run.name / "calls.jsonl").is_file()
    assert CONDITION_ORDER == ("plain", "action-aware")


def test_locked_test_requires_the_expected_request_count(tmp_path):
    dataset = load_benchmark(_write_dataset(tmp_path / "d"), strict_counts=False, id_key=KEY)
    with pytest.raises(RuntimeError, match="expected 225"):
        run_locked_test(dataset, _factory(dataset), expected_requests=225)


def test_confirmatory_decision_applies_the_preregistered_rules(tmp_path):
    dataset = load_benchmark(_write_dataset(tmp_path / "d"), strict_counts=False, id_key=KEY)
    runs = run_locked_test(dataset, _factory(dataset), expected_requests=18,
                           progress=lambda m: None)
    summary = write_stage3_artifacts(
        dataset, runs, output_dir=tmp_path / "out", classification="locked_test",
        model_metadata={"model_id": "fake/m"}, code_commit="a" * 40,
        conditions=[CONDITIONS[n] for n in CONDITION_ORDER], split="test",
        comparison_note=CONFIRMATORY_NOTE, stage=5, bootstrap_resamples=300,
    )
    assert summary["split"] == "test" and summary["locked_test_used"] is True
    assert "stage3_gate" not in summary
    assert (tmp_path / "out" / "trusted_test_index.json").is_file()
    decision = confirmatory_decision(summary)
    # Screens drop the policy-4 false positives: F1 rises and calls fall.
    assert decision["condition"] == "action_aware_screening"
    assert decision["h1"] == "GO" and decision["h1_delta_f1"]["point_estimate"] > 0
    assert decision["h2"] == "GO" and decision["h2_call_reduction"] > 0.5
    assert decision["overall"] == "GO"
    metadata = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert metadata["split"] == "test" and metadata["locked_test_used"] is True
    assert metadata["stage"] == 5


def test_overall_decision_requires_both_hypotheses():
    def summary(h1, h2):
        return {"conditions": {"action_aware_screening": {"vs_full_sequential_both": {
            "h1_status": h1, "h2_status": h2, "h1_delta_f1": {}, "h2_call_reduction": 0.4}}}}

    assert confirmatory_decision(summary("GO", "FAIL"))["overall"] == "NOT SUPPORTED"
    assert confirmatory_decision(summary("INCONCLUSIVE", "GO"))["overall"] == "INCONCLUSIVE"
    assert confirmatory_decision(summary("FAIL", "GO"))["overall"] == "NOT SUPPORTED"
    assert confirmatory_decision(
        {"conditions": {"action_aware_screening": {}}})["status"] == "NOT_EVALUABLE"
