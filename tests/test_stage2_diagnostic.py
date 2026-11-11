from __future__ import annotations

import json
from pathlib import Path

import pytest

from rulegate.data import load_benchmark
from rulegate.model_backend import (
    BackendCallError,
    CallLedger,
    CallRequest,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
    RawResponseRecorder,
    prompt_hash,
)
from rulegate.stage2 import run_design_diagnostic, write_diagnostic_artifacts


KEY = b"stage2-test-key-material-32bytes!"


def _write_dataset(root: Path) -> Path:
    (root / "policy_list").mkdir(parents=True)
    (root / "testing_requests").mkdir()
    (root / "user_requests").mkdir()
    (root / "policy_list" / "policy_list.txt").write_text(
        "".join(f"{i}- Synthetic policy {i}.\n" for i in range(1, 10)),
        encoding="utf-8",
    )
    (root / "testing_requests" / "test_user_requests.txt").write_text(
        ">>! design compliant\n[]\n>>! design policy two\n[2]\n",
        encoding="utf-8",
    )
    for policy_id in range(1, 10):
        (root / "user_requests" / f"part_{policy_id}.txt").write_text(
            f">>! test request {policy_id}\n[{policy_id}]\n",
            encoding="utf-8",
        )
    return root


def _dataset(tmp_path: Path):
    return load_benchmark(_write_dataset(tmp_path / "data"), strict_counts=False, id_key=KEY)


def _instrumented(outcomes):
    return InstrumentedBackend(
        FakeBackend(outcomes),
        ModelSettings(model_id="fake/model", revision="synthetic", seed=7),
        CallLedger(),
    )


def test_diagnostic_run_records_parse_failures_and_keeps_going(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    single = _instrumented(
        [
            RawBackendResponse("Sure! The answer is none.", 10, 6),
            RawBackendResponse('{"violated_policy_ids": [2]}', 10, 3),
        ]
    )
    sequential_outcomes = []
    for request_index in range(2):
        for policy_id in range(1, 10):
            if request_index == 0 and policy_id == 4:
                sequential_outcomes.append(RawBackendResponse("```json\n{}\n```", 8, 4))
            else:
                sequential_outcomes.append(
                    RawBackendResponse(
                        json.dumps({"violated": request_index == 1 and policy_id == 2}),
                        8,
                        2,
                    )
                )
    full = _instrumented(sequential_outcomes)
    progress: list[str] = []

    run = run_design_diagnostic(
        dataset,
        single_backend=single,
        full_sequential_backend=full,
        limit=2,
        progress=progress.append,
    )

    assert not run.all_ok
    assert [o.status for o in run.single_outcomes] == ["failed", "ok"]
    assert run.single_outcomes[0].call_ids == (1,)
    assert run.full_sequential_outcomes[0].failed_policy_ids == (4,)
    assert len(run.full_sequential_outcomes[0].call_ids) == 9
    assert run.full_sequential_outcomes[1].predicted_policy_ids == (2,)
    # Same call count as the strict runner: one call per request-policy pair.
    assert len(full.ledger.records) == 18
    assert len(progress) == 4

    summary = write_diagnostic_artifacts(
        run,
        dataset,
        output_dir=tmp_path / "diag",
        model_metadata={"model_id": "fake/model"},
        code_commit="a" * 40,
    )
    assert summary["all_outputs_parsed"] is False
    assert summary["single_prompting_parse_status"] == {
        "invalid_single_response": 1,
        "valid_single_response": 1,
    }
    assert summary["full_sequential_both_parse_status"]["invalid_verdict"] == 1
    assert "single_prompting_subset_metrics" not in summary
    outcomes_text = (tmp_path / "diag" / "diagnostic_outcomes.json").read_text()
    assert "Sure!" not in outcomes_text
    assert "design policy two" not in outcomes_text


def test_diagnostic_run_reports_subset_metrics_only_when_all_outputs_parse(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    single = _instrumented([RawBackendResponse('{"violated_policy_ids": []}', 5, 2)])
    full = _instrumented([RawBackendResponse('{"violated": false}', 5, 2)] * 9)

    run = run_design_diagnostic(
        dataset, single_backend=single, full_sequential_backend=full, limit=1
    )
    summary = write_diagnostic_artifacts(
        run,
        dataset,
        output_dir=tmp_path / "diag",
        model_metadata={"model_id": "fake/model"},
        code_commit="a" * 40,
    )
    assert run.all_ok
    assert summary["request_count"] == 1
    assert "subset only" in summary["subset_metrics_note"]
    assert summary["full_sequential_both_subset_metrics"]["requests"]["exact_matches"] == 1


@pytest.mark.parametrize("limit", [0, 3, True])
def test_diagnostic_limit_is_bounded_by_the_design_split(tmp_path: Path, limit) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        run_design_diagnostic(
            dataset,
            single_backend=_instrumented([]),
            full_sequential_backend=_instrumented([]),
            limit=limit,
        )


def test_raw_response_recorder_joins_to_ledger_and_survives_errors(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "raw.jsonl"
    recorder = RawResponseRecorder(
        FakeBackend([RawBackendResponse("not json", 4, 2), RuntimeError("boom")]),
        path,
    )
    backend = InstrumentedBackend(
        recorder, ModelSettings(model_id="fake/model", revision="synthetic"), CallLedger()
    )
    request = CallRequest(
        request_id="opaque",
        method="full_sequential_both",
        prompt="inert",
        prompt_template="t-v1",
        policy_id=3,
    )
    backend.complete(request)
    with pytest.raises(BackendCallError):
        backend.complete(request)

    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["text"] for e in entries] == ["not json", None]
    assert entries[1]["error_type"] == "RuntimeError"
    assert entries[0]["prompt_hash"] == backend.ledger.records[0].prompt_hash
    assert entries[0]["prompt_hash"] == prompt_hash(request)
    with pytest.raises(FileExistsError):
        RawResponseRecorder(FakeBackend([]), path)
