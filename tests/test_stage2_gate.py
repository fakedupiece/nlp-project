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
from rulegate.stage2 import run_design_baselines, write_design_artifacts
from rulegate.stage2_gate import evaluate_stage2_gate


KEY = b"stage2-test-key-material-32bytes!"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
CODE = "d" * 40


def _write_dataset(root: Path) -> Path:
    (root / "policy_list").mkdir(parents=True)
    (root / "testing_requests").mkdir()
    (root / "user_requests").mkdir()
    (root / "policy_list" / "policy_list.txt").write_text(
        "".join(f"{i}- Synthetic policy {i}.\n" for i in range(1, 10)), encoding="utf-8"
    )
    (root / "testing_requests" / "test_user_requests.txt").write_text(
        ">>! design compliant\n[]\n>>! design policy two\n[2]\n", encoding="utf-8"
    )
    for policy_id in range(1, 10):
        (root / "user_requests" / f"part_{policy_id}.txt").write_text(
            f">>! test request {policy_id}\n[{policy_id}]\n", encoding="utf-8"
        )
    return root


def _backend(outcomes):
    settings = ModelSettings(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        revision=REVISION,
        quantization="none",
        temperature=0.0,
        seed=484654,
        decoding_settings=(("max_new_tokens", "128"), ("do_sample", "false"),
                           ("dtype", "float16")),
    )
    return InstrumentedBackend(FakeBackend(outcomes), settings, CallLedger())


def _artifacts(tmp_path: Path) -> Path:
    dataset = load_benchmark(_write_dataset(tmp_path / "data"), strict_counts=False, id_key=KEY)
    single = _backend([
        RawBackendResponse('{"violated_policy_ids": []}', 900, 9),
        RawBackendResponse('{"violated_policy_ids": [2]}', 900, 10),
    ])
    full = _backend([
        RawBackendResponse(json.dumps({"violated": r == 1 and p == 2}), 400, 6)
        for r in range(2)
        for p in range(1, 10)
    ])
    run = run_design_baselines(dataset, single_backend=single, full_sequential_backend=full)
    out = tmp_path / "artifacts"
    write_design_artifacts(
        run,
        dataset,
        output_dir=out,
        classification="development_model",
        model_metadata={
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "resolved_revision": REVISION,
            "quantization": "none",
            "requested_dtype": "float16",
            "dtype": "torch.float16",
            "temperature": 0.0,
            "do_sample": False,
            "seed": 484654,
        },
        code_commit=CODE,
    )
    # Emulate the clean pinned Kaggle checkout that a real run records.
    metadata = json.loads((out / "run_metadata.json").read_text())
    metadata["local_git"] = {"commit": CODE, "dirty": False}
    (out / "run_metadata.json").write_text(json.dumps(metadata))
    return out


def test_complete_reconciled_run_is_go(tmp_path):
    report = evaluate_stage2_gate(
        _artifacts(tmp_path), expected_model_revision=REVISION, expected_requests=2
    )
    assert report.decision == "GO", report.to_dict()["failed_checks"]
    assert report.facts["calls"]["single_prompting"] == 2
    assert report.facts["calls"]["full_sequential_both"] == 18


def _rewrite_jsonl(path: Path, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.mark.parametrize(
    "mutation, failed_check",
    [
        (lambda out: _rewrite_jsonl(
            out / "full_sequential_calls.jsonl",
            lambda rows: rows[3].update(parse_status="invalid_verdict", parsed_output=None)),
         "every_output_parsed"),
        (lambda out: (out / "single_predictions.json").write_text(json.dumps([
            {**p, "predicted_policy_ids": [5]}
            for p in json.loads((out / "single_predictions.json").read_text())])),
         "single_reconciles"),
        (lambda out: _rewrite_jsonl(
            out / "full_sequential_calls.jsonl",
            lambda rows: rows[0].update(model_revision="f" * 40)),
         "one_model_configuration"),
        (lambda out: (out / "summary.json").write_text(json.dumps({
            **json.loads((out / "summary.json").read_text()), "locked_test_used": True})),
         "design_split_only"),
        (lambda out: _rewrite_jsonl(
            out / "single_calls.jsonl", lambda rows: rows[0].update(input_tokens=None)),
         "token_counts_known"),
    ],
)
def test_any_failed_condition_is_continue(tmp_path, mutation, failed_check):
    out = _artifacts(tmp_path)
    mutation(out)
    report = evaluate_stage2_gate(out, expected_model_revision=REVISION, expected_requests=2)
    assert report.decision == "CONTINUE"
    assert failed_check in report.to_dict()["failed_checks"]


def test_aborted_run_and_wrong_revision_are_continue(tmp_path):
    out = _artifacts(tmp_path)
    report = evaluate_stage2_gate(out, expected_model_revision="e" * 40, expected_requests=2)
    assert "pinned_model_revision" in report.to_dict()["failed_checks"]
    (out / "failure.json").write_text("{}")
    report = evaluate_stage2_gate(out, expected_requests=2)
    assert report.to_dict()["failed_checks"] == ["artifacts_complete"]


def test_full_design_count_is_required_by_default(tmp_path):
    report = evaluate_stage2_gate(_artifacts(tmp_path))
    assert "all_design_requests" in report.to_dict()["failed_checks"]
