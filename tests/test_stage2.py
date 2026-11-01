from __future__ import annotations

import json
from pathlib import Path

import pytest

from rulegate.data import load_benchmark
from rulegate.model_backend import (
    BackendCallError,
    CallRequest,
    CallLedger,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
)
from rulegate.stage2 import (
    run_design_baselines,
    write_design_artifacts,
    write_failed_design_artifacts,
)


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


def _instrumented(outcomes):
    return InstrumentedBackend(
        FakeBackend(outcomes),
        ModelSettings(model_id="fake/model", revision="synthetic", seed=7),
        CallLedger(),
    )


def test_design_runner_and_artifacts_keep_model_outputs_separate_from_gold(
    tmp_path: Path,
) -> None:
    dataset = load_benchmark(
        _write_dataset(tmp_path / "data"),
        strict_counts=False,
        id_key=b"stage2-test-key-material-32bytes!",
    )
    single = _instrumented(
        [
            RawBackendResponse('{"violated_policy_ids": []}', 10, 2),
            RawBackendResponse('{"violated_policy_ids": [2]}', 10, 3),
        ]
    )
    sequential_outcomes = []
    for request_index in range(2):
        sequential_outcomes.extend(
            RawBackendResponse(
                json.dumps(
                    {"violated": request_index == 1 and policy_id == 2}
                ),
                8,
                2,
            )
            for policy_id in range(1, 10)
        )
    full = _instrumented(sequential_outcomes)

    run = run_design_baselines(
        dataset,
        single_backend=single,
        full_sequential_backend=full,
    )
    output = tmp_path / "artifacts"
    summary = write_design_artifacts(
        run,
        dataset,
        output_dir=output,
        classification="development_model",
        model_metadata={"model_id": "fake/model", "synthetic": True},
        code_commit="a" * 40,
    )

    assert summary["request_count"] == 2
    assert summary["single_prompting_calls"]["total_calls"] == 2
    assert summary["full_sequential_both_calls"]["total_calls"] == 18
    assert summary["single_prompting_metrics"]["requests"]["exact_matches"] == 2
    predictions_text = (output / "single_predictions.json").read_text(encoding="utf-8")
    assert "gold_violated" not in predictions_text
    assert "source_file" not in predictions_text
    assert "design policy two" not in predictions_text
    trusted_index = json.loads(
        (output / "trusted_design_index.json").read_text(encoding="utf-8")
    )
    assert trusted_index[1]["gold_violated_policy_ids"] == [2]
    assert len((output / "full_sequential_calls.jsonl").read_text().splitlines()) == 18
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["locked_test_used"] is False
    assert metadata["prompts"]["single"]["template_sha256"]


def test_synthetic_smoke_summary_does_not_report_model_metrics(tmp_path: Path) -> None:
    dataset = load_benchmark(
        _write_dataset(tmp_path / "data"),
        strict_counts=False,
        id_key=b"stage2-test-key-material-32bytes!",
    )
    single = _instrumented(
        [RawBackendResponse('{"violated_policy_ids": []}') for _ in range(2)]
    )
    full = _instrumented(
        [RawBackendResponse('{"violated": false}') for _ in range(18)]
    )
    run = run_design_baselines(
        dataset,
        single_backend=single,
        full_sequential_backend=full,
    )

    summary = write_design_artifacts(
        run,
        dataset,
        output_dir=tmp_path / "smoke",
        classification="synthetic_smoke",
        model_metadata={"synthetic": True},
        code_commit=None,
    )

    assert "single_prompting_metrics" not in summary
    assert summary["locked_test_used"] is False


def test_artifact_writer_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    dataset = load_benchmark(
        _write_dataset(tmp_path / "data"),
        strict_counts=False,
        id_key=b"stage2-test-key-material-32bytes!",
    )
    single = _instrumented(
        [RawBackendResponse('{"violated_policy_ids": []}') for _ in range(2)]
    )
    full = _instrumented(
        [RawBackendResponse('{"violated": false}') for _ in range(18)]
    )
    run = run_design_baselines(
        dataset,
        single_backend=single,
        full_sequential_backend=full,
    )
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        write_design_artifacts(
            run,
            dataset,
            output_dir=output,
            classification="synthetic_smoke",
            model_metadata={"synthetic": True},
            code_commit=None,
        )


def test_failed_run_artifacts_preserve_attempts_without_exception_message(
    tmp_path: Path,
) -> None:
    single = CallLedger()
    full = _instrumented([RuntimeError("sensitive backend response")])

    with pytest.raises(BackendCallError) as captured:
        full.complete(
            # The concrete request is deliberately minimal; only the ledger
            # record is relevant to the failure artifact.
            CallRequest(
                request_id="opaque-request",
                method="full_sequential_both",
                prompt="inert request",
                prompt_template="test-v1",
                policy_id=1,
            )
        )

    output = tmp_path / "failed"
    write_failed_design_artifacts(
        output_dir=output,
        single_ledger=single,
        full_sequential_ledger=full.ledger,
        error=captured.value,
    )

    failure_text = (output / "failure.json").read_text(encoding="utf-8")
    failure = json.loads(failure_text)
    assert failure["single_prompting_attempted_calls"] == 0
    assert failure["full_sequential_both_attempted_calls"] == 1
    assert failure["error_type"] == "BackendCallError"
    assert "sensitive backend response" not in failure_text
    assert len((output / "full_sequential_calls.jsonl").read_text().splitlines()) == 1
