from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"rulegate_test_{name}", ROOT / "kaggle_exec" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


controller = _load("controller")
runner = _load("runner")

COMMIT = "c" * 40


@pytest.mark.parametrize(
    "text, state",
    [
        ('alice/rulegate-runner has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
        ('alice/rulegate-runner has status "running"', "RUNNING"),
        ('alice/rulegate-runner has status "KernelWorkerStatus.ERROR"\n'
         'Failure message: "boom"', "ERROR"),
        ("404 Not Found", None),
    ],
)
def test_parse_status(text, state):
    parsed_state, failure = controller.parse_status(text)
    assert parsed_state == state
    assert failure == ("boom" if state == "ERROR" else None)


def test_parse_push_output_requires_a_version():
    version, url = controller.parse_push_output(
        "Kernel version 7 successfully pushed.  Please check progress at "
        "https://www.kaggle.com/code/alice/rulegate-runner"
    )
    assert version == 7
    assert url.endswith("/rulegate-runner")
    with pytest.raises(controller.ControllerError):
        controller.parse_push_output("Kernel push error: Notebook not found")
    with pytest.raises(controller.ControllerError):
        controller.parse_push_output("something unexpected")


def _args(**overrides):
    values = dict(
        mode="smoke-gpu",
        repo_url="https://github.com/fakedupiece/nlp-project.git",
        model_revision="resolve",
        quantization="none",
        dtype="float16",
        max_new_tokens=128,
        max_attempts=1,
        limit=2,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_rendered_runner_embeds_a_valid_config(tmp_path):
    config = controller.build_run_config(_args(), COMMIT, "run-1")
    kernel_dir = controller.write_kernel_bundle(tmp_path, config, "alice", "NvidiaTeslaT4")
    rendered = (kernel_dir / controller.CODE_FILE).read_text(encoding="utf-8")
    namespace: dict = {"__name__": "rendered_runner"}
    exec(compile(rendered, "rendered", "exec"), namespace)
    loaded = namespace["load_config"]()
    assert loaded == config
    assert loaded["benchmark_commit"] == "2d1944f2efa362021f931638b724714f7b74f288"
    assert loaded["limit"] == 2

    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    assert metadata["id"] == "alice/rulegate-runner"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["kernel_type"] == "script"
    assert metadata["code_file"] == controller.CODE_FILE


def test_run_design_requires_an_exact_model_revision():
    with pytest.raises(controller.ControllerError, match="run-design requires"):
        controller.build_run_config(_args(mode="run-design"), COMMIT, "run-2")
    with pytest.raises(controller.ControllerError, match="--limit"):
        controller.build_run_config(_args(limit=40), COMMIT, "run-3")


def _write_outputs(output_dir: Path, run_id: str, *, parse_ok: bool = True) -> None:
    root = output_dir / "rulegate"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    (root / "status.json").write_text(json.dumps({
        "run_id": run_id,
        "final_status": "ok" if parse_ok else "completed_with_parse_failures",
        "blocker": None,
        "stages": [{"name": "environment", "status": "ok", "seconds": 1.0, "detail": None}],
        "facts": {"hf_secret_available": True, "cpu_tests": "77 passed"},
    }))
    (root / "environment.json").write_text(json.dumps({
        "torch": {"cuda_available": True, "torch_version": "2.x", "devices": [
            {"index": 0, "name": "Tesla T4", "total_memory_bytes": 15 * 2**30,
             "compute_capability": "7.5"}]}
    }))
    calls = [
        {"call_id": 1, "request_id": "rg-design-1", "method": "single_prompting",
         "policy_id": None, "attempt": 1, "prompt_hash": "h1", "success": True,
         "input_tokens": 900, "output_tokens": 9, "elapsed_seconds": 1.5,
         "parse_status": "valid_single_response" if parse_ok else "invalid_single_response"},
    ]
    (artifacts / "single_calls.jsonl").write_text(
        "\n".join(json.dumps(c) for c in calls) + "\n"
    )
    (artifacts / "summary.json").write_text(json.dumps({
        "classification": "development_smoke", "locked_test_used": False,
        "all_outputs_parsed": parse_ok,
    }))
    (artifacts / "run_metadata.json").write_text(json.dumps({"model": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct", "resolved_revision": "b" * 40,
        "dtype": "torch.float16", "quantization": "none", "device_map": [["model", "0"]],
    }}))
    raw = root / "raw"
    raw.mkdir()
    (raw / "raw_responses.jsonl").write_text(json.dumps({
        "request_id": "rg-design-1", "method": "single_prompting", "policy_id": None,
        "attempt": 1, "prompt_hash": "h1", "text": "Sure! ```json {...}```",
    }) + "\n")


def _record(run_id="run-9"):
    return {
        "run_id": run_id, "created_at_utc": "2026-11-11T00:00:00+00:00",
        "mode": "smoke-gpu", "code_commit": COMMIT, "kernel_ref": "alice/rulegate-runner",
        "kernel_version": 3, "accelerator": "NvidiaTeslaT4", "kaggle_status": "COMPLETE",
    }


def test_report_uses_runner_status_and_contains_no_raw_text(tmp_path):
    _write_outputs(tmp_path, "run-9")
    report, history = controller.build_report(_record(), tmp_path)
    assert "Runner final status: **`ok`**" in report
    assert "Tesla T4" in report
    assert "| Single Prompting | 1 | 1 |" in report
    assert "Sure!" not in report
    assert history["runner_final_status"] == "ok"
    assert history["calls"]["Single Prompting"]["input_tokens"] == 900


def test_report_flags_outputs_from_a_different_run(tmp_path):
    _write_outputs(tmp_path, "some-other-run")
    _, history = controller.build_report(_record(), tmp_path)
    assert history["runner_final_status"] == "MISMATCHED_OUTPUT"


def test_kaggle_complete_without_status_file_is_not_success(tmp_path):
    report, history = controller.build_report(_record(), tmp_path)
    assert history["runner_final_status"] == "missing_status_json"
    assert "Kaggle status: `COMPLETE`" in report


def test_local_diagnosis_lists_unparsed_raw_responses(tmp_path):
    _write_outputs(tmp_path / "out", "run-9", parse_ok=False)
    path = controller.write_local_diagnosis(tmp_path / "out", tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "do not commit" in text
    assert "Sure!" in text
