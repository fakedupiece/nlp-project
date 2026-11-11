from __future__ import annotations

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


runner = _load("runner")

COMMIT = "a" * 40
BENCH = "2d1944f2efa362021f931638b724714f7b74f288"
SECRET = "hf_super_secret_value_123"


def _config(**overrides):
    config = {
        "run_id": "20261111T000000Z-probe-aaaaaaaa",
        "mode": "probe",
        "repo_url": "https://example.invalid/repo.git",
        "code_commit": COMMIT,
        "benchmark_url": "https://example.invalid/bench.git",
        "benchmark_commit": BENCH,
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "resolve",
        "quantization": "none",
        "dtype": "float16",
        "max_new_tokens": 128,
        "max_attempts": 1,
        "seed": 484654,
        "hf_secret_name": "HF_TOKEN",
    }
    config.update(overrides)
    return config


def test_template_refuses_to_run_without_injected_config():
    with pytest.raises(ValueError, match="not injected"):
        runner.load_config()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mode": "locked-test"}, "mode must be one of"),
        ({"code_commit": "main"}, "code_commit"),
        ({"mode": "run-design", "model_revision": "resolve"}, "exact 40-character"),
        ({"model_revision": "main"}, "model_revision"),
        ({"mode": "smoke-gpu", "limit": 19}, "limit"),
        ({"mode": "smoke-gpu"}, "limit"),
        ({"dtype": "float32"}, "dtype"),
        ({"quantization": "8bit"}, "quantization"),
    ],
)
def test_config_validation_rejects_unsafe_or_ambiguous_runs(overrides, message):
    with pytest.raises(ValueError, match=message):
        runner.load_config(json.dumps(_config(**overrides)))


def test_redactor_removes_registered_secret():
    redact = runner.Redactor()
    redact.add(SECRET)
    assert redact(f"token={SECRET} done") == "token=[REDACTED] done"


class FakeRemote:
    """Monkeypatched runner side effects; no network, git, pip or GPU."""

    def __init__(self, monkeypatch, *, cuda=True, stage2_exit=0):
        self.commands: list[tuple[list[str], dict | None]] = []
        self.stage2_exit = stage2_exit
        monkeypatch.setattr(
            runner,
            "probe_environment",
            lambda redact: {
                "torch": {
                    "cuda_available": cuda,
                    "devices": [
                        {"index": 0, "name": "Tesla T4", "total_memory_bytes": 15 * 2**30,
                         "compute_capability": "7.5"}
                    ] if cuda else [],
                }
            },
        )
        monkeypatch.setattr(runner, "git_clone_at", lambda url, commit, dest, redact: commit)
        monkeypatch.setattr(runner, "run_command", self.run_command)
        monkeypatch.setattr(
            runner,
            "check_model_access",
            lambda model_id, revision, token, cache: {
                "model_id": model_id,
                "requested_revision": revision,
                "resolved_revision": "b" * 40,
                "config_torch_dtype": "bfloat16",
            },
        )

    def run_command(self, command, *, redact, log_path=None, env=None, cwd=None, echo=True):
        # Snapshot: the runner scrubs HF_TOKEN from its env dict after the child exits.
        self.commands.append((command, dict(env) if env is not None else None))
        if "rulegate.stage2" in command:
            # Simulate a child that accidentally echoes the token.
            line = redact(f"child saw {env['HF_TOKEN']}")
            if log_path:
                log_path.write_text(line + "\n", encoding="utf-8")
            return self.stage2_exit, line
        if "pytest" in command:
            return 0, "77 passed in 3.0s"
        return 0, ""


def _status(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / "status.json").read_text())


def test_probe_without_secret_reports_actionable_blocker(tmp_path, monkeypatch):
    FakeRemote(monkeypatch)
    runner.main(
        _config(),
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        secret_reader=lambda name: None,
    )
    status = _status(tmp_path)
    assert status["final_status"] == "probe_incomplete"
    assert "not delivered to API-pushed kernels" in status["blocker"]
    assert status["facts"]["hf_secret_available"] is False
    assert [s["name"] for s in status["stages"]][:5] == [
        "environment", "clone_code", "clone_benchmark", "install", "cpu_tests"
    ]


def test_probe_with_secret_checks_model_access_and_never_writes_token(tmp_path, monkeypatch):
    FakeRemote(monkeypatch)
    runner.main(
        _config(),
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        secret_reader=lambda name: SECRET,
    )
    status = _status(tmp_path)
    assert status["final_status"] == "ok"
    assert status["facts"]["model_access"]["resolved_revision"] == "b" * 40
    for path in (tmp_path / "out").rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8")


def test_gpu_modes_fail_fast_without_cuda(tmp_path, monkeypatch):
    remote = FakeRemote(monkeypatch, cuda=False)
    runner.main(
        _config(mode="smoke-gpu", limit=2),
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        secret_reader=lambda name: SECRET,
    )
    status = _status(tmp_path)
    assert status["final_status"] == "failed"
    assert "no CUDA GPU" in status["blocker"]
    assert remote.commands == []


def test_smoke_gpu_invokes_stage2_with_pinned_resolved_revision(tmp_path, monkeypatch):
    remote = FakeRemote(monkeypatch, stage2_exit=3)
    runner.main(
        _config(mode="smoke-gpu", limit=2),
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        secret_reader=lambda name: SECRET,
    )
    status = _status(tmp_path)
    assert status["final_status"] == "completed_with_parse_failures"
    command, env = next(c for c in remote.commands if "rulegate.stage2" in c[0])
    assert command[command.index("rulegate.stage2") + 1] == "smoke-gpu"
    assert command[command.index("--model-revision") + 1] == "b" * 40
    assert command[command.index("--limit") + 1] == "2"
    assert command[command.index("--dtype") + 1] == "float16"
    assert SECRET not in " ".join(command)
    assert env["HF_TOKEN"] == SECRET
    log = (tmp_path / "out" / "stage2.log").read_text()
    assert SECRET not in log and "[REDACTED]" in log


def test_run_design_nonzero_exit_is_a_failure(tmp_path, monkeypatch):
    FakeRemote(monkeypatch, stage2_exit=1)
    runner.main(
        _config(mode="run-design", model_revision="b" * 40),
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        secret_reader=lambda name: SECRET,
    )
    status = _status(tmp_path)
    assert status["final_status"] == "failed"
    assert "exited with code 1" in status["blocker"]


def test_hf_errors_are_classified_without_leaking_details():
    class GatedRepoError(Exception):
        pass

    assert "license" in runner.classify_hf_error(GatedRepoError("403"))
    assert "rejected the token" in runner.classify_hf_error(Exception("401 Client Error"))


def test_secret_lookup_error_is_recorded_without_a_value(tmp_path, monkeypatch):
    FakeRemote(monkeypatch)

    def failing_reader(name):
        runner.SECRET_ERROR["error"] = "BackendError: No user secrets exist for kernel id 1"
        return None

    runner.main(
        _config(),
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        secret_reader=failing_reader,
    )
    status = _status(tmp_path)
    assert status["facts"]["hf_secret_error"].startswith("BackendError: No user secrets")
    runner.SECRET_ERROR.clear()


def test_private_dataset_token_is_used_when_secrets_are_unavailable(tmp_path, monkeypatch):
    FakeRemote(monkeypatch)
    seen = {}

    def dataset_reader(ref, filename):
        seen["args"] = (ref, filename)
        return SECRET, "/kaggle/input/rulegate-hf-token/hf_token.txt"

    runner.main(
        _config(hf_token_dataset="alice/rulegate-hf-token", hf_token_file="hf_token.txt"),
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        secret_reader=lambda name: None,
        dataset_token_reader=dataset_reader,
    )
    status = _status(tmp_path)
    assert seen["args"] == ("alice/rulegate-hf-token", "hf_token.txt")
    assert status["final_status"] == "ok"
    assert status["facts"]["hf_token_source"] == "private_dataset"
    for path in (tmp_path / "out").rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8")


def test_missing_dataset_token_names_both_sources(tmp_path, monkeypatch):
    FakeRemote(monkeypatch)
    runner.main(
        _config(hf_token_dataset="alice/rulegate-hf-token"),
        output_root=tmp_path / "out",
        scratch_root=tmp_path / "scratch",
        secret_reader=lambda name: None,
        dataset_token_reader=lambda ref, filename: (None, None),
    )
    status = _status(tmp_path)
    assert status["final_status"] == "probe_incomplete"
    assert "alice/rulegate-hf-token" in status["blocker"]


def test_token_file_is_found_under_any_mount_layout(tmp_path):
    nested = tmp_path / "datasets" / "alice" / "rulegate-hf-token"
    nested.mkdir(parents=True)
    (nested / "hf_token.txt").write_text(SECRET + "\n", encoding="utf-8")
    other = tmp_path / "other-dataset"
    other.mkdir()
    (other / "hf_token.txt").write_text("wrong", encoding="utf-8")

    value, path = runner.read_token_from_dataset(
        "alice/rulegate-hf-token", "hf_token.txt", input_root=tmp_path
    )
    assert value == SECRET
    assert path.endswith("hf_token.txt") and "rulegate-hf-token" in path
    assert runner.read_token_from_dataset(None, "hf_token.txt", tmp_path) == (None, None)


@pytest.mark.parametrize(
    "text",
    [
        "﻿hf_abc123XYZ\r\n",
        "HF_TOKEN=hf_abc123XYZ\n",
        '"hf_abc123XYZ"',
        "  hf_abc123XYZ  \n\n",
    ],
)
def test_token_file_variants_are_normalized(text):
    assert runner.normalize_token_text(text) == "hf_abc123XYZ"
    assert runner.token_shape(runner.normalize_token_text(text)) == {
        "present": True,
        "length": 12,
        "starts_with_hf_": True,
        "has_whitespace": False,
    }
