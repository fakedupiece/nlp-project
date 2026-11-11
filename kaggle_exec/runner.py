"""RuleGate Kaggle runner.

This file is the template for the script kernel that ``kaggle_exec/controller.py``
pushes with ``kaggle kernels push``. The controller replaces the run-config
sentinel below with a JSON object and uploads the result as the kernel's code
file. The runner contains no research logic: it clones the declared RuleGate Git
commit and the pinned official benchmark, installs them, and invokes
``python -m rulegate.stage2`` exactly as a local user would.

Modes:

* ``probe``      - environment, GPU, clone, install, CPU tests, secret and model
                   access checks. No model weights are loaded.
* ``smoke-gpu``  - ``probe`` plus a real-model diagnostic run on the first N
                   design requests (``rulegate.stage2 smoke-gpu``).
* ``run-design`` - ``probe`` plus the strict 18-request Stage 2 baseline run
                   (``rulegate.stage2 run-design``).

There is no locked-test mode. ``status.json`` in the output directory is the
authoritative outcome; the process exits 0 even on failure so that Kaggle keeps
the diagnostic output files. Credentials never leave memory: the Hugging Face
token is read from Kaggle Secrets, passed only to the child process
environment, and redacted from every line this runner writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Callable


RUN_CONFIG_JSON = r"""__RULEGATE_RUN_CONFIG__"""

MODES = ("probe", "smoke-gpu", "run-design")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/rulegate")
DEFAULT_SCRATCH_ROOT = Path("/tmp/rulegate")
WATCHED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
    "pytest",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(raw: str = RUN_CONFIG_JSON) -> dict[str, Any]:
    """Parse and validate the injected run configuration."""

    if raw.strip() == "__RULEGATE_RUN_CONFIG__":
        raise ValueError("run config was not injected; use kaggle_exec/controller.py")
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("run config must be a JSON object")
    mode = config.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    for key in ("code_commit", "benchmark_commit"):
        if not COMMIT_RE.fullmatch(str(config.get(key, ""))):
            raise ValueError(f"{key} must be a lowercase 40-character commit")
    revision = config.get("model_revision")
    if mode == "run-design" and not COMMIT_RE.fullmatch(str(revision or "")):
        raise ValueError("run-design requires an exact 40-character model_revision")
    if revision not in (None, "resolve") and not COMMIT_RE.fullmatch(str(revision)):
        raise ValueError("model_revision must be a 40-character commit or 'resolve'")
    if mode == "smoke-gpu":
        limit = config.get("limit")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 18:
            raise ValueError("smoke-gpu requires an integer limit from 1 through 18")
    for key in ("repo_url", "benchmark_url", "model_id", "run_id"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be a non-empty string")
    if config.get("quantization", "none") not in {"none", "4bit"}:
        raise ValueError("quantization must be 'none' or '4bit'")
    if config.get("dtype", "auto") not in {"auto", "float16", "bfloat16"}:
        raise ValueError("dtype must be 'auto', 'float16' or 'bfloat16'")
    return config


class Redactor:
    """Replace registered secret values in any text before it is written."""

    def __init__(self) -> None:
        self._secrets: list[str] = []

    def add(self, secret: str | None) -> None:
        if secret and len(secret) >= 6 and secret not in self._secrets:
            self._secrets.append(secret)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        return text


class StatusWriter:
    """Persist stage-by-stage progress so partial runs remain diagnosable."""

    def __init__(self, path: Path, config: dict[str, Any], redact: Redactor) -> None:
        self.path = path
        self.redact = redact
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": config["run_id"],
            "mode": config["mode"],
            "code_commit": config["code_commit"],
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "final_status": "running",
            "blocker": None,
            "stages": [],
            "facts": {},
        }
        self.flush()

    def stage(self, name: str, status: str, seconds: float, detail: str | None = None) -> None:
        self.payload["stages"].append(
            {
                "name": name,
                "status": status,
                "seconds": round(seconds, 3),
                "detail": self.redact(detail) if detail else None,
            }
        )
        print(f"RULEGATE_STAGE {name} {status} {seconds:.1f}s", flush=True)
        self.flush()

    def fact(self, key: str, value: Any) -> None:
        self.payload["facts"][key] = value
        self.flush()

    def finish(self, final_status: str, blocker: str | None = None) -> None:
        self.payload["final_status"] = final_status
        self.payload["blocker"] = self.redact(blocker) if blocker else None
        self.payload["finished_at_utc"] = utc_now()
        self.flush()
        print(f"RULEGATE_FINAL_STATUS={final_status}", flush=True)
        if blocker:
            print(f"RULEGATE_BLOCKER={self.redact(blocker)}", flush=True)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, indent=2, sort_keys=True) + "\n")


class StageFailed(RuntimeError):
    """A stage failed; ``blocker`` is a sanitized, actionable explanation."""

    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def run_command(
    command: list[str],
    *,
    redact: Redactor,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    echo: bool = True,
) -> tuple[int, str]:
    """Run a command, streaming redacted output to stdout and an optional log."""

    shown = " ".join(part if "\n" not in part else "<inline code>" for part in command)
    print("$ " + redact(shown), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(cwd) if cwd else None,
        bufsize=1,
    )
    lines: list[str] = []
    log_handle = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            clean = redact(line.rstrip("\n"))
            lines.append(clean)
            if echo:
                print(clean, flush=True)
            if log_handle:
                log_handle.write(clean + "\n")
                log_handle.flush()
    finally:
        if log_handle:
            log_handle.close()
    return process.wait(), "\n".join(lines)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in WATCHED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


GPU_PROBE_CODE = r"""
import json
try:
    import torch
except Exception as exc:
    print(json.dumps({"torch_import_error": type(exc).__name__}))
    raise SystemExit(0)
info = {
    "torch_version": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "devices": [],
}
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        info["devices"].append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": f"{major}.{minor}",
        })
    try:
        info["bf16_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        info["bf16_native"] = None
print(json.dumps(info))
"""


def probe_environment(redact: Redactor) -> dict[str, Any]:
    """Collect GPU/runtime facts in a child process (keeps this process CUDA-free)."""

    facts: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages_preinstalled": package_versions(),
    }
    if shutil.which("nvidia-smi"):
        code, output = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            redact=redact,
        )
        facts["nvidia_smi"] = output.splitlines() if code == 0 else f"exit {code}"
    else:
        facts["nvidia_smi"] = None
    code, output = run_command(
        [sys.executable, "-c", GPU_PROBE_CODE], redact=redact, echo=False
    )
    try:
        facts["torch"] = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        facts["torch"] = {"probe_exit_code": code}
    for label, path in (("tmp", "/tmp"), ("working", "/kaggle/working")):
        if os.path.isdir(path):
            facts[f"{label}_disk_free_bytes"] = shutil.disk_usage(path).free
    return facts


def git_clone_at(url: str, commit: str, destination: Path, redact: Redactor) -> str:
    if destination.exists():
        shutil.rmtree(destination)
    code, _ = run_command(
        ["git", "-c", "core.autocrlf=false", "clone", "--quiet", url, str(destination)],
        redact=redact,
    )
    if code != 0:
        raise StageFailed(f"git clone failed for {url}; is Internet enabled for the kernel?")
    code, _ = run_command(
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", commit],
        redact=redact,
    )
    if code != 0:
        raise StageFailed(f"commit {commit} is not reachable in {url}; push it first")
    head = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != commit:
        raise StageFailed(f"checkout HEAD {head} does not match {commit}")
    return head


def read_hf_secret(name: str) -> str | None:
    """Return the Kaggle secret value or None. Never prints the value."""

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]
    except ImportError:
        return os.environ.get(name) or None
    try:
        value = UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001 - any failure means "not attached"
        return None
    return value or None


def check_model_access(
    model_id: str,
    revision: str | None,
    token: str,
    cache_dir: Path,
) -> dict[str, Any]:
    """Resolve the model commit and prove file-level gated access via config.json."""

    from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-not-found]

    info = HfApi().model_info(model_id, revision=revision, token=token)
    resolved = str(info.sha).lower()
    if revision and revision != resolved:
        raise StageFailed(f"model revision {revision} resolved to {resolved}")
    config_path = hf_hub_download(
        model_id,
        "config.json",
        revision=resolved,
        token=token,
        cache_dir=str(cache_dir),
    )
    model_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "gated": getattr(info, "gated", None),
        "config_torch_dtype": model_config.get("torch_dtype"),
        "config_architectures": model_config.get("architectures"),
    }


def classify_hf_error(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "GatedRepoError" or "gated" in str(exc).lower():
        return (
            "Hugging Face denied gated access to the model. Accept the Llama 3.1 "
            "license on huggingface.co and wait for approval, and make sure the "
            "HF_TOKEN secret belongs to that account."
        )
    if name in {"RepositoryNotFoundError", "RevisionNotFoundError"}:
        return f"Hugging Face could not resolve the model or revision ({name})."
    if "401" in str(exc) or name == "HfHubHTTPError":
        return f"Hugging Face rejected the token ({name}); regenerate HF_TOKEN."
    return f"model access check failed ({name})"


def main(
    config: dict[str, Any] | None = None,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    secret_reader: Callable[[str], str | None] = read_hf_secret,
) -> int:
    redact = Redactor()
    try:
        config = config or load_config()
    except Exception as exc:  # noqa: BLE001
        print(f"RULEGATE_FINAL_STATUS=config_error {type(exc).__name__}: {exc}", flush=True)
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    status = StatusWriter(output_root / "status.json", config, redact)
    mode = config["mode"]
    token: str | None = None

    def timed(name: str, func: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            value = func()
        except StageFailed as exc:
            status.stage(name, "failed", time.perf_counter() - started, exc.blocker)
            raise
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            status.stage(name, "failed", time.perf_counter() - started, detail)
            raise StageFailed(f"stage {name} raised {type(exc).__name__}") from exc
        status.stage(name, "ok", time.perf_counter() - started)
        return value

    try:
        environment = timed("environment", lambda: probe_environment(redact))
        (output_root / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )
        torch_facts = environment.get("torch") or {}
        status.fact("cuda_available", torch_facts.get("cuda_available"))
        status.fact("gpus", torch_facts.get("devices"))
        if mode != "probe" and not torch_facts.get("cuda_available"):
            raise StageFailed("no CUDA GPU is attached; push with --accelerator NvidiaTeslaT4")

        repo = scratch_root / "nlp-project"
        benchmark = scratch_root / "policy-compliance-requests-AI"
        timed(
            "clone_code",
            lambda: git_clone_at(config["repo_url"], config["code_commit"], repo, redact),
        )
        timed(
            "clone_benchmark",
            lambda: git_clone_at(
                config["benchmark_url"], config["benchmark_commit"], benchmark, redact
            ),
        )

        def install() -> None:
            requirements = (
                "requirements-gpu-4bit.txt"
                if config.get("quantization") == "4bit"
                else "requirements-gpu.txt"
            )
            for command in (
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)],
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / requirements)],
                [sys.executable, "-m", "pip", "install", "-q", "pytest>=8.0"],
            ):
                code, _ = run_command(command, redact=redact)
                if code != 0:
                    raise StageFailed(f"pip install failed: {' '.join(command[3:])}")
            status.fact("packages_after_install", package_versions())

        timed("install", install)

        def cpu_tests() -> None:
            code, output = run_command(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                redact=redact,
                cwd=repo,
                echo=False,
                log_path=output_root / "cpu_tests.log",
            )
            tail = output.strip().splitlines()[-1] if output.strip() else ""
            status.fact("cpu_tests", tail)
            if code != 0:
                raise StageFailed(f"remote CPU tests failed: {tail}")

        timed("cpu_tests", cpu_tests)

        secret_name = config.get("hf_secret_name", "HF_TOKEN")
        token = secret_reader(secret_name)
        redact.add(token)
        status.fact("hf_secret_available", bool(token))
        if not token:
            blocker = (
                f"Kaggle secret {secret_name} is not attached to this notebook. Open "
                "the kernel in the Kaggle editor, Add-ons > Secrets, add "
                f"{secret_name} and enable it for this notebook."
            )
            status.stage("hf_secret", "failed", 0.0, blocker)
            if mode == "probe":
                status.finish("probe_incomplete", blocker)
                return 0
            raise StageFailed(blocker)
        status.stage("hf_secret", "ok", 0.0)

        requested = config.get("model_revision")
        requested = None if requested in (None, "resolve") else requested
        hf_cache = scratch_root / "hf-cache"
        try:
            access = timed(
                "model_access",
                lambda: check_model_access(config["model_id"], requested, token, hf_cache),
            )
        except StageFailed as exc:
            cause = exc.__cause__ or exc
            raise StageFailed(classify_hf_error(cause)) from exc
        status.fact("model_access", access)
        if mode == "probe":
            status.finish("ok")
            return 0

        artifacts = output_root / "artifacts"
        raw_file = output_root / "raw" / "raw_responses.jsonl"
        command = [
            sys.executable,
            "-m",
            "rulegate.stage2",
            mode,
            "--data-root",
            str(benchmark),
            "--output-dir",
            str(artifacts),
            "--raw-response-file",
            str(raw_file),
            "--code-commit",
            config["code_commit"],
            "--model-id",
            config["model_id"],
            "--model-revision",
            access["resolved_revision"],
            "--quantization",
            config.get("quantization", "none"),
            "--dtype",
            config.get("dtype", "auto"),
            "--max-new-tokens",
            str(config.get("max_new_tokens", 128)),
            "--max-attempts",
            str(config.get("max_attempts", 1)),
            "--seed",
            str(config.get("seed", 484654)),
        ]
        if mode == "smoke-gpu":
            command += ["--limit", str(config["limit"])]
        child_env = dict(os.environ)
        child_env.update(
            HF_TOKEN=token,
            HF_HOME=str(hf_cache.parent / "hf-home"),
            HF_HUB_CACHE=str(hf_cache),
            HF_HUB_DISABLE_TELEMETRY="1",
            PYTHONUNBUFFERED="1",
            TOKENIZERS_PARALLELISM="false",
        )

        def stage2() -> int:
            code, _ = run_command(
                command,
                redact=redact,
                env=child_env,
                cwd=repo,
                log_path=output_root / "stage2.log",
            )
            status.fact("stage2_exit_code", code)
            if code == 3 and mode == "smoke-gpu":
                return code
            if code != 0:
                raise StageFailed(f"rulegate.stage2 {mode} exited with code {code}")
            return code

        exit_code = timed("stage2_" + mode.replace("-", "_"), stage2)
        child_env.pop("HF_TOKEN", None)
        if exit_code == 3:
            status.finish(
                "completed_with_parse_failures",
                "some model outputs did not parse; see artifacts/diagnostic_outcomes.json",
            )
        else:
            status.finish("ok")
    except StageFailed as exc:
        status.finish("failed", exc.blocker)
    except BaseException as exc:  # noqa: BLE001 - always leave a status file
        status.finish("failed", f"runner crashed: {type(exc).__name__}")
        print(redact(traceback.format_exc()), flush=True)
    finally:
        token = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
