"""Local controller for RuleGate Kaggle GPU runs.

Submits ``kaggle_exec/runner.py`` as a private script kernel with the installed
Kaggle CLI, waits for the remote session, downloads its outputs and writes a
short report. A run is never considered successful because the upload worked or
because Kaggle reports ``COMPLETE``: the runner's own ``status.json`` (written
inside the remote session) and the downloaded artifacts are authoritative.

Usage (from the repository root, inside the project virtualenv)::

    python kaggle_exec/controller.py doctor
    python kaggle_exec/controller.py run --mode probe
    python kaggle_exec/controller.py run --mode smoke-gpu --limit 2 --model-revision <sha>
    python kaggle_exec/controller.py run --mode run-design --model-revision <sha>
    python kaggle_exec/controller.py status
    python kaggle_exec/controller.py fetch --run-id <id>
    python kaggle_exec/controller.py report --run-id <id>

The code commit must already be on ``origin`` because the remote session clones
it from GitHub. Local run folders live in the git-ignored
``results/dev/runs/kaggle``; reviewed reports go to ``results/dev/reports`` and
one line per run is appended to ``results/dev/kaggle_run_history.jsonl``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_TEMPLATE = Path(__file__).resolve().parent / "runner.py"
RUNS_ROOT = REPO_ROOT / "results" / "dev" / "runs" / "kaggle"
REPORTS_ROOT = REPO_ROOT / "results" / "dev" / "reports"
HISTORY_FILE = REPO_ROOT / "results" / "dev" / "kaggle_run_history.jsonl"

KERNEL_SLUG = "rulegate-runner"
KERNEL_TITLE = "RuleGate Runner"
CODE_FILE = "rulegate_runner.py"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
# Kaggle T4 (compute capability 7.5) has no native bfloat16; float16 across the
# two T4s holds the unquantized 8B model (~16 GB of weights in ~30 GB of VRAM).
DEFAULT_DTYPE = "float16"
DEFAULT_REPO_URL = "https://github.com/fakedupiece/nlp-project.git"
BENCHMARK_URL = "https://github.com/PedroCV/policy-compliance-requests-AI.git"
TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}
ACTIVE_STATES = {"QUEUED", "RUNNING", "NEW_SCRIPT"}
SENTINEL = '"""__RULEGATE_RUN_CONFIG__"""'
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_TIMEOUTS = {"probe": 1800, "smoke-gpu": 3600, "run-design": 7200}


class ControllerError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Small process helpers


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def kaggle_executable() -> str:
    scripts = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        candidate = scripts / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("kaggle")
    if not found:
        raise ControllerError("Kaggle CLI not found; pip install kaggle in the venv")
    return found


def run_process(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    check: bool = False,
    timeout: float | None = 600,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    result = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise ControllerError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def kaggle(*args: str, timeout: float | None = 600) -> subprocess.CompletedProcess[str]:
    return run_process([kaggle_executable(), *args], timeout=timeout)


def git(*args: str, check: bool = True) -> str:
    return run_process(["git", *args], check=check).stdout.strip()


# --------------------------------------------------------------------------
# Kaggle account, kernel and commit checks


def kaggle_username() -> str:
    """Return the authenticated Kaggle username (never prints credentials)."""

    code = (
        "import sys\n"
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api = KaggleApi()\n"
        "try:\n"
        "    api.authenticate()\n"
        "except SystemExit:\n"
        "    sys.exit(3)\n"
        "print('KAGGLE_USER=' + str(api.config_values.get('username', '')))\n"
    )
    result = run_process([sys.executable, "-c", code], timeout=120)
    match = re.search(r"KAGGLE_USER=(\S+)", result.stdout)
    if result.returncode != 0 or not match:
        raise ControllerError(
            "Kaggle CLI is not authenticated. Run `kaggle auth login` (browser "
            "OAuth) or save an API token to ~/.kaggle/access_token, then retry."
        )
    return match.group(1)


def parse_status(output: str) -> tuple[str | None, str | None]:
    """Parse ``kaggle kernels status`` output into (STATE, failure message)."""

    match = re.search(r'has status "([^"]+)"', output)
    if not match:
        return None, None
    state = match.group(1).split(".")[-1].strip().upper()
    failure = re.search(r'Failure message: "(.*)"', output)
    return state, failure.group(1) if failure else None


def kernel_status(ref: str) -> tuple[str | None, str | None, str]:
    result = kaggle("kernels", "status", ref, timeout=120)
    text = (result.stdout or "") + (result.stderr or "")
    state, failure = parse_status(text)
    return state, failure, text.strip()


def parse_push_output(output: str) -> tuple[int | None, str | None]:
    if "Kernel push error" in output:
        raise ControllerError(output.strip())
    match = re.search(r"Kernel version (\d+) successfully pushed", output)
    url = re.search(r"(https://www\.kaggle\.com/\S+)", output)
    if not match:
        raise ControllerError(f"push did not report a pushed version:\n{output.strip()}")
    return int(match.group(1)), url.group(1) if url else None


def resolve_commit(ref: str) -> str:
    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}").lower()
    if not COMMIT_RE.fullmatch(sha):
        raise ControllerError(f"could not resolve {ref} to a commit")
    return sha


def ensure_commit_on_origin(sha: str) -> list[str]:
    run_process(["git", "fetch", "--quiet", "origin"], timeout=300)
    branches = [
        line.strip()
        for line in git("branch", "-r", "--contains", sha, check=False).splitlines()
        if line.strip()
    ]
    if not branches:
        raise ControllerError(
            f"commit {sha} is not on origin; the Kaggle session clones from GitHub. "
            "Push it (e.g. git push origin HEAD) and retry."
        )
    return branches


# --------------------------------------------------------------------------
# Kernel bundle


def build_run_config(args: argparse.Namespace, code_commit: str, run_id: str) -> dict[str, Any]:
    from rulegate.stage2 import BENCHMARK_COMMIT, DEFAULT_MODEL_ID, DEFAULT_SEED

    revision = args.model_revision
    if args.mode == "run-design" and not COMMIT_RE.fullmatch(revision or ""):
        raise ControllerError("run-design requires --model-revision <40-char HF commit>")
    if revision not in (None, "resolve") and not COMMIT_RE.fullmatch(revision):
        raise ControllerError("--model-revision must be a 40-char commit or 'resolve'")
    config: dict[str, Any] = {
        "run_id": run_id,
        "mode": args.mode,
        "repo_url": args.repo_url,
        "code_commit": code_commit,
        "benchmark_url": BENCHMARK_URL,
        "benchmark_commit": BENCHMARK_COMMIT,
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": revision or "resolve",
        "quantization": args.quantization,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "max_attempts": args.max_attempts,
        "seed": DEFAULT_SEED,
        "hf_secret_name": "HF_TOKEN",
    }
    if args.mode == "smoke-gpu":
        if not 1 <= args.limit <= 18:
            raise ControllerError("--limit must be between 1 and 18 design requests")
        config["limit"] = args.limit
    return config


def render_runner(config: dict[str, Any]) -> str:
    template = RUNNER_TEMPLATE.read_text(encoding="utf-8")
    if template.count(SENTINEL) != 1:
        raise ControllerError("runner template must contain the config sentinel once")
    payload = json.dumps(config, sort_keys=True)
    if '"""' in payload or payload.endswith("\\"):
        raise ControllerError("run config cannot be embedded safely")
    # The template line is ``RUN_CONFIG_JSON = r"""<sentinel>"""``; the raw-string
    # prefix stays in place and only the quoted sentinel is replaced.
    return template.replace(SENTINEL, '"""' + payload + '"""', 1)


def kernel_metadata(owner: str, accelerator: str) -> dict[str, Any]:
    """Kernel settings; ``accelerator='none'`` gives a CPU session (probe only)."""

    metadata: dict[str, Any] = {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": accelerator != "none",
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if accelerator != "none":
        metadata["machine_shape"] = accelerator
    return metadata


def write_kernel_bundle(run_dir: Path, config: dict[str, Any], owner: str, accelerator: str) -> Path:
    kernel_dir = run_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=False)
    rendered = render_runner(config)
    (kernel_dir / CODE_FILE).write_text(rendered, encoding="utf-8", newline="\n")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(owner, accelerator), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    compile(rendered, CODE_FILE, "exec")
    return kernel_dir


# --------------------------------------------------------------------------
# Run records


def run_dir_for(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def load_record(run_id: str) -> dict[str, Any]:
    path = run_dir_for(run_id) / "run_record.json"
    if not path.is_file():
        raise ControllerError(f"no local run record for {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_record(record: dict[str, Any]) -> None:
    path = run_dir_for(record["run_id"]) / "run_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latest_run_id() -> str:
    if not RUNS_ROOT.is_dir():
        raise ControllerError("no local Kaggle runs yet")
    runs = sorted(p.name for p in RUNS_ROOT.iterdir() if (p / "run_record.json").is_file())
    if not runs:
        raise ControllerError("no local Kaggle runs yet")
    return runs[-1]


# --------------------------------------------------------------------------
# Wait, fetch, report


def wait_for_completion(record: dict[str, Any], *, interval: float, max_wait: float) -> str:
    ref = record["kernel_ref"]
    started = time.monotonic()
    last_state = None
    while True:
        state, failure, text = kernel_status(ref)
        if state != last_state:
            print(f"[{_utc_now()}] {ref}: {state or text}", flush=True)
            record.setdefault("status_history", []).append(
                {"at": _utc_now(), "state": state, "failure": failure}
            )
            save_record(record)
            last_state = state
        if state in TERMINAL_STATES:
            record["kaggle_status"] = state
            record["kaggle_failure_message"] = failure
            save_record(record)
            return state
        if time.monotonic() - started > max_wait:
            record["kaggle_status"] = f"WAIT_TIMEOUT_{state}"
            save_record(record)
            return record["kaggle_status"]
        time.sleep(interval)


def fetch_outputs(record: dict[str, Any]) -> Path:
    output_dir = run_dir_for(record["run_id"]) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = record["kernel_ref"]
    if record.get("kernel_version"):
        ref = f"{ref}/{record['kernel_version']}"
    result = kaggle("kernels", "output", ref, "-p", str(output_dir), "-o", "-q", timeout=1800)
    if result.returncode != 0:
        # Older versions may not accept the version suffix; fall back to latest.
        result = kaggle(
            "kernels", "output", record["kernel_ref"], "-p", str(output_dir), "-o", "-q",
            timeout=1800,
        )
    record["fetch_exit_code"] = result.returncode
    record["fetch_message"] = (result.stdout + result.stderr).strip()[-2000:]
    logs = kaggle("kernels", "logs", ref, timeout=600)
    if logs.returncode == 0 and logs.stdout.strip():
        (output_dir / "kernel_logs.txt").write_text(logs.stdout, encoding="utf-8")
    save_record(record)
    return output_dir


def _find(output_dir: Path, name: str) -> Path | None:
    matches = sorted(output_dir.rglob(name))
    return matches[0] if matches else None


def _load_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _gib(value: Any) -> str:
    return f"{int(value) / 1024**3:.1f} GiB" if isinstance(value, (int, float)) else "n/a"


def _method_rows(artifacts: Path) -> list[dict[str, Any]]:
    rows = []
    for label, filename in (
        ("Single Prompting", "single_calls.jsonl"),
        ("Full Sequential with Both", "full_sequential_calls.jsonl"),
    ):
        calls = _load_jsonl(artifacts / filename)
        if not calls:
            continue
        requests = {call["request_id"] for call in calls}
        known = [c for c in calls if c.get("input_tokens") is not None]
        elapsed = sum(float(c.get("elapsed_seconds") or 0.0) for c in calls)
        rows.append(
            {
                "method": label,
                "requests": len(requests),
                "calls": len(calls),
                "calls_per_request": len(calls) / len(requests),
                "input_tokens": sum(int(c["input_tokens"]) for c in known),
                "output_tokens": sum(int(c["output_tokens"] or 0) for c in known),
                "latency_seconds": elapsed,
                "mean_latency_seconds": elapsed / len(calls),
                "failed_calls": sum(1 for c in calls if not c.get("success")),
                "parse_status": dict(sorted(Counter(c["parse_status"] for c in calls).items())),
            }
        )
    return rows


def _metric_lines(summary: dict[str, Any]) -> list[str]:
    lines = []
    for key, label in (
        ("single_prompting_metrics", "Single Prompting"),
        ("full_sequential_both_metrics", "Full Sequential with Both"),
        ("single_prompting_subset_metrics", "Single Prompting (subset only)"),
        ("full_sequential_both_subset_metrics", "Full Sequential with Both (subset only)"),
    ):
        metrics = summary.get(key)
        if not metrics:
            continue
        policy = metrics["policy"]
        requests = metrics["requests"]
        per_policy = ", ".join(
            f"{p['policy_id']}:{p['f1']:.2f}" for p in policy["policies"]
        )
        lines.append(
            f"| {label} | {policy['mean_f1']:.4f} | {requests['exact_match_accuracy']:.4f} "
            f"| {requests['compliant_recall']:.4f} "
            f"| {requests['violating_request_detection_recall']:.4f} | {per_policy} |"
        )
    return lines


def build_report(record: dict[str, Any], output_dir: Path) -> tuple[str, dict[str, Any]]:
    """Return (markdown report, compact history entry). No raw model text."""

    status = _load_json(_find(output_dir, "status.json")) or {}
    environment = _load_json(_find(output_dir, "environment.json")) or {}
    artifacts_summary_path = _find(output_dir, "summary.json")
    artifacts = artifacts_summary_path.parent if artifacts_summary_path else None
    summary = _load_json(artifacts_summary_path) or {}
    metadata = _load_json(artifacts / "run_metadata.json") if artifacts else None
    model = (metadata or {}).get("model") or {}
    facts = status.get("facts", {})

    if status and status.get("run_id") != record["run_id"]:
        runner_state = "MISMATCHED_OUTPUT"
    else:
        runner_state = status.get("final_status", "missing_status_json")
    lines = [
        f"# Kaggle run `{record['run_id']}`",
        "",
        f"- Mode: `{record['mode']}`; classification: "
        f"`{summary.get('classification', 'n/a')}`; locked test used: "
        f"`{summary.get('locked_test_used', False)}`",
        f"- Code commit: `{record['code_commit']}`",
        f"- Kernel: `{record['kernel_ref']}` version `{record.get('kernel_version')}`; "
        f"accelerator requested `{record['accelerator']}`",
        f"- Kaggle status: `{record.get('kaggle_status')}`"
        + (
            f" (failure message: {record['kaggle_failure_message']})"
            if record.get("kaggle_failure_message")
            else ""
        ),
        f"- Runner final status: **`{runner_state}`**",
    ]
    if status.get("blocker"):
        lines.append(f"- Blocker: {status['blocker']}")
    lines += ["", "## Stages", "", "| Stage | Status | Seconds | Detail |", "| --- | --- | ---: | --- |"]
    for stage in status.get("stages", []):
        detail = (stage.get("detail") or "").replace("|", "/").replace("\n", " ")[:300]
        lines.append(f"| {stage['name']} | {stage['status']} | {stage['seconds']:.1f} | {detail} |")

    torch_facts = environment.get("torch") or {}
    packages = facts.get("packages_after_install") or environment.get("packages_preinstalled") or {}
    lines += ["", "## Runtime", ""]
    for device in torch_facts.get("devices") or []:
        lines.append(
            f"- GPU {device['index']}: {device['name']}, {_gib(device['total_memory_bytes'])}, "
            f"compute capability {device['compute_capability']}"
        )
    if not torch_facts.get("devices"):
        lines.append(f"- CUDA available: `{torch_facts.get('cuda_available')}`")
    lines.append(
        f"- torch `{torch_facts.get('torch_version')}` (CUDA build "
        f"`{torch_facts.get('torch_cuda_build')}`); native bf16: `{torch_facts.get('bf16_native')}`"
    )
    lines.append(
        "- Packages: "
        + ", ".join(f"{k} `{v}`" for k, v in sorted(packages.items()) if v is not None)
    )
    if facts.get("cpu_tests"):
        lines.append(f"- Remote CPU tests: `{facts['cpu_tests']}`")
    lines.append(f"- HF secret available: `{facts.get('hf_secret_available')}`")
    if facts.get("hf_secret_error"):
        lines.append(f"- HF secret lookup error: `{facts['hf_secret_error']}`")
    access = facts.get("model_access") or {}
    if access:
        lines.append(
            f"- Model access: `{access.get('model_id')}` resolved revision "
            f"`{access.get('resolved_revision')}` (checkpoint dtype "
            f"`{access.get('config_torch_dtype')}`)"
        )

    if model:
        device_counts = Counter(device for _, device in model.get("device_map") or [])
        lines += [
            "",
            "## Model configuration",
            "",
            f"- `{model.get('model_id')}` @ `{model.get('resolved_revision')}`",
            f"- quantization `{model.get('quantization')}`, requested dtype "
            f"`{model.get('requested_dtype')}`, loaded dtype `{model.get('dtype')}`",
            f"- greedy decoding (`do_sample={model.get('do_sample')}`, temperature "
            f"{model.get('temperature')}), max_new_tokens {model.get('max_new_tokens')}, "
            f"seed {model.get('seed')}",
            f"- load time {model.get('load_seconds') or 0:.1f}s; device map modules per device "
            f"{dict(device_counts)}; memory after load "
            + ", ".join(_gib(v) for v in model.get("memory_allocated_bytes_after_load") or []),
            f"- transformers `{model.get('transformers_version')}`, accelerate "
            f"`{model.get('accelerate_version')}`, torch `{model.get('torch_version')}`, "
            f"CUDA `{model.get('cuda_version')}`",
        ]

    rows = _method_rows(artifacts) if artifacts else []
    history_calls: dict[str, Any] = {}
    if rows:
        lines += [
            "",
            "## Measured calls",
            "",
            "| Method | Requests | Calls | Calls/request | Input tok | Output tok "
            "| Latency s | s/call | Failed calls | Parse status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row['method']} | {row['requests']} | {row['calls']} "
                f"| {row['calls_per_request']:.2f} | {row['input_tokens']} "
                f"| {row['output_tokens']} | {row['latency_seconds']:.1f} "
                f"| {row['mean_latency_seconds']:.2f} | {row['failed_calls']} "
                f"| {row['parse_status']} |"
            )
            history_calls[row["method"]] = {
                k: row[k] for k in ("requests", "calls", "input_tokens", "output_tokens",
                                    "latency_seconds", "failed_calls")
            }
    metric_lines = _metric_lines(summary)
    if metric_lines:
        lines += [
            "",
            "## Quality (Appendix D policy F1; design split)",
            "",
            "| Method | Mean policy F1 | Exact-match acc | Compliant recall "
            "| Violating detection recall | Per-policy F1 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            *metric_lines,
        ]
        if summary.get("subset_metrics_note"):
            lines += ["", f"Note: {summary['subset_metrics_note']}."]
    if "all_outputs_parsed" in summary:
        lines += ["", f"All outputs parsed: `{summary['all_outputs_parsed']}`"]
    gate_decision = None
    if artifacts and summary.get("classification") == "development_model":
        from rulegate.stage2_gate import evaluate_stage2_gate

        requested = (record.get("config") or {}).get("model_revision")
        gate = evaluate_stage2_gate(
            artifacts,
            expected_model_revision=requested if requested not in (None, "resolve") else None,
        )
        gate_decision = gate.decision
        lines += ["", f"## Stage 2 gate: **{gate.decision}**", ""]
        lines += [
            f"- {'PASS' if check.passed else 'FAIL'} `{check.name}`: {check.detail}"
            for check in gate.checks
        ]

    history = {
        "run_id": record["run_id"],
        "created_at_utc": record["created_at_utc"],
        "mode": record["mode"],
        "code_commit": record["code_commit"],
        "kernel_version": record.get("kernel_version"),
        "kaggle_status": record.get("kaggle_status"),
        "runner_final_status": runner_state,
        "blocker": status.get("blocker"),
        "gpus": [d.get("name") for d in torch_facts.get("devices") or []],
        "model_revision": model.get("resolved_revision") or access.get("resolved_revision"),
        "dtype": model.get("dtype"),
        "quantization": model.get("quantization"),
        "calls": history_calls,
        "all_outputs_parsed": summary.get("all_outputs_parsed"),
        "stage2_gate": gate_decision,
    }
    return "\n".join(lines) + "\n", history


def write_local_diagnosis(output_dir: Path, run_dir: Path) -> Path | None:
    """Write raw-response excerpts for parse failures to the git-ignored run dir."""

    raw = _load_jsonl(_find(output_dir, "raw_responses.jsonl"))
    summary_path = _find(output_dir, "summary.json")
    if not raw or summary_path is None:
        return None
    calls = []
    for filename in ("single_calls.jsonl", "full_sequential_calls.jsonl"):
        calls += _load_jsonl(summary_path.parent / filename)
    bad = {
        (c["request_id"], c["method"], c["policy_id"], c["attempt"], c["prompt_hash"])
        for c in calls
        if c["parse_status"] not in {"valid_single_response", "valid_verdict"}
    }
    lines = ["# Local diagnosis (contains raw model text; do not commit)", ""]
    for entry in raw:
        key = (entry["request_id"], entry["method"], entry["policy_id"], entry["attempt"],
               entry["prompt_hash"])
        if key in bad:
            lines.append(
                f"- `{entry['method']}` request `{entry['request_id']}` policy "
                f"`{entry['policy_id']}`: {json.dumps((entry.get('text') or '')[:400])}"
            )
    distinct = Counter((entry.get("text") or "").strip()[:60] for entry in raw)
    lines += ["", "## Most common response prefixes", ""]
    lines += [f"- {count} x {json.dumps(prefix)}" for prefix, count in distinct.most_common(10)]
    path = run_dir / "diagnosis.local.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def finalize(record: dict[str, Any]) -> dict[str, Any]:
    run_dir = run_dir_for(record["run_id"])
    output_dir = fetch_outputs(record)
    report, history = build_report(record, output_dir)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORTS_ROOT / f"{record['run_id']}.md").write_text(report, encoding="utf-8")
    diagnosis = write_local_diagnosis(output_dir, run_dir)
    record["runner_final_status"] = history["runner_final_status"]
    record["report"] = str((REPORTS_ROOT / f"{record['run_id']}.md").relative_to(REPO_ROOT))
    save_record(record)
    if not record.get("history_written"):
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(history, sort_keys=True) + "\n")
        record["history_written"] = True
        save_record(record)
    print(report)
    if diagnosis:
        print(f"Local raw-response diagnosis: {diagnosis}")
    return history


# --------------------------------------------------------------------------
# Commands


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print("git:", git("--version"))
    print("kaggle:", kaggle("--version").stdout.strip())
    try:
        user = kaggle_username()
    except ControllerError as exc:
        print(f"kaggle auth: NOT AUTHENTICATED - {exc}")
        return 2
    print(f"kaggle auth: OK as {user}")
    state, failure, text = kernel_status(f"{user}/{KERNEL_SLUG}")
    print(f"kernel {user}/{KERNEL_SLUG}: {state or 'not created yet'}")
    if failure:
        print(f"  failure message: {failure}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    code_commit = resolve_commit(args.commit)
    branches = ensure_commit_on_origin(code_commit)
    owner = kaggle_username()
    ref = f"{owner}/{KERNEL_SLUG}"
    state, _, _ = kernel_status(ref)
    if state in ACTIVE_STATES:
        raise ControllerError(f"{ref} is {state}; wait for it or cancel it first")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{args.mode}-{code_commit[:8]}"
    run_dir = run_dir_for(run_id)
    if args.accelerator == "none" and args.mode != "probe":
        raise ControllerError("--accelerator none is only allowed for --mode probe")
    config = build_run_config(args, code_commit, run_id)
    write_kernel_bundle(run_dir, config, owner, args.accelerator)
    record: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "mode": args.mode,
        "code_commit": code_commit,
        "code_commit_remote_branches": branches,
        "kernel_ref": ref,
        "accelerator": args.accelerator,
        "timeout_seconds": args.timeout,
        "config": config,
    }
    save_record(record)
    print(f"run {run_id}: pushing {ref} (commit {code_commit}, accelerator {args.accelerator})")
    push_args = ["kernels", "push", "-p", str(run_dir / "kernel"), "-t", str(args.timeout)]
    if args.accelerator != "none":
        push_args += ["--accelerator", args.accelerator]
    result = kaggle(*push_args, timeout=600)
    push_text = (result.stdout + result.stderr).strip()
    record["push_output"] = push_text[-2000:]
    save_record(record)
    if result.returncode != 0:
        raise ControllerError(f"kaggle kernels push failed:\n{push_text}")
    version, url = parse_push_output(push_text)
    record.update(kernel_version=version, kernel_url=url, pushed_at_utc=_utc_now())
    save_record(record)
    print(f"pushed version {version}: {url}")
    if args.no_wait:
        return 0
    final_state = wait_for_completion(
        record, interval=args.poll_seconds, max_wait=args.timeout + args.queue_grace
    )
    if final_state not in TERMINAL_STATES:
        print(f"stopped waiting: {final_state}; rerun `fetch --run-id {run_id}` later")
        return 4
    history = finalize(record)
    return 0 if history["runner_final_status"] == "ok" else 1


def cmd_status(args: argparse.Namespace) -> int:
    run_id = args.run_id or latest_run_id()
    record = load_record(run_id)
    state, failure, text = kernel_status(record["kernel_ref"])
    print(f"run {run_id} ({record['mode']}, version {record.get('kernel_version')}): {state or text}")
    if failure:
        print(f"failure message: {failure}")
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    record = load_record(args.run_id or latest_run_id())
    state = wait_for_completion(record, interval=args.poll_seconds, max_wait=args.max_wait)
    if state in TERMINAL_STATES:
        finalize(record)
        return 0
    return 4


def cmd_fetch(args: argparse.Namespace) -> int:
    record = load_record(args.run_id or latest_run_id())
    state, failure, _ = kernel_status(record["kernel_ref"])
    record["kaggle_status"] = state
    record["kaggle_failure_message"] = failure
    history = finalize(record)
    return 0 if history["runner_final_status"] == "ok" else 1


def cmd_report(args: argparse.Namespace) -> int:
    record = load_record(args.run_id or latest_run_id())
    report, _ = build_report(record, run_dir_for(record["run_id"]) / "output")
    print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check local tools and Kaggle authentication")

    run = sub.add_parser("run", help="push a run, wait, download outputs and report")
    run.add_argument("--mode", choices=("probe", "smoke-gpu", "run-design"), required=True)
    run.add_argument("--commit", default="HEAD", help="RuleGate commit (must be on origin)")
    run.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    run.add_argument("--model-revision", default=None,
                     help="40-char Hugging Face commit; smoke/probe may use 'resolve'")
    run.add_argument("--quantization", choices=("none", "4bit"), default="none")
    run.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default=DEFAULT_DTYPE)
    run.add_argument("--max-new-tokens", type=int, default=128)
    run.add_argument("--max-attempts", type=int, default=1)
    run.add_argument("--limit", type=int, default=2, help="smoke-gpu design subset size")
    run.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    run.add_argument("--timeout", type=int, default=None, help="kernel run limit (s)")
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--queue-grace", type=float, default=1800.0)
    run.add_argument("--no-wait", action="store_true")

    for name, help_text in (
        ("status", "show Kaggle status for a run (default: latest)"),
        ("wait", "wait for a pushed run, then fetch and report"),
        ("fetch", "download outputs and write the report"),
        ("report", "rebuild the report from downloaded outputs"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--run-id", default=None)
        if name == "wait":
            command.add_argument("--poll-seconds", type=float, default=30.0)
            command.add_argument("--max-wait", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "timeout", None) is None and args.command == "run":
        args.timeout = DEFAULT_TIMEOUTS[args.mode]
    handlers = {
        "doctor": cmd_doctor,
        "run": cmd_run,
        "status": cmd_status,
        "wait": cmd_wait,
        "fetch": cmd_fetch,
        "report": cmd_report,
    }
    try:
        return handlers[args.command](args)
    except ControllerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
