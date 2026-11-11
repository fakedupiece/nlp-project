"""Stage 2 design-split baseline runner and artifact writer.

The command surface deliberately exposes only the 18-request design split. The
225-request locked split has no Stage 2 execution route.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Sequence

from .accounting import summarize_calls
from .baselines import (
    SinglePromptingBaseline,
    SinglePromptingError,
    make_sequential_both_verifier,
)
from .data import load_benchmark
from .metrics import evaluate
from .model_backend import (
    CallLedger,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackend,
    RawBackendResponse,
    RawResponseRecorder,
)
from .verification import VerificationError
from .prompts import (
    SEQUENTIAL_BOTH_TEMPLATE_ID,
    SEQUENTIAL_BOTH_TEMPLATE_SHA256,
    SINGLE_PROMPT_TEMPLATE_ID,
    SINGLE_PROMPT_TEMPLATE_SHA256,
)
from .schemas import POLICY_IDS, BenchmarkDataset
from .transformers_backend import TransformersBackend


BENCHMARK_COMMIT = "2d1944f2efa362021f931638b724714f7b74f288"
BENCHMARK_DATA_FILE_SHA256 = {
    "policy_list/policy_list.txt": "a993b8e087df1208ad6a15f8b095093543be2e8fd324590a04e500a92d12256b",
    "testing_requests/test_user_requests.txt": "254c8278be74aa597c2eb2f9029da0dc5580058ca8764cfbd8fdb9e9bdf732d6",
    "user_requests/part_1.txt": "a4ae03af6576df22a4910fe4d66171469720aaf0892b6ee707d31876a42fbdef",
    "user_requests/part_2.txt": "6bbbe1213499f762644ed12505f05652094529c9998918803318e5d61a5aed69",
    "user_requests/part_3.txt": "39e771216f4d91755d96461cf140f9f368238a9afeebdfb6510347c463279b55",
    "user_requests/part_4.txt": "152291c00c652c017eb6323b985790081c6fd0760e7a8a994bcb756c6444d4eb",
    "user_requests/part_5.txt": "8a7bb45fff801775d5f4e622aacc5d351e6fcfb4b9035f63cf20615b7f0f7d0e",
    "user_requests/part_6.txt": "717d1cbf82891524a121387e270ef65cf9e7eb997cb3a7f4bce48ab80da92ecd",
    "user_requests/part_7.txt": "fe87a26e1c5362c895640adff67e7f20993b444109a91f03af15dd62b223b90b",
    "user_requests/part_8.txt": "b87873f98172b0066fc0821a02ad1e41970636329c0c3c045e34b47f175bc87a",
    "user_requests/part_9.txt": "2be203dc8d3cb2e297bf3bd8beec2cfdb1c4856f8013dc2cb6e6558b907f6237",
}
BENCHMARK_MANIFEST_SHA256 = sha256(
    json.dumps(
        BENCHMARK_DATA_FILE_SHA256,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SEED = 484654
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    request_id: str
    predicted_policy_ids: tuple[int, ...]
    call_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DesignBaselineRun:
    request_ids: tuple[str, ...]
    single_predictions: tuple[PredictionArtifact, ...]
    full_sequential_predictions: tuple[PredictionArtifact, ...]
    single_ledger: CallLedger
    full_sequential_ledger: CallLedger


ProgressCallback = Callable[[str], None]


def _no_progress(_: str) -> None:
    return None


def run_design_baselines(
    dataset: BenchmarkDataset,
    *,
    single_backend: InstrumentedBackend,
    full_sequential_backend: InstrumentedBackend,
    max_attempts: int = 1,
    progress: ProgressCallback = _no_progress,
) -> DesignBaselineRun:
    """Run both Stage 2 baselines over model-facing design records only."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    single = SinglePromptingBaseline(
        backend=single_backend,
        policies=dataset.policies,
    )
    sequential = make_sequential_both_verifier(
        backend=full_sequential_backend,
        policies=dataset.policies,
    )
    single_predictions: list[PredictionArtifact] = []
    full_predictions: list[PredictionArtifact] = []
    total = len(dataset.design_records)

    for index, trusted_record in enumerate(dataset.design_records, start=1):
        request = trusted_record.to_inference_request()
        single_result = single.predict_with_trace(
            request,
            max_attempts=max_attempts,
        )
        single_predictions.append(
            PredictionArtifact(
                request_id=request.request_id,
                predicted_policy_ids=single_result.predicted_policy_ids,
                call_ids=single_result.call_ids,
            )
        )
        progress(
            f"single_prompting {index}/{total} request={request.request_id} "
            f"calls={len(single_result.call_ids)}"
        )

    for index, trusted_record in enumerate(dataset.design_records, start=1):
        request = trusted_record.to_inference_request()
        full_result = sequential.verify_full(
            request,
            max_attempts=max_attempts,
        )
        full_predictions.append(
            PredictionArtifact(
                request_id=request.request_id,
                predicted_policy_ids=full_result.predicted_policy_ids,
                call_ids=full_result.call_ids,
            )
        )
        progress(
            f"full_sequential_both {index}/{total} request={request.request_id} "
            f"calls={len(full_result.call_ids)}"
        )

    return DesignBaselineRun(
        request_ids=tuple(record.request_id for record in dataset.design_records),
        single_predictions=tuple(single_predictions),
        full_sequential_predictions=tuple(full_predictions),
        single_ledger=single_backend.ledger,
        full_sequential_ledger=full_sequential_backend.ledger,
    )


@dataclass(frozen=True, slots=True)
class DiagnosticOutcome:
    """Per-request outcome of a diagnostic run; never contains model text."""

    request_id: str
    method: str
    status: str
    predicted_policy_ids: tuple[int, ...]
    call_ids: tuple[int, ...]
    failed_policy_ids: tuple[int, ...]
    error_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DesignDiagnosticRun:
    request_ids: tuple[str, ...]
    single_outcomes: tuple[DiagnosticOutcome, ...]
    full_sequential_outcomes: tuple[DiagnosticOutcome, ...]
    single_ledger: CallLedger
    full_sequential_ledger: CallLedger

    @property
    def all_ok(self) -> bool:
        return all(
            outcome.status == "ok"
            for outcome in (*self.single_outcomes, *self.full_sequential_outcomes)
        )


def _new_call_ids(ledger: CallLedger, start: int) -> tuple[int, ...]:
    return tuple(record.call_id for record in ledger.records[start:])


def run_design_diagnostic(
    dataset: BenchmarkDataset,
    *,
    single_backend: InstrumentedBackend,
    full_sequential_backend: InstrumentedBackend,
    limit: int,
    max_attempts: int = 1,
    progress: ProgressCallback = _no_progress,
) -> DesignDiagnosticRun:
    """Run both baselines on the first ``limit`` design records, never aborting.

    This is the GPU smoke-test path. It uses the same Single Prompting baseline
    and the same shared Sequential-with-Both verifier as ``run_design_baselines``
    and issues the same calls: one Single Prompting call per request and one
    verifier call per request-policy pair. The only difference is error
    handling. A parse or backend failure is recorded for that request (or that
    policy) and the run continues, so a single malformed response does not
    discard the evidence from an expensive model load. The 225 locked records
    are not reachable from this function.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    design_count = len(dataset.design_records)
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if not 1 <= limit <= design_count:
        raise ValueError(f"limit must be between 1 and {design_count}")
    records = dataset.design_records[:limit]
    single = SinglePromptingBaseline(backend=single_backend, policies=dataset.policies)
    sequential = make_sequential_both_verifier(
        backend=full_sequential_backend,
        policies=dataset.policies,
    )

    single_outcomes: list[DiagnosticOutcome] = []
    for index, trusted_record in enumerate(records, start=1):
        request = trusted_record.to_inference_request()
        start = len(single_backend.ledger.records)
        try:
            result = single.predict_with_trace(request, max_attempts=max_attempts)
        except SinglePromptingError as exc:
            outcome = DiagnosticOutcome(
                request_id=request.request_id,
                method="single_prompting",
                status="failed",
                predicted_policy_ids=(),
                call_ids=_new_call_ids(single_backend.ledger, start),
                failed_policy_ids=(),
                error_types=(type(exc).__name__,),
            )
        else:
            outcome = DiagnosticOutcome(
                request_id=request.request_id,
                method="single_prompting",
                status="ok",
                predicted_policy_ids=result.predicted_policy_ids,
                call_ids=result.call_ids,
                failed_policy_ids=(),
                error_types=(),
            )
        single_outcomes.append(outcome)
        progress(
            f"single_prompting {index}/{limit} request={request.request_id} "
            f"status={outcome.status} calls={len(outcome.call_ids)}"
        )

    full_outcomes: list[DiagnosticOutcome] = []
    for index, trusted_record in enumerate(records, start=1):
        request = trusted_record.to_inference_request()
        start = len(full_sequential_backend.ledger.records)
        predicted: list[int] = []
        failed: list[int] = []
        errors: list[str] = []
        for policy_id in POLICY_IDS:
            try:
                result = sequential.verify_candidates(
                    request,
                    (policy_id,),
                    max_attempts=max_attempts,
                )
            except VerificationError as exc:
                failed.append(policy_id)
                errors.append(type(exc).__name__)
                continue
            predicted.extend(result.predicted_policy_ids)
        outcome = DiagnosticOutcome(
            request_id=request.request_id,
            method="full_sequential_both",
            status="failed" if failed else "ok",
            predicted_policy_ids=tuple(predicted),
            call_ids=_new_call_ids(full_sequential_backend.ledger, start),
            failed_policy_ids=tuple(failed),
            error_types=tuple(errors),
        )
        full_outcomes.append(outcome)
        progress(
            f"full_sequential_both {index}/{limit} request={request.request_id} "
            f"status={outcome.status} calls={len(outcome.call_ids)} "
            f"failed_policies={list(outcome.failed_policy_ids)}"
        )

    return DesignDiagnosticRun(
        request_ids=tuple(record.request_id for record in records),
        single_outcomes=tuple(single_outcomes),
        full_sequential_outcomes=tuple(full_outcomes),
        single_ledger=single_backend.ledger,
        full_sequential_ledger=full_sequential_backend.ledger,
    )


def _safe_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("rulegate", "torch", "transformers", "accelerate", "bitsandbytes"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_state(repo_root: Path) -> dict[str, object]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    head = run("rev-parse", "--verify", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _require_clean_code_commit(code_commit: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    state = _git_state(repo_root)
    if state["commit"] is None:
        raise RuntimeError("run-design requires code checked out from a Git commit")
    if state["commit"] != code_commit:
        raise RuntimeError(
            f"declared code commit {code_commit} does not match checkout "
            f"{state['commit']}"
        )
    if state["dirty"] is not False:
        raise RuntimeError("run-design requires a clean checkout")


def _verify_benchmark_checkout(data_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(data_root), "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    observed = result.stdout.strip().lower()
    if observed != BENCHMARK_COMMIT:
        raise RuntimeError(
            f"benchmark checkout is {observed}; expected {BENCHMARK_COMMIT}"
        )
    return observed


def _verify_benchmark_files(data_root: Path) -> None:
    for relative_path, expected in BENCHMARK_DATA_FILE_SHA256.items():
        path = data_root / relative_path
        if not path.is_file():
            raise RuntimeError(f"benchmark file is missing: {relative_path}")
        observed = sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(
                f"benchmark file hash mismatch for {relative_path}: {observed}"
            )


def _prediction_payload(
    predictions: Sequence[PredictionArtifact],
) -> list[dict[str, object]]:
    return [
        {
            "request_id": prediction.request_id,
            "predicted_policy_ids": list(prediction.predicted_policy_ids),
            "call_ids": list(prediction.call_ids),
        }
        for prediction in predictions
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _require_new_output_dir(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    return destination


def write_failed_design_artifacts(
    *,
    output_dir: str | Path,
    single_ledger: CallLedger,
    full_sequential_ledger: CallLedger,
    error: Exception,
) -> None:
    """Preserve attempted-call evidence when a real design run aborts.

    The failure payload intentionally records only the exception type. Exception
    messages can contain backend text, so they are not safe experiment metadata.
    """

    destination = _require_new_output_dir(output_dir)
    destination.mkdir(parents=True)
    single_ledger.write_jsonl(destination / "single_calls.jsonl")
    full_sequential_ledger.write_jsonl(destination / "full_sequential_calls.jsonl")
    _write_json(
        destination / "failure.json",
        {
            "classification": "development_model",
            "split": "design",
            "locked_test_used": False,
            "status": "failed",
            "error_type": type(error).__name__,
            "single_prompting_attempted_calls": len(single_ledger.records),
            "full_sequential_both_attempted_calls": len(
                full_sequential_ledger.records
            ),
        },
    )


def write_design_artifacts(
    run: DesignBaselineRun,
    dataset: BenchmarkDataset,
    *,
    output_dir: str | Path,
    classification: str,
    model_metadata: dict[str, object],
    code_commit: str | None,
) -> dict[str, object]:
    """Write auditable design artifacts without raw benchmark request text."""

    if classification not in {"synthetic_smoke", "development_model"}:
        raise ValueError("classification must be synthetic_smoke or development_model")
    destination = _require_new_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=False)

    single_payload = _prediction_payload(run.single_predictions)
    full_payload = _prediction_payload(run.full_sequential_predictions)
    _write_json(destination / "single_predictions.json", single_payload)
    _write_json(destination / "full_sequential_predictions.json", full_payload)
    run.single_ledger.write_jsonl(destination / "single_calls.jsonl")
    run.full_sequential_ledger.write_jsonl(destination / "full_sequential_calls.jsonl")

    # This trusted design-only index stays separate from model-facing predictions.
    _write_json(
        destination / "trusted_design_index.json",
        [
            {
                "request_id": record.request_id,
                "gold_violated_policy_ids": list(record.gold_violated_policy_ids),
                "source_file": record.source_file,
                "source_index": record.source_index,
            }
            for record in dataset.design_records
        ],
    )

    single_summary = summarize_calls(run.single_ledger, run.request_ids)
    full_summary = summarize_calls(run.full_sequential_ledger, run.request_ids)
    summary: dict[str, object] = {
        "classification": classification,
        "split": "design",
        "locked_test_used": False,
        "request_count": len(run.request_ids),
        "single_prompting_calls": asdict(single_summary),
        "full_sequential_both_calls": asdict(full_summary),
    }
    if classification == "development_model":
        gold = [record.gold_violated_policy_ids for record in dataset.design_records]
        summary["single_prompting_metrics"] = asdict(
            evaluate(
                gold,
                [prediction.predicted_policy_ids for prediction in run.single_predictions],
            )
        )
        summary["full_sequential_both_metrics"] = asdict(
            evaluate(
                gold,
                [
                    prediction.predicted_policy_ids
                    for prediction in run.full_sequential_predictions
                ],
            )
        )
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "run_metadata.json",
        _run_metadata(
            classification=classification,
            model_metadata=model_metadata,
            code_commit=code_commit,
        ),
    )
    return summary


def _run_metadata(
    *,
    classification: str,
    model_metadata: dict[str, object],
    code_commit: str | None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    metadata: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "split": "design",
        "locked_test_used": False,
        "benchmark_commit": BENCHMARK_COMMIT,
        "benchmark_manifest_sha256": BENCHMARK_MANIFEST_SHA256,
        "code_commit_declared": code_commit,
        "local_git": _git_state(repo_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _safe_package_versions(),
        "model": model_metadata,
        "prompts": {
            "single": {
                "template_id": SINGLE_PROMPT_TEMPLATE_ID,
                "template_sha256": SINGLE_PROMPT_TEMPLATE_SHA256,
            },
            "full_sequential_both": {
                "template_id": SEQUENTIAL_BOTH_TEMPLATE_ID,
                "template_sha256": SEQUENTIAL_BOTH_TEMPLATE_SHA256,
            },
        },
    }
    if extra:
        metadata.update(extra)
    return metadata


def _parse_status_counts(ledger: CallLedger) -> dict[str, int]:
    return dict(sorted(Counter(record.parse_status for record in ledger.records).items()))


def write_diagnostic_artifacts(
    run: DesignDiagnosticRun,
    dataset: BenchmarkDataset,
    *,
    output_dir: str | Path,
    model_metadata: dict[str, object],
    code_commit: str | None,
    classification: str = "development_smoke",
) -> dict[str, object]:
    """Write smoke-test artifacts. Metrics are subset-only and not reportable."""

    if classification not in {"synthetic_smoke", "development_smoke"}:
        raise ValueError("classification must be synthetic_smoke or development_smoke")
    destination = _require_new_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    run.single_ledger.write_jsonl(destination / "single_calls.jsonl")
    run.full_sequential_ledger.write_jsonl(destination / "full_sequential_calls.jsonl")
    _write_json(
        destination / "diagnostic_outcomes.json",
        {
            "single_prompting": [asdict(o) for o in run.single_outcomes],
            "full_sequential_both": [asdict(o) for o in run.full_sequential_outcomes],
        },
    )
    selected = set(run.request_ids)
    trusted = [record for record in dataset.design_records if record.request_id in selected]
    _write_json(
        destination / "trusted_design_index.json",
        [
            {
                "request_id": record.request_id,
                "gold_violated_policy_ids": list(record.gold_violated_policy_ids),
                "source_file": record.source_file,
                "source_index": record.source_index,
            }
            for record in trusted
        ],
    )

    def outcome_counts(outcomes: Sequence[DiagnosticOutcome]) -> dict[str, int]:
        return dict(sorted(Counter(outcome.status for outcome in outcomes).items()))

    summary: dict[str, object] = {
        "classification": classification,
        "split": "design",
        "locked_test_used": False,
        "request_count": len(run.request_ids),
        "design_subset_limit": len(run.request_ids),
        "all_outputs_parsed": run.all_ok,
        "single_prompting_outcomes": outcome_counts(run.single_outcomes),
        "full_sequential_both_outcomes": outcome_counts(run.full_sequential_outcomes),
        "single_prompting_parse_status": _parse_status_counts(run.single_ledger),
        "full_sequential_both_parse_status": _parse_status_counts(
            run.full_sequential_ledger
        ),
        "single_prompting_calls": asdict(
            summarize_calls(run.single_ledger, run.request_ids)
        ),
        "full_sequential_both_calls": asdict(
            summarize_calls(run.full_sequential_ledger, run.request_ids)
        ),
    }
    if classification == "development_smoke" and run.all_ok:
        gold = [record.gold_violated_policy_ids for record in trusted]
        summary["subset_metrics_note"] = (
            "computed on the diagnostic design subset only; not a Stage 2 result"
        )
        summary["single_prompting_subset_metrics"] = asdict(
            evaluate(gold, [o.predicted_policy_ids for o in run.single_outcomes])
        )
        summary["full_sequential_both_subset_metrics"] = asdict(
            evaluate(gold, [o.predicted_policy_ids for o in run.full_sequential_outcomes])
        )
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "run_metadata.json",
        _run_metadata(
            classification=classification,
            model_metadata=model_metadata,
            code_commit=code_commit,
            extra={"design_subset_limit": len(run.request_ids)},
        ),
    )
    return summary


def _fake_smoke_backends(request_count: int) -> tuple[InstrumentedBackend, InstrumentedBackend]:
    settings = ModelSettings(
        model_id="fake/stage2-smoke",
        revision="synthetic",
        quantization="none",
        temperature=0.0,
        seed=DEFAULT_SEED,
    )
    single = InstrumentedBackend(
        FakeBackend(
            [RawBackendResponse('{"violated_policy_ids": []}') for _ in range(request_count)]
        ),
        settings,
        CallLedger(),
    )
    full = InstrumentedBackend(
        FakeBackend(
            [
                RawBackendResponse('{"violated": false}')
                for _ in range(request_count * 9)
            ]
        ),
        settings,
        CallLedger(),
    )
    return single, full


def _validated_commit(value: str, *, name: str) -> str:
    if not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{name} must be an exact 40-character hexadecimal commit")
    return value.lower()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run both baselines with a fake backend")
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)

    def add_real_model_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--code-commit", required=True)
        command.add_argument("--model-id", default=DEFAULT_MODEL_ID)
        command.add_argument("--model-revision", required=True)
        command.add_argument("--quantization", choices=("none", "4bit"), default="none")
        command.add_argument(
            "--dtype",
            choices=("auto", "float16", "bfloat16"),
            default="auto",
            help="weights/activation dtype; keep identical across compared methods",
        )
        command.add_argument("--max-new-tokens", type=int, default=128)
        command.add_argument("--max-attempts", type=int, default=1)
        command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        command.add_argument(
            "--raw-response-file",
            type=Path,
            default=None,
            help="optional private JSONL of raw model responses for diagnosis",
        )

    design = subparsers.add_parser(
        "run-design",
        help="run both baselines with the pinned model on the 18 design cases",
    )
    add_real_model_arguments(design)

    smoke_gpu = subparsers.add_parser(
        "smoke-gpu",
        help=(
            "diagnostic real-model run on the first N design cases; records "
            "parse failures instead of aborting"
        ),
    )
    add_real_model_arguments(smoke_gpu)
    smoke_gpu.add_argument("--limit", type=int, default=2)
    return parser


def _progress_to_stderr(message: str) -> None:
    print(f"[rulegate] {message}", file=sys.stderr, flush=True)


def _load_real_backend(
    args: argparse.Namespace,
) -> tuple[TransformersBackend, RawBackend, ModelSettings, str]:
    """Validate a real-model invocation and load the pinned model once."""

    code_commit = _validated_commit(args.code_commit, name="code_commit")
    model_revision = _validated_commit(args.model_revision, name="model_revision")
    _require_clean_code_commit(code_commit)
    _verify_benchmark_checkout(args.data_root)
    if args.raw_response_file is not None and Path(args.raw_response_file).exists():
        raise FileExistsError(f"raw response file already exists: {args.raw_response_file}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be supplied through the Kaggle secret environment")
    _progress_to_stderr(
        f"loading {args.model_id}@{model_revision} quantization={args.quantization} "
        f"dtype={args.dtype}"
    )
    model_backend = TransformersBackend.from_pretrained(
        model_id=args.model_id,
        revision=model_revision,
        token=token,
        quantization=args.quantization,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    del token
    _progress_to_stderr(
        f"model loaded in {model_backend.metadata.load_seconds:.1f}s "
        f"dtype={model_backend.metadata.dtype}"
    )
    raw_backend: RawBackend = model_backend
    if args.raw_response_file is not None:
        raw_backend = RawResponseRecorder(model_backend, args.raw_response_file)
    settings = ModelSettings(
        model_id=args.model_id,
        revision=model_revision,
        quantization=args.quantization,
        temperature=0.0,
        seed=args.seed,
        decoding_settings=(
            ("max_new_tokens", str(args.max_new_tokens)),
            ("do_sample", "false"),
            ("dtype", args.dtype),
        ),
    )
    return model_backend, raw_backend, settings, code_commit


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _require_new_output_dir(args.output_dir)
    _verify_benchmark_files(args.data_root)
    dataset = load_benchmark(args.data_root, strict_counts=True)

    if args.command == "smoke":
        single, full = _fake_smoke_backends(len(dataset.design_records))
        run = run_design_baselines(
            dataset,
            single_backend=single,
            full_sequential_backend=full,
        )
        summary = write_design_artifacts(
            run,
            dataset,
            output_dir=args.output_dir,
            classification="synthetic_smoke",
            model_metadata={
                "model_id": "fake/stage2-smoke",
                "synthetic": True,
                "token_counts": "unknown",
            },
            code_commit=None,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0

    if args.command == "smoke-gpu" and not 1 <= args.limit <= len(dataset.design_records):
        raise ValueError(f"--limit must be between 1 and {len(dataset.design_records)}")
    model_backend, raw_backend, settings, code_commit = _load_real_backend(args)
    single = InstrumentedBackend(raw_backend, settings, CallLedger())
    full = InstrumentedBackend(raw_backend, settings, CallLedger())

    if args.command == "smoke-gpu":
        try:
            diagnostic = run_design_diagnostic(
                dataset,
                single_backend=single,
                full_sequential_backend=full,
                limit=args.limit,
                max_attempts=args.max_attempts,
                progress=_progress_to_stderr,
            )
        except Exception as exc:
            write_failed_design_artifacts(
                output_dir=args.output_dir,
                single_ledger=single.ledger,
                full_sequential_ledger=full.ledger,
                error=exc,
            )
            raise
        summary = write_diagnostic_artifacts(
            diagnostic,
            dataset,
            output_dir=args.output_dir,
            model_metadata=model_backend.metadata.to_dict(),
            code_commit=code_commit,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0 if diagnostic.all_ok else 3

    try:
        run = run_design_baselines(
            dataset,
            single_backend=single,
            full_sequential_backend=full,
            max_attempts=args.max_attempts,
            progress=_progress_to_stderr,
        )
    except Exception as exc:
        write_failed_design_artifacts(
            output_dir=args.output_dir,
            single_ledger=single.ledger,
            full_sequential_ledger=full.ledger,
            error=exc,
        )
        raise
    summary = write_design_artifacts(
        run,
        dataset,
        output_dir=args.output_dir,
        classification="development_model",
        model_metadata=model_backend.metadata.to_dict(),
        code_commit=code_commit,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
