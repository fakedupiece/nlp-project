"""Stage 2 design-split baseline runner and artifact writer.

The command surface deliberately exposes only the 18-request design split. The
225-request locked split has no Stage 2 execution route.
"""

from __future__ import annotations

import argparse
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
from .baselines import SinglePromptingBaseline, make_sequential_both_verifier
from .data import load_benchmark
from .metrics import evaluate
from .model_backend import (
    CallLedger,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
)
from .prompts import (
    SEQUENTIAL_BOTH_TEMPLATE_ID,
    SEQUENTIAL_BOTH_TEMPLATE_SHA256,
    SINGLE_PROMPT_TEMPLATE_ID,
    SINGLE_PROMPT_TEMPLATE_SHA256,
)
from .schemas import BenchmarkDataset
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


def run_design_baselines(
    dataset: BenchmarkDataset,
    *,
    single_backend: InstrumentedBackend,
    full_sequential_backend: InstrumentedBackend,
    max_attempts: int = 1,
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

    for trusted_record in dataset.design_records:
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

    for trusted_record in dataset.design_records:
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

    return DesignBaselineRun(
        request_ids=tuple(record.request_id for record in dataset.design_records),
        single_predictions=tuple(single_predictions),
        full_sequential_predictions=tuple(full_predictions),
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
    _write_json(destination / "run_metadata.json", metadata)
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

    design = subparsers.add_parser(
        "run-design",
        help="run both baselines with the pinned model on the 18 design cases",
    )
    design.add_argument("--data-root", type=Path, required=True)
    design.add_argument("--output-dir", type=Path, required=True)
    design.add_argument("--code-commit", required=True)
    design.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    design.add_argument("--model-revision", required=True)
    design.add_argument("--quantization", choices=("none", "4bit"), default="none")
    design.add_argument("--max-new-tokens", type=int, default=128)
    design.add_argument("--max-attempts", type=int, default=1)
    design.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


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

    code_commit = _validated_commit(args.code_commit, name="code_commit")
    model_revision = _validated_commit(args.model_revision, name="model_revision")
    _require_clean_code_commit(code_commit)
    _verify_benchmark_checkout(args.data_root)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be supplied through the Kaggle secret environment")
    raw_backend = TransformersBackend.from_pretrained(
        model_id=args.model_id,
        revision=model_revision,
        token=token,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    settings = ModelSettings(
        model_id=args.model_id,
        revision=model_revision,
        quantization=args.quantization,
        temperature=0.0,
        seed=args.seed,
        decoding_settings=(
            ("max_new_tokens", str(args.max_new_tokens)),
            ("do_sample", "false"),
        ),
    )
    single = InstrumentedBackend(raw_backend, settings, CallLedger())
    full = InstrumentedBackend(raw_backend, settings, CallLedger())
    try:
        run = run_design_baselines(
            dataset,
            single_backend=single,
            full_sequential_backend=full,
            max_attempts=args.max_attempts,
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
        model_metadata=raw_backend.metadata.to_dict(),
        code_commit=code_commit,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
