"""Stage 5: one confirmatory run on the locked 225-request test.

The only execution route to the 225 locked requests. Every model, decoding and
prompt setting comes from the frozen manifest (``configs/freeze_v1.json``),
which must match the current code (``rulegate.freeze.verify_manifest``); no
setting can be overridden from the command line. ``--confirm-locked-test`` is
required. One model load runs, on the same 225 requests:

1. Full Sequential Prompting with Both (reference),
2. Single Prompting (efficiency control),
3. plain screening + verifier (ablation),
4. action-aware screening + verifier (RuleGate, confirmatory).

Each finished condition is checkpointed to ``--partial-dir`` (ledger and
predictions only) so that a crashed multi-hour session leaves evidence. The
pre-registered decisions (H1, H2, overall) are computed for action-aware
screening; plain screening and Single Prompting are descriptive.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .data import load_benchmark
from .freeze import DEFAULT_MANIFEST_PATH, manifest_sha256, verify_manifest
from .model_backend import CallLedger, InstrumentedBackend, ModelSettings
from .schemas import BenchmarkDataset
from .stage2 import (
    _load_real_backend,
    _progress_to_stderr,
    _require_new_output_dir,
    _verify_benchmark_files,
    _write_json,
)
from .stage3 import (
    CONDITIONS,
    FULL_REFERENCE,
    PRIMARY_CONDITION,
    ConditionRun,
    _fake_stage3_backend,
    run_full_reference,
    run_screened_condition,
    run_single_prompting,
    write_stage3_artifacts,
)


CONFIRMATORY_NOTE = (
    "locked 225-request test: confirmatory for action_aware_screening; "
    "exploratory for plain_screening"
)
CONDITION_ORDER = ("plain", "action-aware")


def _checkpoint(partial_dir: Path | None, run: ConditionRun) -> None:
    if partial_dir is None:
        return
    folder = partial_dir / run.name
    folder.mkdir(parents=True, exist_ok=False)
    run.ledger.write_jsonl(folder / "calls.jsonl")
    _write_json(folder / "predictions.json", [asdict(p) for p in run.predictions])


def run_locked_test(
    dataset: BenchmarkDataset,
    backend_factory: Any,
    *,
    expected_requests: int,
    partial_dir: Path | None = None,
    progress: Any = _progress_to_stderr,
) -> list[ConditionRun]:
    records = dataset.test_records
    if len(records) != expected_requests:
        raise RuntimeError(f"expected {expected_requests} locked requests, got {len(records)}")
    runs: list[ConditionRun] = []
    runs.append(run_full_reference(dataset, backend=backend_factory(), records=records,
                                   progress=progress))
    _checkpoint(partial_dir, runs[-1])
    runs.append(run_single_prompting(dataset, backend=backend_factory(), records=records,
                                     progress=progress))
    _checkpoint(partial_dir, runs[-1])
    for name in CONDITION_ORDER:
        runs.append(
            run_screened_condition(dataset, CONDITIONS[name], backend=backend_factory(),
                                   records=records, progress=progress)
        )
        _checkpoint(partial_dir, runs[-1])
    return runs


def confirmatory_decision(summary: dict[str, Any]) -> dict[str, Any]:
    entry = summary["conditions"][PRIMARY_CONDITION]
    comparison = entry.get("vs_full_sequential_both")
    if not comparison:
        return {
            "condition": PRIMARY_CONDITION,
            "status": "NOT_EVALUABLE",
            "reason": "unparsed verifier outputs; see requests_with_failed_verdicts",
        }
    h1, h2 = comparison["h1_status"], comparison["h2_status"]
    overall = "GO" if h1 == "GO" and h2 == "GO" else (
        "INCONCLUSIVE" if h1 == "INCONCLUSIVE" and h2 == "GO" else "NOT SUPPORTED"
    )
    return {
        "condition": PRIMARY_CONDITION,
        "h1": h1,
        "h1_delta_f1": comparison["h1_delta_f1"],
        "h2": h2,
        "h2_call_reduction": comparison["h2_call_reduction"],
        "overall": overall,
        "rule": "research GO requires H1 GO and H2 GO (docs/frozen_protocol.md)",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run-test", "smoke"):
        command = sub.add_parser(name)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
        command.add_argument("--partial-dir", type=Path, default=None)
    run_test = sub.choices["run-test"]
    run_test.add_argument("--code-commit", required=True)
    run_test.add_argument("--raw-response-file", type=Path, default=None)
    run_test.add_argument("--confirm-locked-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = verify_manifest(args.manifest)
    freeze_sha = manifest_sha256(manifest)
    _require_new_output_dir(args.output_dir)
    if args.partial_dir is not None:
        _require_new_output_dir(args.partial_dir)
    _verify_benchmark_files(args.data_root)
    dataset = load_benchmark(args.data_root, strict_counts=True)
    expected = manifest["benchmark"]["expected_requests"]
    conditions = [CONDITIONS[name] for name in CONDITION_ORDER]

    if args.command == "smoke":
        settings = ModelSettings(model_id="fake/stage5-smoke", revision="synthetic",
                                 quantization="none", seed=manifest["model"]["seed"])
        budget = expected * 9 * 2
        runs = run_locked_test(
            dataset, lambda: _fake_stage3_backend(settings, budget),
            expected_requests=expected, partial_dir=args.partial_dir,
            progress=lambda message: None,
        )
        summary = write_stage3_artifacts(
            dataset, runs, output_dir=args.output_dir, classification="synthetic_smoke",
            model_metadata={"model_id": "fake/stage5-smoke", "synthetic": True},
            code_commit=None, conditions=conditions, split="test",
            comparison_note=CONFIRMATORY_NOTE, stage=5,
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, sort_keys=True))
        return 0

    if not args.confirm_locked_test:
        raise SystemExit("refusing to run the locked test without --confirm-locked-test")
    model = manifest["model"]
    load_args = argparse.Namespace(
        code_commit=args.code_commit,
        model_revision=model["revision"],
        model_id=model["model_id"],
        quantization=model["quantization"],
        dtype=model["dtype"],
        max_new_tokens=model["max_new_tokens"],
        seed=model["seed"],
        data_root=args.data_root,
        raw_response_file=args.raw_response_file,
    )
    model_backend, raw_backend, settings, code_commit = _load_real_backend(load_args)
    runs = run_locked_test(
        dataset,
        lambda: InstrumentedBackend(raw_backend, settings, CallLedger()),
        expected_requests=expected,
        partial_dir=args.partial_dir,
    )
    summary = write_stage3_artifacts(
        dataset,
        runs,
        output_dir=args.output_dir,
        classification="locked_test",
        model_metadata=model_backend.metadata.to_dict(),
        code_commit=code_commit,
        conditions=conditions,
        split="test",
        comparison_note=CONFIRMATORY_NOTE,
        stage=5,
    )
    summary["freeze"] = {"freeze_id": manifest["freeze_id"], "manifest_sha256": freeze_sha}
    summary["confirmatory"] = confirmatory_decision(summary)
    _write_json(Path(args.output_dir) / "summary.json", summary)
    print(json.dumps({"confirmatory": summary["confirmatory"], "freeze": summary["freeze"]},
                     sort_keys=True))
    return 0 if summary["all_verdicts_parsed"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
