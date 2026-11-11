"""Stage 3: plain and action-aware screening on the design split.

One model load runs, on the same requests and in the same session:

* the Full Sequential Prompting with Both reference (nine verifier calls per
  request), and
* each requested screening condition: one screening call, then the unchanged
  verifier on every ``POSSIBLE``/``UNCERTAIN`` policy, with full nine-policy
  fallback on an invalid screen.

Every condition has its own call ledger, so H2 counts every screening,
verification, retry and fallback call that actually happened. All conditions
use the same model revision, dtype, decoding settings, verifier prompt, system
message and parser as Stage 2; only the screening call has a larger generation
cap (it lists nine labels). The 225 locked requests are not reachable here.

Design-split H1/H2 numbers are development evidence (n = 18), not the
confirmatory test.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .accounting import compare_call_reduction, summarize_calls
from .baselines import make_sequential_both_verifier
from .bootstrap import paired_bootstrap_delta_f1
from .data import load_benchmark
from .decisions import decide_h1_bootstrap
from .methods import MethodName
from .metrics import evaluate
from .model_backend import (
    CallLedger,
    FakeBackend,
    InstrumentedBackend,
    ModelSettings,
    RawBackendResponse,
)
from .prompts import SEQUENTIAL_BOTH_TEMPLATE_ID, SEQUENTIAL_BOTH_TEMPLATE_SHA256
from .schemas import POLICY_IDS, BenchmarkDataset, Policy
from .screening import (
    PolicyRoute,
    ScreenLabel,
    parse_action_aware_screen_response,
    parse_screen_response,
    run_screening,
)
from .screening_prompts import (
    ACTION_AWARE_SCREENING_TEMPLATE_ID,
    ACTION_AWARE_SCREENING_TEMPLATE_SHA256,
    PLAIN_SCREENING_TEMPLATE_ID,
    PLAIN_SCREENING_TEMPLATE_SHA256,
    SCREENING_MAX_NEW_TOKENS,
    SCREENING_SYSTEM_PROMPT_V1,
    render_action_aware_screening_prompt,
    render_plain_screening_prompt,
)
from .stage2 import (
    DEFAULT_SEED,
    ProgressCallback,
    _load_real_backend,
    _no_progress,
    _progress_to_stderr,
    _require_new_output_dir,
    _run_metadata,
    _verify_benchmark_files,
    _write_json,
    add_real_model_arguments,
)
from .verification import VerificationError


FULL_REFERENCE = "full_sequential_both"


@dataclass(frozen=True, slots=True)
class ScreeningCondition:
    name: str
    method: MethodName
    template_id: str
    template_sha256: str
    render: Callable[[str, Sequence[Policy]], str]
    parser: Callable[[str], tuple[PolicyRoute, ...]]


CONDITIONS: dict[str, ScreeningCondition] = {
    "plain": ScreeningCondition(
        name="plain_screening",
        method=MethodName.PLAIN_SCREENING,
        template_id=PLAIN_SCREENING_TEMPLATE_ID,
        template_sha256=PLAIN_SCREENING_TEMPLATE_SHA256,
        render=render_plain_screening_prompt,
        parser=parse_screen_response,
    ),
    "action-aware": ScreeningCondition(
        name="action_aware_screening",
        method=MethodName.ACTION_AWARE_SCREENING,
        template_id=ACTION_AWARE_SCREENING_TEMPLATE_ID,
        template_sha256=ACTION_AWARE_SCREENING_TEMPLATE_SHA256,
        render=render_action_aware_screening_prompt,
        parser=parse_action_aware_screen_response,
    ),
}


@dataclass(frozen=True, slots=True)
class ConditionPrediction:
    """Per-request outcome; holds IDs and labels only, never model text."""

    request_id: str
    status: str
    predicted_policy_ids: tuple[int, ...]
    checked_policy_ids: tuple[int, ...]
    failed_policy_ids: tuple[int, ...]
    screen_labels: tuple[tuple[int, str], ...]
    screen_valid: bool | None
    fallback_used: bool
    screening_call_ids: tuple[int, ...]
    verification_call_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ConditionRun:
    name: str
    request_ids: tuple[str, ...]
    predictions: tuple[ConditionPrediction, ...]
    ledger: CallLedger

    @property
    def all_verdicts_parsed(self) -> bool:
        return all(p.status == "ok" for p in self.predictions)


def _verify_each(
    verifier: Any,
    request: Any,
    policy_ids: Sequence[int],
    *,
    max_attempts: int,
    fallback_used: bool,
    fallback_type: str | None,
) -> tuple[list[int], list[int], list[int]]:
    """Verify policies one by one (same calls as ``verify_candidates``)."""

    predicted: list[int] = []
    failed: list[int] = []
    call_ids: list[int] = []
    for policy_id in policy_ids:
        start = len(verifier.backend.ledger.records)
        try:
            result = verifier.verify_candidates(
                request,
                (policy_id,),
                max_attempts=max_attempts,
                fallback_used=fallback_used,
                fallback_type=fallback_type,
            )
        except VerificationError:
            failed.append(policy_id)
        else:
            predicted.extend(result.predicted_policy_ids)
        call_ids.extend(r.call_id for r in verifier.backend.ledger.records[start:])
    return predicted, failed, call_ids


def run_full_reference(
    dataset: BenchmarkDataset,
    *,
    backend: InstrumentedBackend,
    limit: int,
    max_attempts: int = 1,
    progress: ProgressCallback = _no_progress,
) -> ConditionRun:
    verifier = make_sequential_both_verifier(backend=backend, policies=dataset.policies)
    predictions = []
    records = dataset.design_records[:limit]
    for index, record in enumerate(records, start=1):
        request = record.to_inference_request()
        predicted, failed, call_ids = _verify_each(
            verifier, request, POLICY_IDS, max_attempts=max_attempts,
            fallback_used=False, fallback_type=None,
        )
        predictions.append(
            ConditionPrediction(
                request_id=request.request_id,
                status="failed" if failed else "ok",
                predicted_policy_ids=tuple(predicted),
                checked_policy_ids=POLICY_IDS,
                failed_policy_ids=tuple(failed),
                screen_labels=(),
                screen_valid=None,
                fallback_used=False,
                screening_call_ids=(),
                verification_call_ids=tuple(call_ids),
            )
        )
        progress(f"{FULL_REFERENCE} {index}/{len(records)} calls={len(call_ids)}")
    return ConditionRun(
        FULL_REFERENCE, tuple(r.request_id for r in records), tuple(predictions), backend.ledger
    )


def run_screened_condition(
    dataset: BenchmarkDataset,
    condition: ScreeningCondition,
    *,
    backend: InstrumentedBackend,
    limit: int,
    max_attempts: int = 1,
    progress: ProgressCallback = _no_progress,
) -> ConditionRun:
    """Screen, then verify retained policies with the shared Stage 2 verifier."""

    verifier = make_sequential_both_verifier(backend=backend, policies=dataset.policies)
    predictions = []
    records = dataset.design_records[:limit]
    for index, record in enumerate(records, start=1):
        request = record.to_inference_request()
        decision = run_screening(
            backend=backend,
            request=request,
            prompt=condition.render(request.text, dataset.policies),
            prompt_template=condition.template_id,
            method=condition.method,
            max_attempts=max_attempts,
            system_prompt=SCREENING_SYSTEM_PROMPT_V1,
            max_new_tokens=SCREENING_MAX_NEW_TOKENS,
            parser=condition.parser,
        )
        predicted, failed, call_ids = _verify_each(
            verifier,
            request,
            decision.candidate_ids,
            max_attempts=max_attempts,
            fallback_used=decision.fallback_used,
            fallback_type=decision.fallback_type,
        )
        predictions.append(
            ConditionPrediction(
                request_id=request.request_id,
                status="failed" if failed else "ok",
                predicted_policy_ids=tuple(predicted),
                checked_policy_ids=tuple(decision.candidate_ids),
                failed_policy_ids=tuple(failed),
                screen_labels=tuple((r.policy_id, r.label.value) for r in decision.routes),
                screen_valid=decision.valid,
                fallback_used=decision.fallback_used,
                screening_call_ids=decision.call_ids,
                verification_call_ids=tuple(call_ids),
            )
        )
        progress(
            f"{condition.name} {index}/{len(records)} valid={decision.valid} "
            f"candidates={len(decision.candidate_ids)} calls="
            f"{len(decision.call_ids) + len(call_ids)}"
        )
    return ConditionRun(
        condition.name, tuple(r.request_id for r in records), tuple(predictions), backend.ledger
    )


# --------------------------------------------------------------------------
# Reporting


def screening_report(
    run: ConditionRun,
    gold: dict[str, tuple[int, ...]],
    full: ConditionRun | None,
) -> dict[str, Any]:
    """Screening recall, induced misses, candidate counts, fallbacks."""

    counts = Counter(len(p.checked_policy_ids) for p in run.predictions)
    labels = Counter(label for p in run.predictions for _, label in p.screen_labels)
    violating = [p for p in run.predictions if gold[p.request_id]]
    retained = [
        p for p in violating if set(gold[p.request_id]) <= set(p.checked_policy_ids)
    ]
    full_by_id = {p.request_id: p for p in full.predictions} if full else {}
    gold_excluded_full_detected = 0
    full_positives_excluded = 0
    verdict_mismatches = 0
    for p in run.predictions:
        reference = full_by_id.get(p.request_id)
        if reference is None:
            continue
        excluded = set(POLICY_IDS) - set(p.checked_policy_ids)
        full_positive = set(reference.predicted_policy_ids)
        full_positives_excluded += len(full_positive & excluded)
        gold_excluded_full_detected += len(set(gold[p.request_id]) & full_positive & excluded)
        for policy_id in set(p.checked_policy_ids) - set(p.failed_policy_ids):
            if policy_id in reference.failed_policy_ids:
                continue
            if (policy_id in p.predicted_policy_ids) != (policy_id in full_positive):
                verdict_mismatches += 1
    return {
        "requests": len(run.predictions),
        "valid_screens": sum(1 for p in run.predictions if p.screen_valid),
        "invalid_screens_full_fallback": sum(1 for p in run.predictions if p.fallback_used),
        "valid_zero_candidate_screens": sum(
            1 for p in run.predictions if p.screen_valid and not p.checked_policy_ids
        ),
        "candidate_count_distribution": dict(sorted(counts.items())),
        "mean_verified_policies": (
            sum(len(p.checked_policy_ids) for p in run.predictions) / len(run.predictions)
        ),
        "label_counts": dict(sorted(labels.items())),
        "gold_policy_retention": {
            "retained": len(retained),
            "violating_requests": len(violating),
            "recall": len(retained) / len(violating) if violating else None,
        },
        "screening_induced_misses": {
            "gold_excluded_but_full_detected": gold_excluded_full_detected if full else None,
            "full_positives_excluded": full_positives_excluded if full else None,
        },
        "verdict_mismatches_vs_full_on_retained": verdict_mismatches if full else None,
    }


def _condition_payload(run: ConditionRun) -> list[dict[str, Any]]:
    return [asdict(p) for p in run.predictions]


def write_stage3_artifacts(
    dataset: BenchmarkDataset,
    runs: Sequence[ConditionRun],
    *,
    output_dir: str | Path,
    classification: str,
    model_metadata: dict[str, Any],
    code_commit: str | None,
    conditions: Sequence[ScreeningCondition],
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    if classification not in {"synthetic_smoke", "development_smoke", "development_model"}:
        raise ValueError("unknown classification")
    destination = _require_new_output_dir(output_dir)
    destination.mkdir(parents=True)
    request_ids = runs[0].request_ids
    if any(run.request_ids != request_ids for run in runs):
        raise ValueError("all conditions must cover the same requests")
    trusted = {r.request_id: r for r in dataset.design_records}
    gold = {rid: trusted[rid].gold_violated_policy_ids for rid in request_ids}
    _write_json(
        destination / "trusted_design_index.json",
        [
            {
                "request_id": rid,
                "gold_violated_policy_ids": list(gold[rid]),
                "source_file": trusted[rid].source_file,
                "source_index": trusted[rid].source_index,
            }
            for rid in request_ids
        ],
    )
    full = next((run for run in runs if run.name == FULL_REFERENCE), None)
    summary: dict[str, Any] = {
        "stage": 3,
        "classification": classification,
        "split": "design",
        "locked_test_used": False,
        "request_count": len(request_ids),
        "all_verdicts_parsed": all(run.all_verdicts_parsed for run in runs),
        "conditions": {},
    }
    if classification != "synthetic_smoke":
        summary["metrics_note"] = (
            "design split only (development evidence); not the confirmatory test"
        )
    gold_list = [gold[rid] for rid in request_ids]
    for run in runs:
        folder = destination / run.name
        folder.mkdir()
        _write_json(folder / "predictions.json", _condition_payload(run))
        run.ledger.write_jsonl(folder / "calls.jsonl")
        entry: dict[str, Any] = {
            "calls": asdict(summarize_calls(run.ledger, request_ids)),
            "parse_status": dict(
                sorted(Counter(r.parse_status for r in run.ledger.records).items())
            ),
            "requests_with_failed_verdicts": sum(
                1 for p in run.predictions if p.status != "ok"
            ),
        }
        if run.name != FULL_REFERENCE:
            entry["screening"] = screening_report(run, gold, full)
        if classification != "synthetic_smoke" and run.all_verdicts_parsed:
            entry["metrics"] = asdict(
                evaluate(gold_list, [p.predicted_policy_ids for p in run.predictions])
            )
        if (
            full is not None
            and run is not full
            and classification != "synthetic_smoke"
            and run.all_verdicts_parsed
            and full.all_verdicts_parsed
        ):
            h2 = compare_call_reduction(
                full_ledger=full.ledger, screened_ledger=run.ledger, request_ids=request_ids
            )
            bootstrap = paired_bootstrap_delta_f1(
                gold_list,
                [p.predicted_policy_ids for p in run.predictions],
                [p.predicted_policy_ids for p in full.predictions],
                n_resamples=bootstrap_resamples,
            )
            entry["vs_full_sequential_both"] = {
                "note": "design split, n=18: indicative only, not a hypothesis test",
                "h2_call_reduction": h2.call_reduction,
                "h2_status": h2.status.value,
                "token_ratio_input": (
                    h2.screened.input_tokens / h2.full.input_tokens if h2.full.input_tokens else None
                ),
                "token_ratio_output": (
                    h2.screened.output_tokens / h2.full.output_tokens
                    if h2.full.output_tokens
                    else None
                ),
                "latency_ratio": (
                    h2.screened.elapsed_seconds / h2.full.elapsed_seconds
                    if h2.full.elapsed_seconds
                    else None
                ),
                "h1_delta_f1": asdict(bootstrap),
                "h1_status": decide_h1_bootstrap(bootstrap).status.value,
            }
        summary["conditions"][run.name] = entry
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "run_metadata.json",
        _run_metadata(
            classification=classification,
            model_metadata=model_metadata,
            code_commit=code_commit,
            extra={
                "stage": 3,
                "design_subset_limit": len(request_ids),
                "verifier": {
                    "template_id": SEQUENTIAL_BOTH_TEMPLATE_ID,
                    "template_sha256": SEQUENTIAL_BOTH_TEMPLATE_SHA256,
                },
                "screening": {
                    c.name: {
                        "template_id": c.template_id,
                        "template_sha256": c.template_sha256,
                        "max_new_tokens": SCREENING_MAX_NEW_TOKENS,
                    }
                    for c in conditions
                },
            },
        ),
    )
    return summary


# --------------------------------------------------------------------------
# CLI


def _fake_stage3_backend(settings: ModelSettings, count: int) -> InstrumentedBackend:
    """Deterministic synthetic backend: retains policies 1-3, verdicts all false."""

    def respond(request: Any) -> RawBackendResponse:
        if request.prompt_template in {
            PLAIN_SCREENING_TEMPLATE_ID,
            ACTION_AWARE_SCREENING_TEMPLATE_ID,
        }:
            labels = [
                {"policy_id": pid, "label": "POSSIBLE" if pid <= 2 else
                 "UNCERTAIN" if pid == 3 else "EXCLUDE"}
                for pid in POLICY_IDS
            ]
            payload: dict[str, Any] = {"labels": labels}
            if request.prompt_template == ACTION_AWARE_SCREENING_TEMPLATE_ID:
                payload = {"action": "a", "target": "t", "content": "c", **payload}
            return RawBackendResponse(json.dumps(payload))
        return RawBackendResponse('{"violated": false}')

    return InstrumentedBackend(FakeBackend([respond] * count), settings, CallLedger())


def _parse_conditions(value: str) -> list[ScreeningCondition]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in CONDITIONS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"conditions must be a comma list of {sorted(CONDITIONS)}; got {value!r}"
        )
    return [CONDITIONS[name] for name in dict.fromkeys(names)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="synthetic fake-backend integration run")
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--conditions", type=_parse_conditions,
                       default=_parse_conditions("plain,action-aware"))
    design = sub.add_parser(
        "run-design",
        help="real model: full reference plus screening conditions on design requests",
    )
    add_real_model_arguments(design)
    design.add_argument("--conditions", type=_parse_conditions,
                        default=_parse_conditions("plain,action-aware"))
    design.add_argument("--limit", type=int, default=18)
    design.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _require_new_output_dir(args.output_dir)
    _verify_benchmark_files(args.data_root)
    dataset = load_benchmark(args.data_root, strict_counts=True)
    design_count = len(dataset.design_records)

    if args.command == "smoke":
        settings = ModelSettings(model_id="fake/stage3-smoke", revision="synthetic",
                                 quantization="none", seed=DEFAULT_SEED)
        budget = design_count * 9 * 2
        runs = [run_full_reference(dataset, backend=_fake_stage3_backend(settings, budget),
                                   limit=design_count)]
        runs += [
            run_screened_condition(dataset, condition,
                                   backend=_fake_stage3_backend(settings, budget),
                                   limit=design_count)
            for condition in args.conditions
        ]
        summary = write_stage3_artifacts(
            dataset, runs, output_dir=args.output_dir, classification="synthetic_smoke",
            model_metadata={"model_id": "fake/stage3-smoke", "synthetic": True},
            code_commit=None, conditions=args.conditions,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0

    if not 1 <= args.limit <= design_count:
        raise ValueError(f"--limit must be between 1 and {design_count}")
    model_backend, raw_backend, settings, code_commit = _load_real_backend(args)
    runs = [
        run_full_reference(
            dataset,
            backend=InstrumentedBackend(raw_backend, settings, CallLedger()),
            limit=args.limit,
            max_attempts=args.max_attempts,
            progress=_progress_to_stderr,
        )
    ]
    for condition in args.conditions:
        runs.append(
            run_screened_condition(
                dataset,
                condition,
                backend=InstrumentedBackend(raw_backend, settings, CallLedger()),
                limit=args.limit,
                max_attempts=args.max_attempts,
                progress=_progress_to_stderr,
            )
        )
    summary = write_stage3_artifacts(
        dataset,
        runs,
        output_dir=args.output_dir,
        classification="development_model" if args.limit == design_count else "development_smoke",
        model_metadata=model_backend.metadata.to_dict(),
        code_commit=code_commit,
        conditions=args.conditions,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, sort_keys=True))
    return 0 if summary["all_verdicts_parsed"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
