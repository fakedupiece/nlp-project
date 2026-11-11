"""Mechanical Stage 2 GO audit of a real ``run-design`` artifact directory.

Stage 2 is GO only when the real 18-request design run completed, every model
output parsed, every prediction reconciles with the measured call ledger, and
all calls used one pinned model configuration. This module checks those
conditions from the written artifacts alone; it never re-runs a model and never
reads the locked test split. Quality scores are reported, not gated: Stage 2 has
no F1 threshold.

Decision values:

* ``GO``       - every check passed.
* ``CONTINUE`` - at least one check failed; the failed checks say what to fix
                 (for example a versioned prompt/parser revision after parse
                 failures). ``FAIL`` is reserved for a reviewed human decision
                 that the baselines cannot be run under the fixed constraints.

Usage: ``python -m rulegate.stage2_gate <artifacts-dir> [--model-revision SHA]``
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from .prompts import SEQUENTIAL_BOTH_TEMPLATE_ID, SINGLE_PROMPT_TEMPLATE_ID
from .schemas import POLICY_IDS


EXPECTED_DESIGN_REQUESTS = 18
VALID_SINGLE = "valid_single_response"
VALID_VERDICT = "valid_verdict"


@dataclass(slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(slots=True)
class Stage2GateReport:
    decision: str
    checks: list[GateCheck] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "failed_checks": [check.name for check in self.failed],
            "checks": [asdict(check) for check in self.checks],
            "facts": self.facts,
        }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_stage2_gate(
    artifacts_dir: str | Path,
    *,
    expected_model_revision: str | None = None,
    expected_requests: int = EXPECTED_DESIGN_REQUESTS,
) -> Stage2GateReport:
    root = Path(artifacts_dir)
    checks: list[GateCheck] = []
    facts: dict[str, Any] = {}

    def check(name: str, passed: bool, detail: str = "") -> bool:
        checks.append(GateCheck(name, bool(passed), detail))
        return bool(passed)

    required = (
        "summary.json",
        "run_metadata.json",
        "trusted_design_index.json",
        "single_predictions.json",
        "full_sequential_predictions.json",
        "single_calls.jsonl",
        "full_sequential_calls.jsonl",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if (root / "failure.json").is_file():
        missing.append("failure.json present (run aborted)")
    if not check("artifacts_complete", not missing, ", ".join(missing)):
        return Stage2GateReport("CONTINUE", checks, facts)

    summary = _load_json(root / "summary.json")
    metadata = _load_json(root / "run_metadata.json")
    trusted = _load_json(root / "trusted_design_index.json")
    single_predictions = _load_json(root / "single_predictions.json")
    full_predictions = _load_json(root / "full_sequential_predictions.json")
    single_calls = _load_jsonl(root / "single_calls.jsonl")
    full_calls = _load_jsonl(root / "full_sequential_calls.jsonl")

    check(
        "real_development_run",
        summary.get("classification") == "development_model"
        and metadata.get("classification") == "development_model",
        f"classification={summary.get('classification')}",
    )
    check(
        "design_split_only",
        summary.get("split") == "design"
        and summary.get("locked_test_used") is False
        and metadata.get("locked_test_used") is False
        and all(str(r["request_id"]).startswith("rg-design-") for r in trusted),
        "split/locked-test flags and design request IDs",
    )
    trusted_ids = [r["request_id"] for r in trusted]
    check(
        "all_design_requests",
        len(trusted_ids) == expected_requests == len(set(trusted_ids)),
        f"{len(trusted_ids)} trusted design requests",
    )
    for label, predictions in (
        ("single", single_predictions),
        ("full_sequential", full_predictions),
    ):
        ids = [p["request_id"] for p in predictions]
        check(
            f"{label}_predictions_cover_design",
            ids == trusted_ids,
            f"{len(ids)} predictions in trusted order",
        )

    model = metadata.get("model") or {}
    revision = model.get("resolved_revision")
    facts["model"] = {
        key: model.get(key)
        for key in (
            "model_id", "resolved_revision", "quantization", "requested_dtype", "dtype",
            "max_new_tokens", "temperature", "do_sample", "seed", "device_name",
        )
    }
    revision_ok = isinstance(revision, str) and len(revision) == 40
    if expected_model_revision is not None:
        revision_ok = revision_ok and revision == expected_model_revision
    check("pinned_model_revision", revision_ok, f"resolved_revision={revision}")
    code = metadata.get("code_commit_declared")
    local_git = metadata.get("local_git") or {}
    check(
        "clean_pinned_code",
        isinstance(code, str)
        and len(code) == 40
        and local_git.get("commit") == code
        and local_git.get("dirty") is False,
        f"code={code} local_git={local_git}",
    )

    all_calls = single_calls + full_calls
    settings = {
        (
            c.get("model_id"), c.get("model_revision"), c.get("quantization"),
            c.get("temperature"), c.get("seed"), json.dumps(c.get("decoding_settings")),
        )
        for c in all_calls
    }
    check(
        "one_model_configuration",
        len(settings) == 1 and (revision is None or all(
            c.get("model_revision") == revision for c in all_calls
        )),
        f"{len(settings)} distinct call configurations",
    )
    check(
        "greedy_decoding",
        all(c.get("temperature") == 0.0 for c in all_calls)
        and model.get("do_sample") is False,
        "temperature=0 and do_sample=False",
    )
    check(
        "prompt_templates",
        all(c.get("prompt_template") == SINGLE_PROMPT_TEMPLATE_ID for c in single_calls)
        and all(c.get("prompt_template") == SEQUENTIAL_BOTH_TEMPLATE_ID for c in full_calls),
        "ledger template IDs match the versioned baselines",
    )
    check(
        "no_backend_failures",
        all(c.get("success") for c in all_calls),
        f"{sum(1 for c in all_calls if not c.get('success'))} failed calls",
    )
    single_invalid = [c for c in single_calls if c.get("parse_status") != VALID_SINGLE]
    full_invalid = [c for c in full_calls if c.get("parse_status") != VALID_VERDICT]
    check(
        "every_output_parsed",
        not single_invalid and not full_invalid,
        f"invalid single={len(single_invalid)} invalid verdicts={len(full_invalid)}",
    )
    check(
        "token_counts_known",
        all(c.get("input_tokens") is not None and c.get("output_tokens") is not None
            for c in all_calls),
        "input/output tokens recorded for every call",
    )

    # Reconcile predictions with exactly the ledger calls they claim.
    def reconcile_single() -> tuple[bool, str]:
        by_id = {c["call_id"]: c for c in single_calls}
        claimed: list[int] = []
        for prediction in single_predictions:
            ids = prediction["call_ids"]
            claimed.extend(ids)
            calls = [by_id.get(i) for i in ids]
            if not ids or any(c is None or c["request_id"] != prediction["request_id"]
                              for c in calls):
                return False, f"call IDs do not match request {prediction['request_id']}"
            final = calls[-1]
            if final["parse_status"] != VALID_SINGLE:
                return False, f"final call for {prediction['request_id']} is not valid"
            if json.loads(final["parsed_output"]) != prediction["predicted_policy_ids"]:
                return False, f"prediction differs from parsed output for {prediction['request_id']}"
        if sorted(claimed) != sorted(by_id):
            return False, "ledger calls not claimed exactly once by predictions"
        return True, f"{len(by_id)} calls reconciled"

    def reconcile_full() -> tuple[bool, str]:
        by_id = {c["call_id"]: c for c in full_calls}
        claimed: list[int] = []
        for prediction in full_predictions:
            ids = prediction["call_ids"]
            claimed.extend(ids)
            calls = [by_id.get(i) for i in ids]
            if any(c is None or c["request_id"] != prediction["request_id"] for c in calls):
                return False, f"call IDs do not match request {prediction['request_id']}"
            verdicts: dict[int, bool] = {}
            for call in calls:
                if call["parse_status"] == VALID_VERDICT:
                    verdicts[call["policy_id"]] = call["parsed_output"] == "true"
            if tuple(sorted(verdicts)) != POLICY_IDS:
                return False, f"{prediction['request_id']} lacks a verdict for every policy"
            expected = [pid for pid in POLICY_IDS if verdicts[pid]]
            if expected != prediction["predicted_policy_ids"]:
                return False, f"prediction differs from verdicts for {prediction['request_id']}"
        if sorted(claimed) != sorted(by_id):
            return False, "ledger calls not claimed exactly once by predictions"
        return True, f"{len(by_id)} calls reconciled"

    check("single_reconciles", *reconcile_single())
    check("full_sequential_reconciles", *reconcile_full())

    retries = defaultdict(int)
    for c in all_calls:
        if c.get("attempt", 1) > 1:
            retries[c["method"]] += 1
    facts["calls"] = {
        "single_prompting": len(single_calls),
        "full_sequential_both": len(full_calls),
        "retries": dict(retries),
        "minimum_expected": {
            "single_prompting": expected_requests,
            "full_sequential_both": expected_requests * len(POLICY_IDS),
        },
    }
    check(
        "call_counts",
        len(single_calls) >= expected_requests
        and len(full_calls) >= expected_requests * len(POLICY_IDS),
        json.dumps(facts["calls"]),
    )
    for key in ("single_prompting_metrics", "full_sequential_both_metrics"):
        metrics = summary.get(key)
        if metrics:
            facts[key] = {
                "mean_policy_f1": metrics["policy"]["mean_f1"],
                "exact_match_accuracy": metrics["requests"]["exact_match_accuracy"],
            }

    decision = "GO" if all(c.passed for c in checks) else "CONTINUE"
    return Stage2GateReport(decision, checks, facts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 GO audit of run-design artifacts")
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("--model-revision", default=None)
    args = parser.parse_args(argv)
    report = evaluate_stage2_gate(args.artifacts_dir, expected_model_revision=args.model_revision)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.decision == "GO" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
