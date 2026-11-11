"""Frozen experimental configuration for the locked 225-request test (Stage 5).

``build_manifest()`` derives the manifest from the code itself: model and
decoding settings, every prompt template ID and hash (system message included),
the benchmark data hashes, the pre-registered analysis settings, and SHA-256
hashes of every source file that shapes predictions or the analysis. The
committed ``configs/freeze_v1.json`` must equal ``build_manifest()``; a CPU test
and the locked-test runner both check this, so any change to a frozen file
after the freeze is detected and requires a new, documented freeze.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .prompts import (
    SEQUENTIAL_BOTH_TEMPLATE_ID,
    SEQUENTIAL_BOTH_TEMPLATE_SHA256,
    SINGLE_PROMPT_TEMPLATE_ID,
    SINGLE_PROMPT_TEMPLATE_SHA256,
)
from .screening_prompts import (
    ACTION_AWARE_SCREENING_TEMPLATE_ID,
    ACTION_AWARE_SCREENING_TEMPLATE_SHA256,
    PLAIN_SCREENING_TEMPLATE_ID,
    PLAIN_SCREENING_TEMPLATE_SHA256,
    SCREENING_MAX_NEW_TOKENS,
)
from .stage2 import (
    BENCHMARK_COMMIT,
    BENCHMARK_MANIFEST_SHA256,
    DEFAULT_MODEL_ID,
    DEFAULT_SEED,
)


FREEZE_ID = "rulegate-freeze-v1"
FROZEN_ON = "2026-11-11"
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "configs" / "freeze_v1.json"

MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

# Every module whose code can change a prompt, a model call, a parsed output,
# a prediction, a metric or a decision.
FROZEN_SOURCE_FILES = (
    "accounting.py",
    "baselines.py",
    "bootstrap.py",
    "data.py",
    "decisions.py",
    "jsonreply.py",
    "metrics.py",
    "methods.py",
    "model_backend.py",
    "prompts.py",
    "schemas.py",
    "screening.py",
    "screening_prompts.py",
    "stage3.py",
    "transformers_backend.py",
    "verification.py",
)


class FreezeError(RuntimeError):
    """The code or configuration no longer matches the frozen manifest."""


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_manifest() -> dict[str, Any]:
    return {
        "freeze_id": FREEZE_ID,
        "frozen_on": FROZEN_ON,
        "benchmark": {
            "repository": "https://github.com/PedroCV/policy-compliance-requests-AI",
            "commit": BENCHMARK_COMMIT,
            "data_manifest_sha256": BENCHMARK_MANIFEST_SHA256,
            "split": "test",
            "expected_requests": 225,
        },
        "model": {
            "model_id": DEFAULT_MODEL_ID,
            "revision": MODEL_REVISION,
            "quantization": "none",
            "dtype": "float16",
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 128,
            "screening_max_new_tokens": SCREENING_MAX_NEW_TOKENS,
            "seed": DEFAULT_SEED,
            "max_attempts": 1,
            "runtime": "Kaggle NvidiaTeslaT4 (2 x Tesla T4), device_map=auto",
        },
        "prompts": {
            "single_prompting": {
                "template_id": SINGLE_PROMPT_TEMPLATE_ID,
                "sha256": SINGLE_PROMPT_TEMPLATE_SHA256,
            },
            "sequential_both_verifier": {
                "template_id": SEQUENTIAL_BOTH_TEMPLATE_ID,
                "sha256": SEQUENTIAL_BOTH_TEMPLATE_SHA256,
            },
            "plain_screening": {
                "template_id": PLAIN_SCREENING_TEMPLATE_ID,
                "sha256": PLAIN_SCREENING_TEMPLATE_SHA256,
            },
            "action_aware_screening": {
                "template_id": ACTION_AWARE_SCREENING_TEMPLATE_ID,
                "sha256": ACTION_AWARE_SCREENING_TEMPLATE_SHA256,
            },
        },
        "conditions": {
            "action_aware_screening": "primary (RuleGate): action-aware screen + verifier",
            "full_sequential_both": "reference comparator for H1 and H2",
            "plain_screening": "ablation (H3, exploratory)",
            "single_prompting": "efficiency control (descriptive)",
        },
        "analysis": {
            "primary_metric": (
                "mean of nine Appendix D policy-specific F1 scores, protocol reading "
                "(docs/protocol.md); zero denominator -> 0.0"
            ),
            "h1": {
                "delta": "mean_policy_f1(action_aware_screening) - mean_policy_f1(full_sequential_both)",
                "bootstrap": "paired by request, 10000 resamples, seed 484654, percentile 95% CI",
                "margin": 0.05,
                "rule": "GO if CI_low > -0.05; FAIL if CI_high <= -0.05; otherwise INCONCLUSIVE",
            },
            "h2": {
                "threshold": 0.5,
                "rule": "GO if 1 - calls(action_aware)/calls(full) >= 0.50 on measured ledgers",
                "counted": "every backend attempt: screening, verification, retries, fallbacks",
            },
            "overall": "research GO requires H1 GO and H2 GO",
            "secondary": [
                "exact-match accuracy, compliant recall, violating detection recall",
                "per-policy F1 and Appendix D counts",
                "gold-policy retention, screening-induced false negatives (proposal definition)",
                "candidate-count distribution, zero-candidate and invalid screens, fallbacks",
                "input/output tokens and latency for every condition",
                "plain vs action-aware (H3) and Single Prompting (efficiency control): descriptive",
            ],
        },
        "frozen_source_sha256": {
            name: _file_sha256(PACKAGE_ROOT / name) for name in FROZEN_SOURCE_FILES
        },
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _differences(expected: Any, observed: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(observed, dict):
        diffs: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in observed:
                diffs.append(f"{path}: missing in manifest file")
            elif key not in expected:
                diffs.append(f"{path}: not in current code")
            else:
                diffs.extend(_differences(expected[key], observed[key], path))
        return diffs
    return [] if expected == observed else [f"{prefix}: file={observed!r} code={expected!r}"]


def verify_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Return the manifest if the committed file matches the current code."""

    observed = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = build_manifest()
    diffs = _differences(expected, observed)
    if diffs:
        raise FreezeError(
            "frozen manifest does not match the current code:\n  " + "\n  ".join(diffs)
        )
    return observed


def write_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> str:
    manifest = build_manifest()
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest_sha256(manifest)


if __name__ == "__main__":  # pragma: no cover
    import sys

    if sys.argv[1:] == ["write"]:
        print(write_manifest())
    else:
        print(manifest_sha256(verify_manifest()))
