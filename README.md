# RuleGate

RuleGate evaluates a fixed architecture for policy-compliance screening:

`one conservative all-policy screen -> unchanged Cisneros Sequential Prompting with Both verifier for retained policies`

The screener routes every policy as `POSSIBLE`, `UNCERTAIN`, or `EXCLUDE`.
`POSSIBLE` and `UNCERTAIN` policies are verified; invalid screens fall back to
all nine policies. Candidate counts are adaptive. This project does not implement
BM25, dense retrieval, RRF, similarity ranking, or fixed Top-k selection.

## Current status

Stage 0/1 is complete. Stage 2 baseline code is implemented and has passed an
18-request fake-backend integration run; the real Kaggle/model run remains
`NOT RUN`. The official benchmark was loaded and audited at commit
`2d1944f2efa362021f931638b724714f7b74f288`:

- 9 policies;
- 18 design requests (9 compliant and one violation for each policy);
- 225 locked-test requests (119 compliant, 106 violating);
- one exact duplicate pair with matching gold `[7]`, preserved in place.

No LLM was run in Codex Cloud. The 225 locked requests were parsed for audit only
and were not used for prompting, tuning, or model evaluation. The Stage 2 CLI has
no locked-test execution route.

See [the protocol](docs/protocol.md), [data audit](docs/data_audit.md),
[experiment log](docs/experiment_log.md), and
[Kaggle handoff](docs/kaggle_execution.md). Stage 2 source decisions and current
gate status are in [source review](docs/source_review.md) and
[Stage 2 status](docs/stage2_status.md).

## Install and test

Python 3.11 or newer is required. The Stage 0/1 package has no runtime
dependencies outside the standard library.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The current suite has 107 tests covering multiline parsing, leakage
boundaries, Appendix D F1 corner cases, paired bootstrap decisions, strict screen
parsing, full fallback, retry accounting, tokens, latency, baseline prompts,
design-only artifacts, the optional Transformers adapter, H2 call reduction, the
GPU diagnostic path, and the Kaggle runner/controller.

On Windows, clone the benchmark with `git -c core.autocrlf=false clone ...`;
CRLF conversion changes the file hashes that Stage 2 verifies.

## Run on Kaggle GPUs

`python kaggle_exec/controller.py run --mode probe|smoke-gpu|run-design` pushes
a private GPU script kernel with the Kaggle CLI, waits for it, downloads its
outputs and writes a report. See [Kaggle execution](docs/kaggle_execution.md)
and the [debugging log](docs/debug_log.md).

## Obtain and audit the benchmark

Raw data is not redistributed because the upstream repository does not provide a
standard data license. Clone the official source outside this repository and
point the audit command to it:

```bash
git clone https://github.com/PedroCV/policy-compliance-requests-AI.git /tmp/policy-compliance-requests-AI
RULEGATE_DATA_ROOT=/tmp/policy-compliance-requests-AI
PYTHONPATH=src python -m rulegate.data audit --data-root "$RULEGATE_DATA_ROOT"
```

The loader reads the gold label only from the final nonblank line of each `>>!`
request block, so multiline code and internal label-like text remain inert. It
keeps trusted provenance and gold values in `DatasetRecord`; model-facing
`InferenceRequest` contains only a keyed-HMAC request ID and text. The HMAC key is
generated per load unless a trusted runner supplies a private key for cross-
process joins; it must never be put in a model prompt or exported result. Gold
and source metadata have no field in the model-facing type. Screening prompt
templates remain review-gated Stage 2 stubs, so their leakage review is required
before inference is enabled.

## Metrics and synthetic acceptance examples

The primary metric is the fixed mean of nine Appendix D policy-specific F1
scores. It implements the paper's unusual rule that gold `[2]`, prediction
`[2,7]` contributes `TP_2`, `FNstar_2`, and `FP2_7`. A paired 10,000-resample
bootstrap compares RuleGate with Full Sequential Prompting with Both.

Synthetic test fixtures currently demonstrate:

| Gate | Fixture result | Decision |
| --- | --- | --- |
| H1 preservation | delta `0.0000`, 95% CI `[0.0000, 0.0000]` | GO |
| H1 preservation | delta `-0.0741`, 95% CI `[-0.2222, 0.0000]` | INCONCLUSIVE |
| H1 preservation | delta `-0.1111`, 95% CI `[-0.1111, -0.1111]` | FAIL |
| H2 measured calls | full `18`, screened `8`, reduction `55.56%` | GO |
| H2 measured calls | full `9`, screened `6`, reduction `33.33%` | FAIL |

These are deliberately synthetic infrastructure checks, not benchmark results.
The H1 GO rule is `CI_lower(delta) > -0.05`; the negative sign corrects the old
proposal typo. H2 counts every observed backend attempt, including screening,
verification, retries, auxiliary calls, and full fallback.

## Package map

- `src/rulegate/data.py`, `schemas.py`: read-only loader, audit, trusted/model
  boundary.
- `src/rulegate/metrics.py`: Appendix D and request-level metrics.
- `src/rulegate/bootstrap.py`, `decisions.py`: paired H1 bootstrap and gates.
- `src/rulegate/model_backend.py`, `accounting.py`: fake backend and complete
  measured call ledger for H2.
- `src/rulegate/screening.py`, `verification.py`: strict routing, safe fallback,
  and a shared unchanged verifier path.
- `src/rulegate/prompts.py`, `baselines.py`: versioned documented prompt
  reimplementations and Stage 2 baseline runners. Exact author strings are not
  published.
- `src/rulegate/transformers_backend.py`: optional pinned Kaggle/Hugging Face
  backend with no import-time GPU dependency.
- `src/rulegate/stage2.py`: design-only baseline CLI (`smoke`, `smoke-gpu`,
  `run-design`) and safe artifact writer.
- `kaggle_exec/runner.py`, `kaggle_exec/controller.py`: remote Kaggle runner
  template and local push/poll/fetch/report controller.
- `src/rulegate/methods.py`: typed method contracts.
