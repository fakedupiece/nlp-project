# Stage 2 baseline status

Decision: **GO** (2026-11-11). The strict 18-request design run completed on
Kaggle and passed every check of `rulegate.stage2_gate`.

## GO evidence

- Run `20261111T132516Z-run-design-8c58fa30` (kernel `edudum/rulegate-runner`
  v8, report `results/dev/reports/20261111T132516Z-run-design-8c58fa30.md`).
- Code `8c58fa307e3c239ffd319ca4f283493a8f0632b5` (clean checkout, verified
  remotely); benchmark `2d1944f2efa362021f931638b724714f7b74f288`.
- Model `meta-llama/Llama-3.1-8B-Instruct` @
  `0e9e39f249a16976918f6564b8830bc894c89659`, unquantized float16 on
  2 x Tesla T4 (14.6 GiB, cc 7.5), `device_map=auto`, load 131.7 s;
  greedy, temperature 0, `max_new_tokens=128`, seed 484654; torch
  2.11.0+cu130, transformers 4.57.6, accelerate 1.15.0.
- Prompts: `single-prompting-v2-reimplementation`,
  `sequential-both-v2-reimplementation` (`docs/prompt_versions.md`).
- Gate: 17/17 checks passed (complete artifacts, design split only, pinned
  clean code and model revision, one model configuration, greedy decoding,
  0 backend failures, 180/180 outputs parsed, token counts known, predictions
  reconciled 1:1 with ledger calls, no retries).

| Method | Calls | Input tok | Output tok | Latency s | Mean policy F1 | Exact-match acc | Compliant recall | Gold policy detected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single Prompting | 18 | 8,202 | 221 | 23.8 | 0.2457 | 0.1111 | 1/9 | 6/9 |
| Full Sequential with Both | 162 | 45,828 | 1,134 | 117.5 | 0.3407 | 0.1111 | 2/9 | 9/9 |

Design-split numbers (n = 18) are development evidence only, not benchmark
results; the paper's 225-request reference for Llama-3.1-8B Sequential with
Both (F1 0.3640, ACC 53.33%) is not comparable. Full Sequential with Both
over-predicts (2.56 policy IDs per request on average; FN* = 9, FP = 7) while
retaining the gold policy for every violating request.

Implication for Stage 3: the verifier is fixed. RuleGate can only keep or lose
the verifier's true positives, so screening recall on gold policies is the
critical quantity for H1, and fewer than 4.5 verified policies per request on
average (with one screening call) is needed for H2.

## History

Decision: **CONTINUE**. Baseline implementation and CPU integration are ready;
the required Llama-3.1-8B-Instruct run on the 18 design cases remains `NOT RUN`
until executed in Kaggle with a pinned model revision and available GPU.

2026-11-11: the automated Kaggle loop (`kaggle_exec/`) is implemented and CPU
tested. Planned sequence: `probe` -> `smoke-gpu --limit 2` -> `run-design`.
Remaining blockers are account-level: local Kaggle CLI authentication, Hugging
Face gated access to Llama-3.1-8B-Instruct, and the `HF_TOKEN` Kaggle secret.
See `docs/kaggle_execution.md` and the run history in
`results/dev/kaggle_run_history.jsonl`.

Update (same day): token delivery solved via a private Kaggle dataset; the
first real smoke run loaded the model but 0/20 v1-template outputs parsed.
Templates were revised to v2 (`docs/prompt_versions.md`). Stage 2 remains
CONTINUE until the strict 18-request run passes `rulegate.stage2_gate`.

## Implemented

- Versioned `single-prompting-v1-reimplementation` template: all nine policies
  and one request in one call; zero or more policy IDs may be returned.
- Versioned `sequential-both-v1-reimplementation` template: one request-policy
  call combining Policy Association and Request Plan reframings.
- Strict JSON parsers, retries, parse failure logging, prompt/template hashes,
  token counts and per-call timing.
- One shared `SequentialBothVerifier` factory for the full baseline and later
  screened conditions.
- Optional Kaggle/Hugging Face backend with exact 40-character model commit,
  CUDA check, greedy decoding, safe token handling, runtime/device/dtype metadata,
  and explicit optional 4-bit loading.
- A design-only CLI. There is no command for the 225 locked requests.
- Separate prediction, trusted design index, call ledger, aggregate summary and
  reproducibility metadata artifacts. Raw request text and credentials are not
  written to model-result artifacts.
- If a real design run aborts, the attempted-call ledgers and a sanitized failure
  record are preserved for cost reconciliation and diagnosis.

The prompt strings are documented reimplementations of the public Section 4
method description. The official sources contain no exact author templates.

## Official-design fake-backend smoke test

Command:

```bash
python -m rulegate.stage2 smoke \
  --data-root /path/to/policy-compliance-requests-AI \
  --output-dir results/dev/runs/stage2-fake-smoke
```

Observed on the official 18 design requests:

| Condition | Requests | Calls | Calls/request | Parse failures | Backend failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single Prompting | 18 | 18 | 1.0 | 0 | 0 |
| Full Sequential with Both | 18 | 162 | 9.0 | 0 | 0 |

All 180 call records reconciled with predictions and contained a valid parse
status. Fake responses always predicted compliance and did not use gold labels.
Token counts are explicitly `unknown`. These are infrastructure results, not
development model scores and not evidence for H1 or H2.

## GO requirement still outstanding

Run the design-only command in a clean checkout of a known Git commit:

```bash
python -m rulegate.stage2 run-design \
  --data-root /path/to/policy-compliance-requests-AI \
  --output-dir /kaggle/working/rulegate-design-baselines \
  --code-commit <40-char RuleGate Git commit> \
  --model-revision <40-char Hugging Face commit> \
  --quantization none
```

`HF_TOKEN` must come from Kaggle Secrets. The command verifies a clean matching
code checkout, verifies the official benchmark commit when Git metadata is
available, requires CUDA, and runs only the 18 design requests. Stage 2 reaches
GO after these real development calls complete, every output parses, all calls
reconcile, and the resulting metadata/artifacts are reviewed. Prompt changes
after that review must be versioned; the 225-case test remains sealed.
