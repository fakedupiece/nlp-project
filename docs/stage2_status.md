# Stage 2 baseline status

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
