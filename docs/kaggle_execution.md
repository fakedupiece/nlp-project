# Kaggle execution handoff (Stage 2+)

This guide runs the Stage 2 baselines on the 18 design requests. The notebook is
disabled by default. The Stage 2 CLI has no locked-test command; do not add one
until the model revision, prompts, parsers, decoding settings, and experiment
configuration are reviewed and frozen.

## Automated loop (preferred)

`kaggle_exec/controller.py` drives the whole cycle from a local checkout with
the Kaggle CLI (tested with Kaggle CLI 2.2.5). It renders
`kaggle_exec/runner.py` with a run config, pushes it as the private script
kernel `<kaggle-user>/rulegate-runner` with GPU T4 x2 and Internet enabled,
polls `kaggle kernels status`, downloads outputs with `kaggle kernels output`
(plus `kaggle kernels logs`), and writes a report.

```bash
python kaggle_exec/controller.py doctor
python kaggle_exec/controller.py run --mode probe
python kaggle_exec/controller.py run --mode smoke-gpu --limit 2 --model-revision <40-char HF commit>
python kaggle_exec/controller.py run --mode run-design --model-revision <40-char HF commit>
```

Modes:

| Mode | What runs remotely | Model weights |
| --- | --- | --- |
| `probe` | GPU/CUDA/VRAM inventory, clone code + benchmark at pinned commits, install, remote CPU tests, `HF_TOKEN` secret check, gated model access check (resolves revision, downloads `config.json`) | not loaded |
| `smoke-gpu` | probe + `rulegate.stage2 smoke-gpu --limit N`: real inference on the first N design requests; parse failures are recorded per request/policy instead of aborting | loaded |
| `run-design` | probe + strict `rulegate.stage2 run-design` on all 18 design requests | loaded |

There is no locked-test mode in the runner or the Stage 2 CLI.

Rules the loop enforces:

- The code commit must already be on `origin`; the remote session clones it
  and verifies `HEAD`. `run-design` refuses anything but an exact 40-character
  Hugging Face commit; probe/smoke may pass `resolve`, and the resolved commit
  is recorded and then passed to Stage 2 as an exact revision.
- Kaggle `COMPLETE` is not success. The runner writes `rulegate/status.json`
  stage by stage inside the session and always exits 0 so that Kaggle keeps the
  diagnostic outputs; the controller reads `final_status` and checks that the
  downloaded `run_id` matches the run it pushed.
- `HF_TOKEN` is read with Kaggle Secrets, passed only to the Stage 2 child
  process environment, and redacted from every line the runner logs.
- Raw model responses go to `rulegate/raw/raw_responses.jsonl` in the private
  kernel output and to the git-ignored local run folder. Committed reports
  contain counts, IDs and metadata only. Parse-failure excerpts are written to
  the git-ignored `diagnosis.local.md`.
- Local run folders: `results/dev/runs/kaggle/<run_id>/` (pushed kernel bundle,
  `run_record.json`, downloaded `output/`, report). Reviewed reports:
  `results/dev/reports/<run_id>.md`. History: one JSON line per run in
  `results/dev/kaggle_run_history.jsonl`.

GPU profile: `NvidiaTeslaT4` (2 x T4, 15 GB each, compute capability 7.5).
T4 has no native bfloat16, so the controller defaults to `--dtype float16`
with `--quantization none`; `device_map="auto"` splits the ~16 GB of
unquantized weights across both GPUs. The same dtype, quantization, revision
and decoding settings must be used for every compared condition.

One-time user actions (credentials never pass through the chat or the repo):

1. Authenticate the local Kaggle CLI: `kaggle auth login` (browser OAuth) or an
   API token in `~/.kaggle/access_token`. The Kaggle account must be phone
   verified to use GPUs and Internet in kernels.
2. Request access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face (manual
   approval) and create a read token.
3. After the first `probe` push creates the kernel, open it in the Kaggle editor,
   Add-ons > Secrets, add `HF_TOKEN` and enable it for this notebook. Later API
   pushes of the same kernel keep the attachment.

## Manual notebook path (fallback)

The rest of this guide describes the original manual notebook workflow.

## 1. Pin the code

Choose and record an exact reviewed Git commit as `<RULEGATE_COMMIT>`. In a Kaggle
notebook with Internet enabled, clone the public repository and check out that
commit. For a private repository, use an approved private-code import or manual
upload workflow; do not place a GitHub token in notebook cells or committed files.
Kaggle and GitHub do not synchronize automatically.

Install the local package and optional model dependencies from the checkout:

```bash
python -m pip install -e .
python -m pip install -r requirements-gpu.txt
```

If and only if the inspected GPU supports the chosen 4-bit path, use
`requirements-gpu-4bit.txt` and pass `--quantization 4bit` consistently.

Record the resolved package versions. Do not blindly replace Kaggle's PyTorch
build; first verify that its CUDA build matches the attached accelerator.

## 2. Check runtime and model authorization

Enable a GPU in Kaggle notebook settings, then inspect it at runtime. Do not
hardcode a GPU type or assume that one was allocated. Record:

- GPU name and memory;
- Python, CUDA, PyTorch, transformers, accelerate and optional bitsandbytes
  versions;
- RuleGate Git commit;
- exact Hugging Face model and revision;
- dtype, quantization, device mapping, seed and decoding settings.

Accept the access terms for `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face.
Create a Kaggle secret named `HF_TOKEN`; retrieve it through Kaggle's secret API
at runtime and pass it to the Hugging Face client without printing it. Never save
the token to the repository, notebook output, model log, or result artifact.

If no GPU is attached, model access is denied, or available memory is
insufficient, stop with an informative error. Select 4-bit loading only after
confirming that the actual GPU and installed libraries support it, then record
that choice for every compared condition.

Resolve the authorized model to an exact 40-character Hugging Face commit before
the run. A branch name such as `main` is rejected. The runner records requested
and resolved revisions.

## 3. Run design cases first

Set `RULEGATE_DATA_ROOT` to an authorized local copy of the official benchmark.
Start with the 18 design cases only. Use the same model revision and decoding
configuration for Single Prompting, Full Sequential Prompting with Both, plain
screening, and action-aware screening. Full and screened conditions must call the
same verifier implementation.

The notebook defaults `RUN_DESIGN_BASELINES = False` and
`RUN_LOCKED_TEST = False`. Enabling the design run must be an explicit edit.
The command requires a clean checkout matching the declared code commit and,
when the data root is a Git checkout, the audited benchmark commit.

```bash
python -m rulegate.stage2 run-design \
  --data-root "$RULEGATE_DATA_ROOT" \
  --output-dir /kaggle/working/rulegate-design-baselines \
  --code-commit <40-char RuleGate Git commit> \
  --model-revision <40-char Hugging Face commit> \
  --quantization none
```

Every backend attempt must write its audit record: opaque request ID, method,
template version/hash, model revision, quantization/settings, policy ID when
applicable, parsed result, token counts, elapsed time, parse state, retry index,
and fallback flags. Store raw responses only in a reviewed local location.

## 4. Export results explicitly

Write predictions, call ledgers, timings, run metadata and aggregate summaries to
the Kaggle output directory. Export only permitted artifacts, preferably using
opaque request IDs. Download them and add reviewed summaries to GitHub or provide
them to the evaluation environment. This is a manual step; a Kaggle run does not
update GitHub automatically.

Keep locked-test predictions sealed until the confirmatory run is complete. Never
use those 225 outputs to tune prompts, routing behavior, parsing, or thresholds.

Review `summary.json`, `run_metadata.json`, both prediction files, both JSONL
ledgers, and the separate trusted design index. A complete normal run has 18
Single Prompting calls and 162 Full Sequential with Both calls; retries increase
those observed totals and must remain visible. Stage 2 is not GO merely because
the process exits successfully: confirm every parse status and reconcile every
prediction with its call IDs.

## Restart after the tokenizer compatibility fix

Older code treated Hugging Face `BatchEncoding` as a tensor because it checked
only for `dict`. This can raise `AttributeError` on the first inference after
the model loads. The backend now accepts all `Mapping` outputs and preserves
both `input_ids` and `attention_mask`. Regression tests cover a `UserDict`
batch (the same container base as `BatchEncoding`), ordinary dictionaries,
and the tensor-only fallback. This fix has CPU test coverage; it is not a
claim of a completed Kaggle GPU run.

For an existing Kaggle session, set `CODE_COMMIT` to the full commit containing
the fix. Fetch and check out that commit in the existing clean `REPO_ROOT`
instead of rerunning the clone cell (which intentionally rejects an existing
directory). Leave the pinned model revision and the chosen quantization
unchanged. Use a new `OUTPUT_DIR` so that the failed run's artifacts remain
available. Run the configuration cell, then the GPU/Secrets cell, and finally
the execution cell. The Secrets cell must run again because the execution
cell removes `HF_TOKEN` from its environment on exit, including on failure.
