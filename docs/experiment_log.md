# Experiment log

## 2026-11-01 — Stage 0/1 implementation

Classification: CPU infrastructure and synthetic validation. No model inference.

- Read the finalized RuleGate context and retained the screening-to-unchanged-
  verifier architecture.
- Inspected official benchmark commit
  `2d1944f2efa362021f931638b724714f7b74f288`.
- Loaded and audited nine policies, 18 design records, and 225 locked-test records.
- Kept the one exact duplicate pair; no test record was removed.
- Implemented leak-free inference records, Appendix D metrics, paired bootstrap,
  H1 decisions, backend-boundary call accounting, H2 decisions, strict screen
  parsing, fallback behavior, and deterministic fake backends.
- Used only synthetic unit fixtures for metric, bootstrap, screening, retry,
  fallback, and call-reduction behavior.
- GPU/model execution: `NOT RUN (Kaggle is the planned runtime)`.
- Locked 225-request inference: `NOT RUN (configuration is not frozen)`.

Validation command: `PYTHONPATH=src python -m pytest -q`.

Observed result: 36 tests passed. Pytest emitted one deprecation
warning from the preinstalled `anyio` plugin; no RuleGate warning or failure was
reported. `python -m compileall -q src tests`, notebook JSON validation, and
`git diff --check` also passed.

Synthetic gate examples (not benchmark measurements):

- H1 GO: delta `0.0000`, 95% CI `[0.0000, 0.0000]`.
- H1 INCONCLUSIVE: delta `-0.0741`, CI `[-0.2222, 0.0000]`.
- H1 FAIL: delta `-0.1111`, CI `[-0.1111, -0.1111]`.
- H2 GO: 18 measured full calls versus 8 screened calls (`55.56%` reduction).
- H2 FAIL: 9 measured full calls versus 6 screened calls (`33.33%` reduction).

## 2026-11-01 — Stage 2 baseline implementation

Classification: source review, CPU tests and fake-backend integration. No model
inference.

- Read the uploaded proposal, five-slide presentation and official paper.
- Confirmed the proposal's `+0.05` sentence is a typo; slide 4 uses `-0.05`.
- Confirmed that exact author prompt strings are not published and that
  Sequential Prompting with Both is one call per request-policy pair.
- Added versioned Single Prompting and Sequential-with-Both reimplementation
  templates, strict parsers, the shared verifier factory, and a pinned optional
  Transformers backend for Kaggle.
- Added a design-only runner and artifact contract. No locked-test execution route
  exists in the Stage 2 CLI.
- Ran both baselines over all 18 official design requests with a fake backend:
  18/18 Single Prompting calls and 162/162 sequential calls were logged and
  parsed, with zero fallback, parse or backend failures. Token usage was marked
  unknown. No fake prediction was reported as a model result.
- Full test suite: 68 tests passed; one unrelated preinstalled `anyio` deprecation
  warning remained.
- Real Llama/Kaggle design run: `NOT RUN`.

## 2026-11-11 — Local-to-Kaggle execution loop

Classification: engineering infrastructure, CPU tests and local fake-backend
smoke. No model inference.

- Audited the repository: Stage 0/1 GO; Stage 2 code complete, real run `NOT
  RUN` (no Kaggle evidence in the repository). Latest completed stage: 1.
- Local workspace: Windows 11, Python 3.12.10 (installed per-user), Git 2.37,
  Kaggle CLI 2.2.5 in `.venv`. Benchmark cloned at the audited commit with LF
  line endings (see `docs/debug_log.md` D1, D2).
- Existing suite: 71 passed. Local `rulegate.stage2 smoke` on the official 18
  design requests: 18 + 162 synthetic calls, all parsed.
- Added `kaggle_exec/runner.py` (remote script kernel) and
  `kaggle_exec/controller.py` (push, poll, fetch, report, history).
- Added `rulegate.stage2 smoke-gpu`: real-model diagnostic on the first N design
  requests using the same baselines and shared verifier; parse failures are
  recorded per request/policy instead of aborting.
- Added explicit `--dtype` (T4 profile: `float16`, unquantized), multi-GPU
  device metadata, load time, and an opt-in private raw-response recorder.
- Model revision candidate: `0e9e39f249a16976918f6564b8830bc894c89659`
  (Hugging Face `main`, public API). Gated with manual approval.
- Full suite after changes: 107 passed.
- Kaggle GPU runs: `NOT RUN` — local Kaggle CLI is not authenticated yet.

## 2026-11-11 — Kaggle probe (run `20261111T123836Z-probe-bb70169f`)

Classification: remote environment probe. No model weights loaded, no inference.

- Code `bb70169f`; kernel `edudum/rulegate-runner` v1 (private, GPU, Internet).
- Kaggle status `COMPLETE`; runner status `probe_incomplete` (expected).
- GPU: 2 x Tesla T4, 14.6 GiB each, compute capability 7.5, no native bf16
  (confirms the float16 profile). torch `2.11.0+cu130`, transformers `4.57.6`,
  accelerate `1.15.0`, huggingface_hub `0.36.2` (all preinstalled versions
  satisfied the requirements).
- Code and benchmark clones verified at the pinned commits; remote CPU tests
  `107 passed`.
- Blocker: `HF_TOKEN` Kaggle secret not attached to the kernel (user action).
  Report: `results/dev/reports/20261111T123836Z-probe-bb70169f.md`.

## 2026-11-11 — First real Llama inference (smoke, 2 design requests)

Classification: development smoke on the design split. Not a Stage 2 result.

- Runs v2–v4: blocked on token delivery (Kaggle Secrets are not delivered to
  API-pushed kernels; `docs/debug_log.md` D3). Switched, with user approval,
  to a private Kaggle dataset holding the token. v5: invalid token value (D4).
- Run `20261111T130925Z-smoke-gpu-096c8150` (code `096c815`, kernel v6):
  Llama-3.1-8B-Instruct @ `0e9e39f249a16976918f6564b8830bc894c89659`,
  unquantized float16 on 2 x Tesla T4 (cc 7.5), `device_map=auto`
  (15/21 modules), load 138 s, greedy, `max_new_tokens=128`, seed 484654.
- Calls: Single Prompting 2 (949 input / 256 output tokens, 22.6 s);
  Full Sequential with Both 18 (4,945 / 2,304 tokens, 173.7 s). 0 backend
  failures; **0/20 outputs parsed** (v1 prompt format, D5).
- Next: v2 prompt templates (`docs/prompt_versions.md`), then repeat the smoke.
