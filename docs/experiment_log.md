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
