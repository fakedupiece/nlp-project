# Frozen protocol for the locked test (Stage 5)

Freeze: **`rulegate-freeze-v1`** (2026-11-11), manifest `configs/freeze_v1.json`.
This document and the manifest are fixed before any model output on the 225
locked requests exists. Nothing below may be changed after the locked run; a
change requires a new freeze and is reported as such.

## What is frozen

The manifest is generated from the code (`python -m rulegate.freeze write`) and
checked by a CPU test and by the locked-test runner (`verify_manifest`). It pins:

- **Benchmark:** official repository commit `2d1944f2…`, data-file manifest
  hash, test split, 225 requests (119 compliant, 106 violating).
- **Model:** `meta-llama/Llama-3.1-8B-Instruct` @
  `0e9e39f249a16976918f6564b8830bc894c89659`, unquantized float16 on Kaggle
  2 x Tesla T4 (`device_map=auto`), greedy (temperature 0, no sampling),
  `max_new_tokens` 128 (screening call: 384), seed 484654, one attempt, no
  retries.
- **Prompts (ID and SHA-256 of system message + template):**
  `single-prompting-v2-reimplementation`,
  `sequential-both-v2-reimplementation` (verifier, shared by every
  condition), `plain-screening-v1`, `action-aware-screening-v1`.
- **Code:** SHA-256 of every module that shapes prompts, model calls, parsing,
  predictions, metrics or decisions (16 files, listed in the manifest).

All frozen prompt hashes and settings equal those of the Stage 3 GO run
`20261111T161052Z-stage3-design-9469a85c` (verified).

## Conditions (one session, same 225 requests, one run each)

| Condition | Role |
| --- | --- |
| Action-aware screening + verifier | **RuleGate, confirmatory** |
| Full Sequential Prompting with Both | reference comparator for H1 and H2 |
| Plain screening + verifier | ablation (H3), exploratory |
| Single Prompting | efficiency control, descriptive |

## Pre-registered analysis

- **Primary metric:** mean of the nine Appendix D policy-specific F1 scores,
  reading as in `docs/protocol.md` (zero denominator -> 0.0).
- **H1 (quality):** delta = F1(action-aware) - F1(full). Paired bootstrap over
  requests, 10,000 resamples, seed 484654, percentile 95% CI, nine-policy
  denominator in every resample. **GO** if CI lower bound > -0.05; **FAIL** if
  CI upper bound <= -0.05; otherwise **INCONCLUSIVE**. (The proposal's "+0.05"
  is a typo.)
- **H2 (efficiency):** call reduction = 1 - calls(action-aware) / calls(full)
  from the measured ledgers, counting every screening, verification, retry and
  fallback call. **GO** if >= 0.50.
- **Overall:** research **GO** only if H1 GO and H2 GO. H1 INCONCLUSIVE with
  H2 GO is reported as **INCONCLUSIVE**; anything else is **NOT SUPPORTED**.
  Negative and inconclusive results are reported as they are.
- **Secondary (descriptive):** exact-match accuracy, compliant recall,
  violating-request detection recall, per-policy F1 and Appendix D counts;
  gold-policy retention; screening-induced false negatives (proposal
  definition: full Sequential exactly right, gold policy excluded by the
  screen) and looser counts; candidate-count distribution; zero-candidate and
  invalid screens; fallbacks; input/output tokens and latency for every
  condition; plain vs action-aware (H3); comparison with Single Prompting.
- **Not part of the decision:** the paper's published values (F1 0.3640,
  ACC 53.33%) are reported as references only.

## Execution and integrity rules

- The only route to the 225 requests is `rulegate.stage5 run-test`, launched
  via `python kaggle_exec/controller.py run --mode locked-test
  --confirm-locked-test`. All settings come from the verified manifest; none
  can be passed on the command line. The working tree must be clean.
- **One confirmatory run per freeze.** The controller refuses a second run
  once a run of this freeze has completed (`results/dev/kaggle_run_history.jsonl`).
  A rerun is allowed only after an infrastructure failure that produced no
  complete results, with a written `--rerun-reason`; every attempt is logged.
- Finished conditions are checkpointed during the run; raw responses stay in
  the private Kaggle output and the git-ignored local run folder.
- No prompt, parser, threshold or configuration change is made in response to
  locked-test results. Exploratory analyses after the run are labelled as such.
- The report is written to `results/locked_test/` and committed after review.

## Known limitations carried into the report

- Prompts are documented reimplementations; exact author prompts are
  unpublished. The v2 verifier flagged 7/9 compliant design requests, unlike
  the paper's conservative Llama-8B profile; a more literal candidate (v3)
  flagged even more and was rejected (`docs/prompt_versions.md`). Because
  screening removes verifier positives, an over-flagging verifier favours H1.
- The design-split H2 margin was one call (80 vs 81 allowed).
- Screening dominates RuleGate latency; call reduction is not cost reduction.
- The paper's tables imply 105/120 violating/compliant requests; the released
  data has 106/119 and is used unchanged.

Expected runtime on 2 x T4: about 2.5-3 hours of GPU time for the four
conditions (about 5,000 model calls); kernel timeout 6 hours.
