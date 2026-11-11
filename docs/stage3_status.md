# Stage 3 screening pilot status

Decision: **GO** for action-aware screening (RuleGate) on the provisional
engineering gate of the stage-gate plan (2026-11-11). Plain screening (ablation)
does not meet the 3.5 mean-verified bound. Design split only (18 requests);
this is development evidence, not a hypothesis test.

## Run

- `20261111T161052Z-stage3-design-9469a85c` (kernel `edudum/rulegate-runner`
  v9, report `results/dev/reports/20261111T161052Z-stage3-design-9469a85c.md`).
- Code `9469a85ce96a025ba1945ca3e325dc4422463c1c`; benchmark `2d1944f2`.
- Same configuration as Stage 2: `meta-llama/Llama-3.1-8B-Instruct` @
  `0e9e39f249a16976918f6564b8830bc894c89659`, float16, 2 x Tesla T4, greedy,
  verifier `sequential-both-v2-reimplementation` with `max_new_tokens=128`;
  screening `plain-screening-v1` / `action-aware-screening-v1` with a 384-token
  cap. All four conditions ran in one session on the same requests.

## Results (design split, n = 18; 9 violating)

| Condition | Calls | Calls/req | Input tok | Output tok | Latency s | Mean policy F1 | Exact-match acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Sequential with Both | 162 | 9.00 | 45,828 | 1,134 | 123.2 | 0.3407 | 0.111 |
| Action-aware screening + verifier (RuleGate) | 80 | 4.44 | 32,697 | 3,292 | 292.5 | 0.3980 | 0.222 |
| Plain screening + verifier (ablation) | 98 | 5.44 | 35,630 | 2,998 | 265.7 | 0.3948 | 0.278 |
| Single Prompting (efficiency control) | 18 | 1.00 | 8,202 | 221 | 23.3 | 0.2457 | 0.111 |

| Screening measure | Action-aware | Plain |
| --- | ---: | ---: |
| Gold-policy retention | **9/9** | 9/9 |
| Mean verified policies (all / compliant / violating) | **3.44** / 2.11 / 4.78 | 4.44 / 3.00 / 5.89 |
| Requests with >= 7 verified | 1 | 4 |
| Invalid screens (full fallback) | 0/18 | 0/18 |
| Valid zero-candidate screens | 2 (both compliant, final `[]` correct) | 2 |
| Screening-induced false negatives (proposal definition) | 0 | 0 |
| Full-verifier positives excluded (all non-gold) | 11 | 11 |
| Verdict mismatches vs full on retained policies | 0 | 0 |
| Measured call reduction vs full | 50.6% | 39.5% |
| ΔF1 vs full, 95% paired bootstrap CI | +0.057 [+0.000, +0.077] | +0.054 [+0.000, +0.077] |

## Gate (action-aware)

PASS all 18 requests; PASS 9/9 retention; PASS mean verified 3.44 <= 3.5;
PASS 0/18 invalid screens; PASS every verifier output parsed. No PIVOT
warning (1 request kept >= 7 policies). Action-aware passed on its first
version, so no screening prompt iteration was used.

## Interpretation and caveats

- **H3 direction (descriptive).** With the same retention (9/9), action-aware
  screening verified one policy fewer per request than plain screening
  (3.44 vs 4.44) and used 18 fewer calls.
- **H2 margin is one call.** 80 calls against 81 allowed (50% of 162). The
  locked-test result can fall on either side.
- **Calls are not cost.** Each screening call writes ~159 tokens (13.6 s on
  2 x T4) versus ~7 tokens (0.76 s) per verifier call. Screening is 84% of
  RuleGate latency; RuleGate is 2.4x slower and writes 2.9x more output tokens
  than full Sequential, while reading 29% fewer input tokens. Single Prompting
  is far cheaper (18 calls, 23 s) but has lower F1 on this split.
- **The F1 gain comes from the verifier's false positives.** The v2 verifier
  over-flags (7/9 compliant requests flagged in full mode). Every excluded
  full-verifier positive was a non-gold policy, so screening removed false
  positives and raised F1. With a more conservative verifier (as in the paper)
  the same screen would yield a smaller gain. The verifier-fidelity review
  is closed: v3 flagged more, the research lead kept v2, and this run used v2
  (template hash verified), so these Stage 3 results stand.
- **Small sample.** 9/9 retention has a 95% lower bound of 0.66; the bootstrap
  CI on 18 requests is wide and not confirmatory.

## Next (stage-gate plan)

- Stage 4 integration checks are largely covered by this run (calls and
  predictions reconciled per condition, identical verifier verdicts on
  retained policies, candidate and latency logs). Verifier decision made
  (v2). Remaining: freeze prompts, parsers and configuration for the locked
  225-request run (Stage 5).
