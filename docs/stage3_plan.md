# Stage 3 plan: plain and action-aware screening

Status (2026-11-11): **implemented, CPU-tested and approved; first design run pending.**
Stage 2 is GO (`docs/stage2_status.md`).

## What Stage 3 must show on the design split

RuleGate = one conservative screening call over the request and all nine
policies, then the unchanged Stage 2 verifier (`sequential-both-v2`) on every
`POSSIBLE`/`UNCERTAIN` policy; invalid screens fall back to all nine policies.
Stage 3 develops and checks the screening prompts on the 18 design requests
before anything is frozen for the 225-request confirmatory run.

Quantities reported per condition (`rulegate.stage3`):

- calls, input/output tokens and latency from the measured ledger (screening,
  verification, retries and fallback calls all counted);
- Appendix D mean policy F1 and request metrics;
- screening recall: violating requests whose gold policy was retained;
- screening-induced misses: gold policies the full verifier detected but the
  screen excluded, and all full-verifier positives excluded;
- candidate-count distribution, label counts, valid zero-candidate screens,
  invalid screens and fallbacks;
- verdict mismatches between the screened and full runs on retained policies
  (a determinism check, since both use the same greedy verifier);
- indicative H1 (paired bootstrap delta F1, 95% CI) and H2 (call reduction)
  against the Full Sequential with Both reference run **in the same session**,
  so request IDs, model load and settings are identical.

## Implemented

| Piece | Where |
| --- | --- |
| `plain-screening-v1`, `action-aware-screening-v1` templates, shared screening system message, 384-token screening cap | `src/rulegate/screening_prompts.py` |
| Action-aware parser (exactly `action`, `target`, `content`, `labels`; same strict label rules; fence-tolerant) | `src/rulegate/screening.py` |
| Per-call `max_new_tokens` (recorded in the ledger); verifier keeps 128 | `model_backend.py`, `transformers_backend.py` |
| Full reference + screened runners, screening report, artifacts, H1/H2 indicative comparison, CLI (`smoke`, `run-design`) | `src/rulegate/stage3.py` |
| Kaggle mode `stage3-design` (`--conditions`, `--limit`) and report section | `kaggle_exec/runner.py`, `kaggle_exec/controller.py` |
| Tests | `tests/test_stage3.py`, Stage 3 cases in the Kaggle tests |

Design choices:

- **Isolation of action-awareness.** Plain and action-aware prompts share the
  system message, blocks, inert-data instruction and the exact labelling rules;
  only the action/target/content step and output fields differ (tested).
- **Analysis fields stay private.** `action`/`target`/`content` may paraphrase
  the request, so they are validated but not exported; they remain in the
  private raw-response file.
- **Screening token cap.** A nine-label reply is ~130-200 tokens, so the
  screening call gets `max_new_tokens=384`; the verifier keeps the Stage 2 cap
  of 128 in every condition. Greedy decoding is unchanged.
- **No retries** (`max_attempts=1`), as in Stage 2; an invalid screen uses the
  full nine-policy fallback and is counted.
- One model load runs the full reference and both screening conditions:
  162 + 2 x (18 + verified) calls, roughly 6-8 minutes on 2 x T4.

## Stage 2 facts that shape Stage 3

On the design split the verifier detects the gold policy in 9/9 violating
requests but also predicts 2.56 policies per request (7/9 compliant requests
flagged). Screening can only keep or lose verifier positives: excluding a
false-positive policy raises F1, excluding a gold policy lowers it. H2 needs a
mean of at most 3.5 verified policies per request (one screening call, no
retries) for a 50% call reduction.

## Decisions (research lead, 2026-11-11)

1. Action-aware screening is RuleGate (primary); plain screening is the
   ablation; Single Prompting runs in the same session as the efficiency
   control.
2. The pre-registered per-policy F1 reading in `docs/protocol.md` stays the
   primary metric. The goal is to carry out the proposal, not to replicate
   every unreported detail of Cisneros-Velarde; H1 compares RuleGate with our
   own full baseline under the same verifier and metric.
3. Screening prompt wording follows the proposal: POSSIBLE = potential
   violation, UNCERTAIN = plausible but unclear violation, EXCLUDE = no
   plausible violation based on the request; explicit and implicit signs, not
   word overlap; action-aware adds the requested action, target and relevant
   explicit or implied content.
4. Screening-induced false negatives use the proposal's definition: requests
   that full Sequential with Both classified exactly right but the screened
   system missed because the gold policy was excluded. Looser counts are also
   reported.

## Stage 3 gate (encoded in `rulegate.stage3.evaluate_stage3_gate`)

From the stage-gate plan, applied to action-aware screening (plain screening
is reported with the same checks as the H3 ablation):

- all 18 design requests run;
- gold policy retained for every violating design request (9/9);
- mean verified policies per request <= 3.5 (nine-call baseline, one screening
  call, no retries);
- invalid screens <= 5% (0/18 expected);
- every verifier output parsed.

GO if all pass. PIVOT_WARNING if more than half of the requests keep >= 7
policies. Otherwise CONTINUE: revise screening prompts on the design split
only, at most two further versions per template, selecting by retention first
and then mean verified policies; wording shared by both templates changes in
both. With 9 violating requests, 9/9 retention has a 95% lower bound of 0.66:
this is an engineering gate, not a recall claim.

Known limitation carried forward: the v2 verifier over-flags on the design
split (7/9 compliant requests flagged), unlike the paper's Llama-8B profile.
The screener gate above does not depend on it; end-to-end F1 comparisons do.
A bounded verifier-fidelity review is planned after Stage 3.

```bash
python kaggle_exec/controller.py run --mode stage3-design --model-revision 0e9e39f249a16976918f6564b8830bc894c89659
```
