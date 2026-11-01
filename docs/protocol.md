# RuleGate Stage 0/1 protocol

Status: protocol and CPU evaluation infrastructure only. No pretrained model was
loaded and no model inference was run on the 225 locked requests.

## Research question and fixed architecture

RuleGate tests whether one conservative policy-screening call can reduce detailed
compliance checks while preserving the result of the Cisneros-Velarde Sequential
Prompting with Both verifier.

The proposed condition is:

1. Send the request and all nine policies to one action-aware screener.
2. Assign exactly one of `POSSIBLE`, `UNCERTAIN`, or `EXCLUDE` to every policy.
3. Retain every `POSSIBLE` and `UNCERTAIN` policy; the candidate count is adaptive
   from zero through nine.
4. Run the same per-policy Sequential Prompting with Both verifier used by the
   full baseline on every retained policy.
5. Return the verifier's policy IDs, or `[]` when none are returned.

BM25, dense retrieval, reciprocal-rank fusion, similarity ranking, fixed Top-k,
fine-tuning, and a custom screened verifier are outside this protocol.

## H1: policy-specific F1 preservation

The primary quality metric is the arithmetic mean of nine policy-specific F1
scores from Appendix D of Cisneros-Velarde (2026):

`F1_i = 2 TP_i / (2 TP_i + FP1_i + FP2_i + FN_i + FNstar_i)`

`mean_policy_F1 = sum(F1_i for i in 1..9) / 9`

For policy `i`, `TP_i` counts a prediction containing the true policy even if it
also contains another policy. `FP1_i` is policy `i` predicted for a compliant
request. `FP2_i` is policy `i` predicted when a different policy is gold. `FN_i`
is a gold policy `i` with an empty prediction. `FNstar_i` is a gold policy `i`
with at least one other predicted policy, whether or not `i` is also present.
Consequently, gold `[2]` and prediction `[2, 7]` contributes both `TP_2` and
`FNstar_2`; it also contributes `FP2_7`. This differs from standard one-vs-rest
F1. Duplicate prediction IDs are normalized as a set after schema validation.
Invalid policy IDs are rejected. A zero denominator yields zero by RuleGate
convention because the paper does not specify that edge case. The nine-policy
denominator remains fixed in every bootstrap resample.

Let `delta = F1_RuleGate - F1_FullSequentialBoth`. Pair the two systems by
request and bootstrap request indices together for 10,000 deterministic
resamples. Recompute all nine F1 values for every resample and report the 2.5th
and 97.5th percentiles.

The confirmatory H1 gate is:

`lower_95_CI(delta) > -0.05`

The `-0.05` boundary corrects the submitted proposal's mistaken `+0.05` sign.
The margin is an engineering non-inferiority-style threshold, not an established
safety or medical margin. A confidence interval crossing `-0.05` is
`INCONCLUSIVE`; an interval entirely at or below it is `FAIL`.

## H2: measured total call reduction

All actual LLM attempts are recorded at the shared backend boundary. This
includes screening, per-policy verification, Single Prompting, any separately
generated plan, retries after failures, invalid-output handling, and all fallback
verification calls. Expected formulas such as `1 + k` are never substituted for
the measured ledger.

For the same `N` requests:

`call_reduction = 1 - total_calls_rulegate / total_calls_full`

The means-per-request form is algebraically identical because `N` is shared.
H2 is `GO` when reduction is at least 0.50. Input/output tokens, end-to-end
latency, and screening overhead are reported separately because fewer calls does
not prove fewer tokens or lower latency. If the measured baseline is exactly nine
calls per request and screening uses one call without retries, `mean(k) <= 3.5`
is a planning implication only.

## Dataset and leakage boundary

The official source is the public benchmark repository at
<https://github.com/PedroCV/policy-compliance-requests-AI>. Stage 0/1 was audited
at commit `2d1944f2efa362021f931638b724714f7b74f288`.

Trusted evaluation records contain text, gold IDs, source filename, and source
index. The inference boundary exposes only a keyed-HMAC request ID and raw request
text. The key is generated per load or supplied privately by a trusted runner; it
never enters model-facing data or exported artifacts. This prevents public source
coordinates from being reversed into a policy-label proxy. The inference type
never exposes gold IDs, `part_n.txt`, source indices, or metadata derived from the
label. The one-gold-policy construction is used for offline evaluation only and
must not constrain model output. Predictions may contain zero, one, or multiple
policy IDs.

The 18 design cases may later be used to validate prompt formatting. The 225
main cases are a locked test set: loading and auditing are permitted now, while
model execution, prompt tuning, threshold tuning, and test-informed changes are
forbidden until prompts/configuration are frozen.

Benchmark requests are inert strings. Code, commands, or instructions found in
them are never executed.

## Screen parsing and fallback

A valid screen must contain each integer policy ID 1 through 9 exactly once and
one exact routing label per ID. Missing, duplicate, extra, malformed, or
unparseable entries invalidate the complete screen. Invalid screens route all
nine policies to the unchanged verifier and record a failure reason,
`fallback_used=true`, and fallback type. A valid zero-candidate screen returns
`[]` and is counted separately for audit.

No prompt is claimed to reproduce unpublished author wording. The paper and
official repository publish no exact prompt templates. Stage 2 implements
versioned `single-prompting-v1-reimplementation` and
`sequential-both-v1-reimplementation` templates from the public Section 4 method
description; they remain subject to design-set review before experimental freeze.
Plain and action-aware screening templates are still future versioned
reimplementations. Section 4 describes one request-policy call for Sequential
Prompting with Both; it does not describe a separate plan-generation LLM call.
Full and screened conditions must use the same model revision, tokenizer,
quantization, decoding settings, parser, and verifier template.

## Additional reported metrics

Reports include exact-match request accuracy, compliant recall, noncompliant
request detection recall, gold-policy recall, policy support/F1, and the paper's
`TP`, `FP`, `FN`, `FNstar`, and `TN` categories. The paper is ambiguous about
whether global `TP` and `FNstar` overlap when both correct and wrong IDs appear;
RuleGate follows Appendix D and permits overlap, documenting that choice rather
than treating the categories as a conventional confusion matrix. Screening
reports additionally include gold-policy retention, screening-induced misses,
candidate-count distribution, valid zero-candidate screens, invalid-screen and
fallback rates, calls, tokens, and latency.

The paper's Llama-3.1-8B-Instruct + Sequential Prompting with Both values
(`F1=0.3640`, `ACC=53.33%`) are published references only, not reproduced
measurements.

## Stage gates

Stage 0 is `GO` when this protocol, frozen configuration fields, correction of
the H1 sign, fallback semantics, and measured call convention are present and
tested where applicable.

Stage 1 is `GO` when the official 9/18/225 records are parsed and audited without
leakage or deduplication, the Appendix D metric and global metrics pass synthetic
corner cases, and paired bootstrap/H1 plus measured H2 calculations pass
synthetic tests. No real model score is required or permitted at this gate.

## References

1. Pedro Cisneros-Velarde. “Policy Compliance of User Requests in Natural
   Language for AI Systems.” ACL Industry 2026, Sections 4–5 and Appendix D.
   <https://aclanthology.org/2026.acl-industry.21/>
2. Official benchmark repository.
   <https://github.com/PedroCV/policy-compliance-requests-AI>
