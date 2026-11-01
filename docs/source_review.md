# Stage 2 source reconciliation

Reviewed on 2026-11-01:

- `Team2_Proposal.pdf` (three pages);
- `RuleGate_Proposal_Presentation_Updated_5_Slides.pptx` (five slides and notes);
- Cisneros-Velarde (2026), `2026.acl-industry.21.pdf`, especially Sections 3–5
  and Appendices B–D;
- the official benchmark repository at
  <https://github.com/PedroCV/policy-compliance-requests-AI>.

These source documents are research references. The repository protocol remains
the executable contract for code and tests.

## Reconciled decisions

The proposal and slides agree on the final system: one conservative screen over
the request and all nine policies, followed by the unchanged Cisneros Sequential
Prompting with Both verifier for every `POSSIBLE` or `UNCERTAIN` policy. Candidate
count is adaptive from zero through nine. Invalid screen output falls back to all
nine policy checks.

The proposal body contains the already identified H1 sign typo: it asks for the
lower confidence bound to be above `+0.05`. Slide 4 states the intended rule,
which matches the finalized project context:

`lower_95_CI(F1_RuleGate - F1_FullSequentialBoth) > -0.05`

Code and reporting use the slide/context rule. The proposal PDF is retained as a
source but does not override this explicit correction.

The proposal mentions a preliminary two-example TF-IDF sanity check. It labels
that check as motivation rather than a benchmark result and says lexical
retrieval was removed. RuleGate therefore contains no BM25, TF-IDF retrieval,
dense retrieval, RRF, or fixed Top-k implementation.

## What the paper specifies

Section 4 defines:

- **Single Prompting:** list all policies and the request in one prompt and ask
  which policies are violated.
- **Sequential Prompting:** make one request-policy compliance call per policy.
- **Policy Association reframing:** ask whether the request is associated with
  the forbidden action referenced by the policy.
- **Request Plan reframing:** ask whether a step-by-step list of actions that
  would fulfill the request would violate the policy.
- **Sequential Prompting with Both:** combine both reframings in the single
  request-policy judgment.

The paper does not describe a separate plan-generation call for Sequential
Prompting with Both. Stage 2 therefore keeps exactly one call per request-policy
pair. Any future auxiliary call would be a RuleGate deviation and must be logged
and counted.

Appendix B specifies temperature zero. It reports that the paper ran
Llama-3.1-8B-Instruct on one H100, but this does not justify assuming the same
hardware, revision, dtype, or quantization on Kaggle. Every actual run must record
those values.

## Prompt fidelity boundary

The paper, proposal, presentation, ACL page, and audited benchmark checkout do not
provide exact author prompt strings. A 2026-11-01 recheck found only the `main`
branch at `2d1944f2efa362021f931638b724714f7b74f288`, with no tags, releases,
other branches, code, prompt files, or ACL supplement. The Stage 2 templates are
therefore versioned **documented reimplementations** of the public method
description. They must never be called exact author prompts or a bit-for-bit
replication.

The reimplementation follows these declared choices:

- benchmark requests are delimited and treated as inert text;
- no benchmark examples are inserted into prompts;
- Single Prompting may return zero, one, or multiple policy IDs;
- Sequential Prompting with Both considers both public reframings in one call and
  returns one strict boolean verdict for the policy under review;
- prompt text, output schema, parser, model revision, tokenizer, decoding,
  quantization and retry rules are versioned before a measured run.

Stage 2 can validate implementation and run the 18 design cases. It cannot claim
faithful reproduction of unpublished prompt wording.
