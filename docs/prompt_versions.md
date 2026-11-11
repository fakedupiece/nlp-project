# Prompt versions

All templates are documented reimplementations of Cisneros-Velarde (2026),
Section 4. The paper publishes no prompt strings. Every model call records its
template ID in the call ledger; run metadata records the template SHA-256,
which covers both the system message and the user template. Templates are
revised only with design-set evidence (the 18 official design requests). The
225 locked-test requests are never used.

## v1 — `single-prompting-v1-reimplementation`, `sequential-both-v1-reimplementation`

- Inputs as JSON under `POLICIES_JSON:` / `POLICY_JSON:` / `USER_REQUEST_JSON:`
  labels; no system message; output instruction "Return only one JSON object".
- Status: **retired 2026-11-11**.
- Evidence: Kaggle run `20261111T130925Z-smoke-gpu-096c8150` (code `096c815`,
  Llama-3.1-8B-Instruct @ `0e9e39f2…`, float16, greedy, `max_new_tokens=128`),
  first two design requests. 20/20 calls unparseable and every call hit the
  128-token limit: Single Prompting wrote Python (`import json` ...); the
  verifier wrote Python functions or step-by-step analysis despite "do not
  output a plan or reasoning". The JSON-variable presentation reads as a
  programming task to this model.

## v2 — `single-prompting-v2-reimplementation`, `sequential-both-v2-reimplementation` (active)

Unchanged from v1: the method (one all-policy call; one call per request-policy
pair), the information given to the model, both reframings combined in one
verifier judgment, the "zero, one or several policies" wording, the inert-data
instruction, and the output schemas `{"violated_policy_ids": [...]}` and
`{"violated": true|false}`.

Changed presentation only:

1. One shared system message for both methods, stating that the model is a
   classifier that never fulfils requests and replies with exactly one JSON
   object (no code, explanation or Markdown).
2. Policies and the request in plain-text `<policies>`/`<policy>` and
   `<user_request>` blocks instead of JSON-escaped strings. A literal
   `</user_request>` inside a request is escaped so it cannot close its block.
3. The format instruction is the last thing in the prompt, with the literal
   reply objects to copy.
4. Parser revision: a reply that is exactly one JSON object inside a single
   Markdown code fence is accepted; any other prose still fails.

Also fixed: template rendering no longer rejects request text containing
`{{`; placeholders are validated against the template, not the rendered text.

The full Sequential-with-Both baseline and the later screened conditions use
the same `make_sequential_both_verifier` factory, so the verifier prompt,
system message and parser stay identical across compared conditions.

## Verifier v3 — `sequential-both-v3-reimplementation` (fidelity revision, active)

Why (design-split evidence, Stage 2 run `20261111T132516Z-run-design-8c58fa30`):
v2 said "violated" for 37 of 153 non-gold request-policy pairs (24%),
concentrated on policy 4 (9/17) and policy 6 (9/17). Over nine independent
checks this flags 7/9 compliant requests, whereas the paper's Llama-3.1-8B
S.P. with Both flags about 19% of compliant requests (about 2% per pair).
The profile resembles the paper's plain S.P. for Llama-8B, where the Policy
Association reframing had not yet been applied.

What changed: Section 4 describes each reframing as replacing one side of the
compliance question (policy -> the forbidden action it refers to; request ->
a plan that would fulfil it) and combines the two because they are
independent. v2 listed the two reframings as separate views and then asked
whether the request violates the policy, which the model can read as "yes if
either view says yes". v3 asks the single composed question: is the plan that
would fulfil the request associated with the forbidden action the policy
refers to? Unchanged: one call per request-policy pair, system message,
delimited blocks, inert-data instruction, output schema `{"violated": ...}`
and parser. Single Prompting stays on v2.

Pre-declared adoption rule (written before any v3 run): v3 replaces v2 because
it is the more literal reading of the published method description, provided
every v3 output parses in the strict 18-request Stage 2 run. Adoption does not
depend on F1, on gate outcomes, or on closeness to the paper's numbers. If
outputs fail to parse, at most one formatting-only revision (v4) is allowed.
The v2 runs remain recorded, and Stage 2 and Stage 3 are rerun with v3 because
the verifier is shared by every condition.

## Screening v1 — `plain-screening-v1`, `action-aware-screening-v1` (Stage 3, not yet run)

RuleGate's own method, so these are new templates rather than
reimplementations. Shared: a screening system message (separate from the
verifier's), `<policies>`/`<user_request>` blocks, the inert-data instruction,
and identical labelling rules (POSSIBLE, UNCERTAIN, EXCLUDE; EXCLUDE only when
confident; UNCERTAIN when in doubt; zero, one or several policies may need
review). The action-aware template adds one step (identify action, target and
content, then judge each policy against that) and three short output fields.
Output: `{"labels": [...nine entries...]}`, plus `action`/`target`/`content`
for action-aware. Generation cap 384 tokens for the screening call only.

Before the first run, the label definitions and cues were aligned with the
proposal (POSSIBLE = potential violation; UNCERTAIN = plausible but unclear
violation; EXCLUDE = no plausible violation based on the request; consider
explicit and implicit signs rather than word overlap; action-aware content =
relevant explicit or implied request content). No run used the earlier draft,
so the version ID stays `v1`; the template hash in run metadata identifies
the exact text.
