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
