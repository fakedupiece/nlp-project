# Debugging log

Failures met while building and running the Kaggle loop, with root cause and
fix. Newest last. GPU submissions are counted per debugging cycle (limit: three
before reporting a blocker).

## 2026-11-11 — Local Windows workspace setup

### D1. Benchmark hash mismatch on Windows

- Symptom: `python -m rulegate.stage2 smoke` failed with
  `benchmark file hash mismatch for policy_list/policy_list.txt`.
- Root cause: the machine's global `core.autocrlf=true` rewrote the benchmark's
  LF files to CRLF on checkout. The committed blob still hashed to the expected
  `a993b8e0…`, so the data was intact; only the working-tree bytes changed.
- Fix: clone the benchmark with `git -c core.autocrlf=false clone …` and set
  `core.autocrlf=false` in both local checkouts. The Kaggle runner also clones
  with `core.autocrlf=false`. No code or hash change was needed.

### D2. Kaggle CLI 2.2.5 import failure

- Symptom: every `kaggle` command crashed with `ModuleNotFoundError: No module
  named 'six'`.
- Root cause: `kaggle==2.2.5` imports `six` in `kaggle/models/upload_file.py`
  but does not declare it as a dependency.
- Fix: `pip install six` in the project virtualenv.

## 2026-11-11 — Hugging Face token not reaching the Kaggle kernel

Debugging cycle: 3 submissions (smoke `d130bf8` v2 on GPU; probes `edb0cc0`
v3 and `585ac4d` v4 on CPU only, so no GPU quota was spent on diagnosis).

### D3. `HF_TOKEN` Kaggle secret unavailable in API-pushed kernels

- Symptom: `hf_secret` stage failed in every run, including after the secret
  was added and enabled in the Kaggle editor.
- Diagnosis: the runner originally swallowed the lookup exception. After
  recording it: `ConnectionError: Connection error trying to communicate with
  service.`, caused by `HTTPError 400 Bad Request` from the user-secrets
  service. Session facts: run type `Batch`, user-secrets token present.
- Root cause: Kaggle Secrets attached in the editor are not delivered to
  kernels pushed with `kaggle kernels push`; `kernel-metadata.json` has no
  secrets field (Kaggle community guide "Cron-Schedule Kaggle Notebooks That
  Need Secrets", citing Kaggle API issue #582). Not a code bug in RuleGate.
- Considered alternatives: Kaggle-hosted Llama 3.1 weights would avoid the
  token but cannot be verified byte-for-byte against the pinned Hugging Face
  revision, because Hugging Face masks LFS SHA-256 values of gated repos for
  anonymous callers.
- Fix (pending user approval of the deviation from "Kaggle Secrets"): the
  runner still tries Kaggle Secrets first, then reads `hf_token.txt` from the
  private dataset `<user>/rulegate-hf-token` attached via `dataset_sources`.
  The controller refuses to push unless that dataset exists and is private.
  The value is redacted from all logs and never copied to outputs.

### D4. Hugging Face 401 with the dataset token

- Run `20261111T130638Z-smoke-gpu-8845d8d4` (v5): token delivered from the
  private dataset, but `model_access` failed with `GatedRepoError 401`.
- Diagnosis from recorded token shape only: the parsed value was 6 characters
  without the `hf_` prefix, from a 45-byte file. The reader had taken the wrong
  part of a multi-part file (e.g. a label line). 401 means unauthenticated,
  not "access not approved" (403).
- Fix (`27c3766`, `096c815`): extract the `hf_...` token anywhere in the file;
  record only raw length, line count and whether the pattern was found; give
  distinct guidance for 401 and 403. Run v6 then passed `model_access`.

## 2026-11-11 — First real inference: prompt format

### D5. Llama-3.1-8B-Instruct ignored the v1 JSON-only instruction

- Run `20261111T130925Z-smoke-gpu-096c8150` (v6, GPU): model loaded in 138 s
  (float16, 2 x T4, 6.7 + 8.3 GiB); 20 real calls, all `invalid_*`, every one
  at 128 output tokens (~10 s/call). Raw responses (private) were Python code
  or step-by-step reasoning.
- Root cause: prompt presentation (JSON-variable labels, no system message),
  not decoding, numerics or the parser.
- Fix: versioned v2 templates and a fence-tolerant parser; see
  `docs/prompt_versions.md`. Debugging cycle for this issue starts here.
