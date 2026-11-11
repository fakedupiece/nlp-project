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
