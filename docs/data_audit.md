# Official benchmark data audit

Audit scope: public repository
`PedroCV/policy-compliance-requests-AI`, commit
`2d1944f2efa362021f931638b724714f7b74f288` (2026-04-24). This is a data-only
audit. No request was sent to an LLM.

## Source layout and parser observations

The repository contains:

- `policy_list/policy_list.txt` with nine numbered policies. The upstream README
  refers to `policy_list_txt`, which is not the actual filename.
- `testing_requests/test_user_requests.txt` with 18 design records.
- `user_requests/part_1.txt` through `part_9.txt`, with 25 locked-test records
  in each file.

Each request block begins with `>>!`. The final nonblank line in the block is the
gold label (`[]` or `[n]`). Requests can span lines, contain blank lines and code,
and can contain label-like text internally. The loader therefore parses complete
blocks and validates only their final nonblank line. One official label has
trailing whitespace; whitespace around the final label is accepted.

## Counts

| Split | Total | Compliant `[]` | Violating | Status |
| --- | ---: | ---: | ---: | --- |
| Design | 18 | 9 | 9 | one compliant and one violation per policy |
| Locked test | 225 | 119 (52.8889%) | 106 (47.1111%) | preserved exactly |

Locked-test gold distribution:

| Gold | Count |
| --- | ---: |
| `[]` | 119 |
| `[1]` | 11 |
| `[2]` | 10 |
| `[3]` | 12 |
| `[4]` | 11 |
| `[5]` | 14 |
| `[6]` | 12 |
| `[7]` | 12 |
| `[8]` | 12 |
| `[9]` | 12 |

Each `part_n.txt` contains 25 records. Its compliant/violating counts are:

| Source | Compliant | Gold policy `n` |
| --- | ---: | ---: |
| `part_1.txt` | 14 | 11 |
| `part_2.txt` | 15 | 10 |
| `part_3.txt` | 13 | 12 |
| `part_4.txt` | 14 | 11 |
| `part_5.txt` | 11 | 14 |
| `part_6.txt` | 13 | 12 |
| `part_7.txt` | 13 | 12 |
| `part_8.txt` | 13 | 12 |
| `part_9.txt` | 13 | 12 |

All policy IDs 1–9 are present, no request is empty, and no nonempty main-set
label contradicts its containing `part_n.txt`. Labels are read from file content,
never inferred from filenames. No exact text occurs in both design and locked
splits.

## Duplicate audit

There is one exact duplicate group in the locked test. It consists of the third
and sixth records in `part_7.txt` (zero-based positions 2 and 5); both have gold
`[7]`. Their normalized text SHA-256 prefix is `95a5c61b8d89`. Both official
records remain in the
225-item set. Raw request text is omitted here because the upstream repository
does not include a standard redistribution license.

As a report-only gold example, the first design record has offline gold `[1]`;
its inference object contains only a keyed-HMAC ID and request text. Opaque IDs
are run-specific unless a private trusted-runner key is supplied. The key, source
filename, index, and gold value remain in trusted evaluation storage.

## Rights and handling

The upstream repository has no `LICENSE` or `COPYING` file. Its README describes
the benchmark as primarily provided for fair use in academic research and
includes a use-at-own-risk disclaimer. Raw benchmark files are therefore ignored
by this repository and must be obtained from the official source. Audit summaries
and opaque identifiers may be committed; raw requests should not be republished
without permission.
