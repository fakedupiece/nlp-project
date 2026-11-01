"""Read-only loader and audit for the official Cisneros-Velarde benchmark."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import secrets
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .schemas import (
    POLICY_IDS,
    BenchmarkDataset,
    DatasetRecord,
    Policy,
    Split,
)


EXPECTED_POLICY_COUNT = 9
EXPECTED_DESIGN_COUNT = 18
EXPECTED_TEST_COUNT = 225

_POLICY_LINE = re.compile(r"^\s*(\d+)\s*-\s*(.*?)\s*$")
_LABEL_LINE = re.compile(r"^\s*\[\s*(?P<body>(?:\d+\s*(?:,\s*\d+\s*)*)?)\]\s*$")
_PART_NAME = re.compile(r"^part_(\d+)\.txt$")


class DatasetFormatError(ValueError):
    """Raised when the upstream text format is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class ParsedRequest:
    text: str
    gold_violated_policy_ids: tuple[int, ...]
    source_index: int


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    text_sha256: str
    request_ids: tuple[str, ...]
    gold_violated_policy_ids: tuple[tuple[int, ...], ...]
    source_files: tuple[str, ...]
    source_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DataAudit:
    policy_count: int
    design_count: int
    test_count: int
    design_distribution: dict[str, int]
    test_distribution: dict[str, int]
    test_source_counts: dict[str, int]
    design_missing_policy_ids: tuple[int, ...]
    test_missing_policy_ids: tuple[int, ...]
    test_exact_duplicate_groups: tuple[DuplicateGroup, ...]
    cross_split_exact_duplicate_groups: tuple[DuplicateGroup, ...]
    test_source_label_mismatch_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_policies(path: str | Path) -> tuple[Policy, ...]:
    """Load numbered policy lines, requiring each policy ID exactly once."""

    path = Path(path)
    policies: list[Policy] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        match = _POLICY_LINE.fullmatch(line)
        if not match:
            raise DatasetFormatError(
                f"{path}:{line_number}: expected '<policy_id>- <policy text>'"
            )
        policies.append(Policy(policy_id=int(match.group(1)), text=match.group(2)))

    ids = tuple(policy.policy_id for policy in policies)
    if ids != POLICY_IDS:
        raise DatasetFormatError(f"{path}: expected ordered policy IDs 1..9; got {ids!r}")
    return tuple(policies)


def _parse_label(line: str, *, location: str) -> tuple[int, ...]:
    match = _LABEL_LINE.fullmatch(line)
    if not match:
        raise DatasetFormatError(f"{location}: request block must end with [] or [id]")
    body = match.group("body").strip()
    if not body:
        return ()
    ids = tuple(int(item.strip()) for item in body.split(","))
    if len(ids) != len(set(ids)):
        raise DatasetFormatError(f"{location}: duplicate gold policy IDs in {line!r}")
    invalid = [policy_id for policy_id in ids if policy_id not in POLICY_IDS]
    if invalid:
        raise DatasetFormatError(
            f"{location}: gold policy IDs must be in 1..9; got {invalid!r}"
        )
    return ids


def parse_request_file(path: str | Path) -> tuple[ParsedRequest, ...]:
    """Parse ``>>!`` request blocks whose final nonblank line is the gold label.

    Only the final line of a block is interpreted as a label.  A label-looking
    line inside a multiline request therefore remains inert request text.
    """

    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    marker_indices = [index for index, line in enumerate(lines) if line.startswith(">>!")]
    if not marker_indices:
        raise DatasetFormatError(f"{path}: no request markers ('>>!') found")
    if any(line.strip() for line in lines[: marker_indices[0]]):
        raise DatasetFormatError(f"{path}: unexpected content before first request marker")

    parsed: list[ParsedRequest] = []
    boundaries = [*marker_indices, len(lines)]
    for source_index, (start, end) in enumerate(
        zip(boundaries, boundaries[1:]), start=1
    ):
        first_line = lines[start][3:]
        if first_line.startswith(" "):
            first_line = first_line[1:]
        block_lines = [first_line, *lines[start + 1 : end]]
        while block_lines and not block_lines[-1].strip():
            block_lines.pop()
        location = f"{path}:request {source_index}"
        if len(block_lines) < 2:
            raise DatasetFormatError(f"{location}: missing request text or gold label")
        gold = _parse_label(block_lines[-1], location=location)
        text = "\n".join(block_lines[:-1]).strip()
        if not text:
            raise DatasetFormatError(f"{location}: empty request text")
        parsed.append(
            ParsedRequest(
                text=text,
                gold_violated_policy_ids=gold,
                source_index=source_index,
            )
        )
    return tuple(parsed)


def _records_for_file(
    path: Path,
    *,
    split: Split,
    id_key: bytes,
) -> tuple[DatasetRecord, ...]:
    prefix = "design" if split is Split.DESIGN else "test"
    records: list[DatasetRecord] = []
    for parsed in parse_request_file(path):
        if len(parsed.gold_violated_policy_ids) > 1:
            raise DatasetFormatError(
                f"{path}:request {parsed.source_index}: official gold permits at "
                "most one policy ID"
            )
        opaque_digest = hmac.new(
            id_key,
            f"{split.value}\0{path.name}\0{parsed.source_index}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:20]
        records.append(
            DatasetRecord(
                request_id=f"rg-{prefix}-{opaque_digest}",
                split=split,
                text=parsed.text,
                gold_violated_policy_ids=parsed.gold_violated_policy_ids,
                source_file=path.name,
                source_index=parsed.source_index,
            )
        )
    return tuple(records)


def load_benchmark(
    root: str | Path,
    *,
    strict_counts: bool = True,
    id_key: bytes | None = None,
) -> BenchmarkDataset:
    """Load the official checkout without modifying or deduplicating it.

    IDs are keyed so public source coordinates cannot be brute-forced into label
    proxies. By default a fresh key is generated for this in-memory load. A
    trusted runner may supply the same private key when IDs must be stable across
    processes; that key must never cross the model boundary or enter artifacts.
    """

    root = Path(root)
    if id_key is None:
        id_key = secrets.token_bytes(32)
    if not isinstance(id_key, bytes) or len(id_key) < 16:
        raise ValueError("id_key must be at least 16 bytes when supplied")
    policies = load_policies(root / "policy_list" / "policy_list.txt")
    design_records = _records_for_file(
        root / "testing_requests" / "test_user_requests.txt",
        split=Split.DESIGN,
        id_key=id_key,
    )

    user_requests_dir = root / "user_requests"
    numbered_parts: dict[int, Path] = {}
    for path in user_requests_dir.glob("part_*.txt"):
        match = _PART_NAME.fullmatch(path.name)
        if not match:
            continue
        number = int(match.group(1))
        if number in numbered_parts:
            raise DatasetFormatError(f"duplicate part number {number} in {user_requests_dir}")
        numbered_parts[number] = path
    if set(numbered_parts) != set(POLICY_IDS):
        raise DatasetFormatError(
            f"{user_requests_dir}: expected part_1.txt..part_9.txt; "
            f"found parts {sorted(numbered_parts)!r}"
        )

    test_records: list[DatasetRecord] = []
    for part_number in POLICY_IDS:
        records = _records_for_file(
            numbered_parts[part_number],
            split=Split.TEST,
            id_key=id_key,
        )
        if strict_counts and len(records) != EXPECTED_TEST_COUNT // len(POLICY_IDS):
            raise DatasetFormatError(
                f"expected 25 test requests in {numbered_parts[part_number].name}; "
                f"got {len(records)}"
            )
        test_records.extend(records)

    if strict_counts and len(design_records) != EXPECTED_DESIGN_COUNT:
        raise DatasetFormatError(
            f"expected {EXPECTED_DESIGN_COUNT} design requests; got {len(design_records)}"
        )
    if strict_counts and len(test_records) != EXPECTED_TEST_COUNT:
        raise DatasetFormatError(
            f"expected {EXPECTED_TEST_COUNT} test requests; got {len(test_records)}"
        )

    return BenchmarkDataset(
        policies=policies,
        design_records=design_records,
        test_records=tuple(test_records),
    )


def _distribution(records: Sequence[DatasetRecord]) -> dict[str, int]:
    distribution = {"compliant": 0, **{f"policy_{i}": 0 for i in POLICY_IDS}}
    for record in records:
        if not record.gold_violated_policy_ids:
            distribution["compliant"] += 1
        for policy_id in record.gold_violated_policy_ids:
            distribution[f"policy_{policy_id}"] += 1
    return distribution


def _duplicate_groups(records: Iterable[DatasetRecord]) -> tuple[DuplicateGroup, ...]:
    by_text: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        by_text[record.text].append(record)
    groups: list[DuplicateGroup] = []
    for text, duplicates in by_text.items():
        if len(duplicates) < 2:
            continue
        groups.append(
            DuplicateGroup(
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                request_ids=tuple(record.request_id for record in duplicates),
                gold_violated_policy_ids=tuple(
                    record.gold_violated_policy_ids for record in duplicates
                ),
                source_files=tuple(record.source_file for record in duplicates),
                source_indices=tuple(record.source_index for record in duplicates),
            )
        )
    return tuple(sorted(groups, key=lambda group: group.request_ids))


def audit_benchmark(dataset: BenchmarkDataset) -> DataAudit:
    """Return aggregate audit metadata without reproducing raw request text."""

    design_distribution = _distribution(dataset.design_records)
    test_distribution = _distribution(dataset.test_records)
    source_counts = Counter(record.source_file for record in dataset.test_records)

    test_texts = {record.text for record in dataset.test_records}
    cross_split_records = [
        record
        for record in (*dataset.design_records, *dataset.test_records)
        if record.text in test_texts
    ]
    cross_split_groups = tuple(
        group
        for group in _duplicate_groups(cross_split_records)
        if any(request_id.startswith("rg-design-") for request_id in group.request_ids)
        and any(request_id.startswith("rg-test-") for request_id in group.request_ids)
    )

    mismatches: list[str] = []
    for record in dataset.test_records:
        match = _PART_NAME.fullmatch(record.source_file)
        if not match or not record.gold_violated_policy_ids:
            continue
        part_number = int(match.group(1))
        if any(policy_id != part_number for policy_id in record.gold_violated_policy_ids):
            mismatches.append(record.request_id)

    return DataAudit(
        policy_count=len(dataset.policies),
        design_count=len(dataset.design_records),
        test_count=len(dataset.test_records),
        design_distribution=design_distribution,
        test_distribution=test_distribution,
        test_source_counts=dict(sorted(source_counts.items())),
        design_missing_policy_ids=tuple(
            policy_id
            for policy_id in POLICY_IDS
            if design_distribution[f"policy_{policy_id}"] == 0
        ),
        test_missing_policy_ids=tuple(
            policy_id
            for policy_id in POLICY_IDS
            if test_distribution[f"policy_{policy_id}"] == 0
        ),
        test_exact_duplicate_groups=_duplicate_groups(dataset.test_records),
        cross_split_exact_duplicate_groups=cross_split_groups,
        test_source_label_mismatch_ids=tuple(mismatches),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="print a JSON data audit")
    audit_parser.add_argument("--data-root", type=Path, required=True)
    audit_parser.add_argument(
        "--no-strict-counts",
        action="store_true",
        help="allow non-official counts (useful only for local fixtures)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "audit":
        dataset = load_benchmark(
            args.data_root,
            strict_counts=not args.no_strict_counts,
        )
        print(json.dumps(audit_benchmark(dataset).to_dict(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
