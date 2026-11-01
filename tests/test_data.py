from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

import pytest

from rulegate.data import (
    DatasetFormatError,
    audit_benchmark,
    load_benchmark,
    parse_request_file,
)


def _write_fixture(root: Path) -> Path:
    (root / "policy_list").mkdir(parents=True)
    (root / "testing_requests").mkdir()
    (root / "user_requests").mkdir()
    (root / "policy_list" / "policy_list.txt").write_text(
        "".join(f"{i}- Synthetic policy {i}.\n" for i in range(1, 10)),
        encoding="utf-8",
    )
    (root / "testing_requests" / "test_user_requests.txt").write_text(
        ">>! design compliant\n[]\n>>! design violation\n[1]\n",
        encoding="utf-8",
    )
    for part in range(1, 10):
        label = 9 if part == 1 else part
        text = f">>! request {part}\n[{label}]\n"
        if part == 2:
            text += ">>! duplicate text\n[]\n"
        if part == 3:
            text += ">>! duplicate text\n[]\n"
        (root / "user_requests" / f"part_{part}.txt").write_text(
            text,
            encoding="utf-8",
        )
    return root


def test_multiline_parser_only_treats_final_line_as_label(tmp_path: Path) -> None:
    path = tmp_path / "requests.txt"
    path.write_text(
        ">>! First line\n[]\ncode-like text stays inert\n[3]\n"
        ">>! A compliant request\n[]\n",
        encoding="utf-8",
    )

    parsed = parse_request_file(path)

    assert len(parsed) == 2
    assert parsed[0].text == "First line\n[]\ncode-like text stays inert"
    assert parsed[0].gold_violated_policy_ids == (3,)
    assert parsed[1].gold_violated_policy_ids == ()


def test_loader_creates_opaque_model_payload_without_gold_or_provenance(
    tmp_path: Path,
) -> None:
    dataset = load_benchmark(_write_fixture(tmp_path), strict_counts=False)
    source_record = dataset.test_records[0]

    assert source_record.source_file == "part_1.txt"
    assert source_record.gold_violated_policy_ids == (9,)
    assert re.fullmatch(r"rg-test-[0-9a-f]{20}", source_record.request_id)

    payload = asdict(source_record.to_inference_request())
    assert payload == {"request_id": source_record.request_id, "text": "request 1"}
    assert "gold" not in repr(payload).lower()
    assert "part_" not in repr(payload)


def test_opaque_ids_require_a_private_key_for_cross_load_stability(
    tmp_path: Path,
) -> None:
    root = _write_fixture(tmp_path)
    first = load_benchmark(root, strict_counts=False, id_key=b"a" * 32)
    repeated = load_benchmark(root, strict_counts=False, id_key=b"a" * 32)
    different = load_benchmark(root, strict_counts=False, id_key=b"b" * 32)

    assert first.test_records[0].request_id == repeated.test_records[0].request_id
    assert first.test_records[0].request_id != different.test_records[0].request_id

    with pytest.raises(ValueError, match="at least 16 bytes"):
        load_benchmark(root, strict_counts=False, id_key=b"too-short")


def test_audit_reports_distributions_duplicates_and_filename_mismatch(
    tmp_path: Path,
) -> None:
    dataset = load_benchmark(_write_fixture(tmp_path), strict_counts=False)

    audit = audit_benchmark(dataset)

    assert audit.policy_count == 9
    assert audit.design_count == 2
    assert audit.test_count == 11
    assert audit.test_distribution["compliant"] == 2
    assert audit.test_distribution["policy_9"] == 2
    assert audit.test_source_label_mismatch_ids == (dataset.test_records[0].request_id,)
    assert len(audit.test_exact_duplicate_groups) == 1
    duplicate = audit.test_exact_duplicate_groups[0]
    assert duplicate.request_ids == (
        dataset.test_records[2].request_id,
        dataset.test_records[4].request_id,
    )
    assert duplicate.gold_violated_policy_ids == ((), ())


def test_loader_rejects_missing_official_part(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    (root / "user_requests" / "part_9.txt").unlink()

    with pytest.raises(DatasetFormatError, match="expected part_1.txt..part_9.txt"):
        load_benchmark(root, strict_counts=False)


def test_parser_rejects_block_without_terminal_label(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(">>! unlabeled request\n", encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="missing request text or gold label"):
        parse_request_file(path)


def test_loader_rejects_multi_policy_gold(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    (root / "user_requests" / "part_1.txt").write_text(
        ">>! invalid official gold\n[1,2]\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetFormatError, match="at most one policy"):
        load_benchmark(root, strict_counts=False)
