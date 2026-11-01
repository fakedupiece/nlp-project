"""Typed, validated records for benchmark and inference data.

``DatasetRecord`` belongs to trusted evaluation code and contains gold labels and
provenance.  Model-facing code receives only ``InferenceRequest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


POLICY_IDS: tuple[int, ...] = tuple(range(1, 10))


class Split(str, Enum):
    """Official benchmark splits."""

    DESIGN = "design"
    TEST = "test"


def _validate_policy_ids(policy_ids: tuple[int, ...]) -> None:
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("policy IDs must be unique")
    invalid = [
        policy_id
        for policy_id in policy_ids
        if isinstance(policy_id, bool) or policy_id not in POLICY_IDS
    ]
    if invalid:
        raise ValueError(f"policy IDs must be integers in 1..9; got {invalid!r}")


@dataclass(frozen=True, slots=True)
class Policy:
    policy_id: int
    text: str

    def __post_init__(self) -> None:
        _validate_policy_ids((self.policy_id,))
        if not self.text.strip():
            raise ValueError("policy text must not be empty")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """The complete request payload allowed to cross the model boundary."""

    request_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.text.strip():
            raise ValueError("request text must not be empty")


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Trusted evaluation record; never serialize this object into a prompt."""

    request_id: str
    split: Split
    text: str
    gold_violated_policy_ids: tuple[int, ...]
    source_file: str
    source_index: int

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.text.strip():
            raise ValueError("request text must not be empty")
        _validate_policy_ids(self.gold_violated_policy_ids)
        if len(self.gold_violated_policy_ids) > 1:
            raise ValueError("official benchmark gold permits at most one policy ID")
        if not self.source_file.strip():
            raise ValueError("source_file must not be empty")
        if self.source_index < 1:
            raise ValueError("source_index is one-based and must be positive")

    def to_inference_request(self) -> InferenceRequest:
        """Drop gold labels and provenance at the trusted/model boundary."""

        return InferenceRequest(request_id=self.request_id, text=self.text)


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    policies: tuple[Policy, ...]
    design_records: tuple[DatasetRecord, ...]
    test_records: tuple[DatasetRecord, ...]

    def __post_init__(self) -> None:
        policy_ids = tuple(policy.policy_id for policy in self.policies)
        if policy_ids != POLICY_IDS:
            raise ValueError(f"expected policies ordered 1..9; got {policy_ids!r}")
        all_ids = [
            record.request_id
            for record in (*self.design_records, *self.test_records)
        ]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("request IDs must be unique across splits")
