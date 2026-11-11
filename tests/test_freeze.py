from __future__ import annotations

import json

import pytest

from rulegate.freeze import (
    DEFAULT_MANIFEST_PATH,
    FROZEN_SOURCE_FILES,
    FreezeError,
    build_manifest,
    manifest_sha256,
    verify_manifest,
)


def test_committed_manifest_matches_the_current_code():
    manifest = verify_manifest(DEFAULT_MANIFEST_PATH)
    assert manifest == build_manifest()
    assert len(manifest_sha256(manifest)) == 64


def test_manifest_pins_the_agreed_configuration():
    manifest = build_manifest()
    assert manifest["model"]["revision"] == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert manifest["model"]["dtype"] == "float16"
    assert manifest["model"]["temperature"] == 0.0
    assert manifest["prompts"]["sequential_both_verifier"]["template_id"] == (
        "sequential-both-v2-reimplementation"
    )
    assert manifest["prompts"]["action_aware_screening"]["template_id"] == (
        "action-aware-screening-v1"
    )
    assert manifest["analysis"]["h1"]["margin"] == 0.05
    assert "CI_low > -0.05" in manifest["analysis"]["h1"]["rule"]
    assert manifest["analysis"]["h2"]["threshold"] == 0.5
    assert manifest["conditions"]["action_aware_screening"].startswith("primary")
    assert set(manifest["frozen_source_sha256"]) == set(FROZEN_SOURCE_FILES)


def test_any_change_to_the_frozen_configuration_is_detected(tmp_path):
    tampered = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    tampered["model"]["dtype"] = "bfloat16"
    tampered["frozen_source_sha256"]["metrics.py"] = "0" * 64
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(FreezeError) as excinfo:
        verify_manifest(path)
    assert "model.dtype" in str(excinfo.value)
    assert "frozen_source_sha256.metrics.py" in str(excinfo.value)
