"""The pilot only accepts a PASS/FAIL decision whose rule hash matches config/check.yaml."""

import json

import pytest

from pilot import check as CK


@pytest.fixture
def tmp_check_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(CK, "CHECK_DIR", tmp_path)
    return tmp_path


def _write(d, **kw):
    base = {"status": "PASS", "pilot_model": "Qwen/Qwen3-1.7B", "rule_hash": CK.rule_hash()}
    base.update(kw)
    (d / "decision.json").write_text(json.dumps(base))


def test_missing_decision(tmp_check_dir):
    with pytest.raises(FileNotFoundError):
        CK.verify_decision_for_pilot()


def test_invalid_decision_stops_pilot(tmp_check_dir):
    _write(tmp_check_dir, status="INVALID", pilot_model=None)
    with pytest.raises(SystemExit):
        CK.verify_decision_for_pilot()


def test_rule_hash_mismatch_stops_pilot(tmp_check_dir):
    _write(tmp_check_dir, rule_hash="0" * 64)
    with pytest.raises(SystemExit):
        CK.verify_decision_for_pilot()


def test_valid_decision_accepted(tmp_check_dir):
    _write(tmp_check_dir, status="FAIL", pilot_model="Qwen/Qwen3.5-4B")
    d = CK.verify_decision_for_pilot()
    assert d["pilot_model"] == "Qwen/Qwen3.5-4B" and len(d["_decision_file_sha256"]) == 64


def test_smoke_and_dev_paths_never_alias_the_real_decision():
    assert CK.decision_path(True).name == "decision_smoke.json"
    assert CK.decision_path("dev").name == "decision_dev.json"
    assert CK.decision_path(False).name == "decision.json"


def test_check_template_matches_pilot_template():
    from pilot.utils import load_pilot_config

    assert CK.load_check_config()["template"] == load_pilot_config(False)["tasks"]["truth"]["template"]
