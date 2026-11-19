"""Freeze / verify / deviation on a throwaway git repository."""

import os
import shutil
import subprocess

import pytest

from pilot import prereg
from pilot.utils import REPO_ROOT

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@localhost", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@localhost"}


def _git(root, *a):
    return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, env={**os.environ, **ENV})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    for f in prereg.FROZEN:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        if f != "prereg/PREREGISTRATION.md":
            shutil.copy2(REPO_ROOT / f, tmp_path / f)
    prereg.write_render(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_verify_refuses_without_freeze(repo):
    v = prereg.verify(repo)
    assert not v["ok"] and "FROZEN.json" in v["reason"]
    ok, _, _ = prereg.enforce(repo, "a reason")
    assert not ok                                   # a deviation cannot replace a missing freeze


def test_freeze_refuses_dirty_tree(repo):
    with open(repo / "config" / "v2.yaml", "a") as f:
        f.write("\n# uncommitted\n")
    assert prereg.freeze(repo) == 2


def test_freeze_then_verify_then_edit(repo):
    assert prereg.freeze(repo) == 0
    assert _git(repo, "tag", "-l", prereg.TAG).stdout.strip() == prereg.TAG
    assert prereg.verify(repo)["ok"]
    with open(repo / "src" / "pilot" / "analysis_v2.py", "a") as f:
        f.write("\n# edited after the freeze\n")
    v = prereg.verify(repo)
    assert not v["ok"] and v["changed"] == ["src/pilot/analysis_v2.py"]
    ok, _, _ = prereg.enforce(repo, None)
    assert not ok
    ok, _, dev = prereg.enforce(repo, "fixed a typo")
    assert ok and dev["changed_files"] == ["src/pilot/analysis_v2.py"]
    assert "fixed a typo" in (repo / "prereg" / "deviations.md").read_text()
    assert prereg.freeze(repo) == 2                 # tag exists: no silent re-freeze


def test_rendered_preregistration_is_in_sync():
    assert (REPO_ROOT / "prereg" / "PREREGISTRATION.md").read_text() == prereg.render(REPO_ROOT)
