"""v3 preregistration: render prereg/v3/PREREGISTRATION.md from prereg/v3/hypotheses.yaml, freeze (hash + commit +
tag `prereg-v3-frozen`), verify and enforce. Same procedure as v2 (`prereg.py`), with v3 paths.

Frozen files: the v3 hypotheses and their rendering, config/v3.yaml, config/v3_sae_input_norm.yaml (the input
scaling measured by the smoke test), the v3 analysis code and the v2 / stats functions it calls. The run plan
(priority order, dropped blocks, time budget) lives in hypotheses.yaml and config/v3.yaml, so it is frozen too.
--confirm refuses unless every hash matches; a change needs --deviation "<reason>", appended to
prereg/v3/deviations.md and listed at the top of the report. Nothing is ever uploaded anywhere.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

DIR = "prereg/v3"
FROZEN = [f"{DIR}/hypotheses.yaml", f"{DIR}/PREREGISTRATION.md", "config/v3.yaml", "config/v3_sae_input_norm.yaml",
          "src/pilot/analysis_v3.py", "src/pilot/analysis_v2.py", "src/pilot/stats.py"]
TAG = "prereg-v3-frozen"


def _git(root: Path, *args, check=True) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def render(root: Path) -> str:
    h = yaml.safe_load((Path(root) / DIR / "hypotheses.yaml").read_text())
    m = h["metric"]
    rs = h["reference_setting"]
    L = [f"# {h['title']}", "", f"Status: **{h['status']}** · drafted {h['date_drafted']}", "",
         "Generated from `prereg/v3/hypotheses.yaml` by `bash run_pilot.sh --v3 --render-prereg`; do not edit by hand.", "",
         "## 1. Headline", "", h["headline"], "", "## 2. Design", "",
         f"- Reference setting: {rs['model']}, task {rs['task']}, {rs['depth']:.0%} depth, {rs['strategy']} ablation, "
         f"{rs['sae']} SAE. Each test varies only its own factor; everything else stays at the reference.",
         f"- Reference task fallback (if it fails a gate on either model): {' → '.join(h['reference_task_order'])}.",
         f"- Models: {', '.join(h['models'])}. Bases: {', '.join(h['bases'])}. Rules: {', '.join(h['rules'])} "
         "(random is the floor).",
         f"- Budgets: {', '.join(str(k) for k in h['budgets'])} units; no doubling.",
         f"- Measure: {m['name']}. {m['definition']}",
         f"- Damage: {m['damage']}. Cost: {m['cost']}.",
         f"- δ = {m['delta']}; α = {m['alpha']}; {m['bootstrap']} bootstrap resamples. {m['reliable']}", "",
         "## 3. Inclusion rules (mechanical)", ""]
    L += [f"- {k}: {v}" for k, v in h["inclusion_rules"].items()]
    L += ["", "## 4. Dependence tests", "", "| ID | Claim | Compared | Holds if |", "|---|---|---|---|"]
    for k, v in h["tests"].items():
        L.append(f"| {k} | {v['claim']} | {v['compared']} | {v['holds_if']} |")
    L += ["", "## 5. Headline label", ""] + [f"- `{k}`: {v}" for k, v in h["label"].items()]
    L += ["", "## 6. Secondary analyses (never change the label)", ""] + [f"- {k}: {v}" for k, v in h["secondary"].items()]
    L += ["", "## 7. Run plan (frozen)", "", f"- Priority order: {' → '.join(h['run_plan']['priority'])}.",
          f"- Dropped before the run: {', '.join(h['run_plan']['dropped_blocks']) or 'none'}.",
          f"- Time guard: {h['run_plan']['time_guard']}.", "", "## 8. Known limitations", ""]
    L += [f"- {x}" for x in h["known_limitations"]]
    L += ["", "## 9. Frozen files", "", f"SHA-256 hashes are recorded in `{DIR}/FROZEN.json` at the commit tagged "
          f"`{TAG}`:", ""] + [f"- `{f}`" for f in FROZEN]
    return "\n".join(L) + "\n"


def write_render(root: Path) -> Path:
    p = Path(root) / DIR / "PREREGISTRATION.md"
    p.write_text(render(root))
    return p


def freeze(root: Path) -> int:
    root = Path(root)
    try:
        _git(root, "rev-parse", "HEAD")
    except RuntimeError as e:
        print(f"[freeze] REFUSED: not a git repository with a commit ({e})")
        return 2
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("[freeze] REFUSED: the git working tree has uncommitted changes:\n" + dirty + "\nCommit (or revert) them first.")
        return 2
    tracked = set(_git(root, "ls-files").splitlines())
    missing = [f for f in FROZEN if f not in tracked]
    if missing:
        print(f"[freeze] REFUSED: frozen files are not committed: {missing}")
        return 2
    if (root / DIR / "PREREGISTRATION.md").read_text() != render(root):
        print("[freeze] REFUSED: prereg/v3/PREREGISTRATION.md is out of date with hypotheses.yaml. Run "
              "`bash run_pilot.sh --v3 --render-prereg`, review, commit, then freeze.")
        return 2
    if _git(root, "tag", "-l", TAG):
        print(f"[freeze] REFUSED: tag {TAG} already exists (a freeze happened). Changes after a freeze need "
              "`--confirm --deviation \"<reason>\"`; to re-freeze deliberately, delete the tag yourself first.")
        return 2
    commit = _git(root, "rev-parse", "HEAD")
    rec = {"files": {f: _sha(root / f) for f in FROZEN}, "git_commit": commit,
           "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tag": TAG}
    (root / DIR / "FROZEN.json").write_text(json.dumps(rec, indent=2) + "\n")
    _git(root, "add", f"{DIR}/FROZEN.json")
    _git(root, "commit", "-m", f"Freeze v3 preregistration ({rec['utc']})", "--", f"{DIR}/FROZEN.json")
    _git(root, "tag", "-a", TAG, "-m", f"v3 preregistration frozen at {rec['utc']}")
    print(f"[freeze] frozen {len(FROZEN)} files at commit {commit[:12]}; FROZEN.json committed and tagged {TAG}.")
    for f, h in rec["files"].items():
        print(f"           {h[:16]}  {f}")
    print("[freeze] Reminder: uploading prereg/v3/PREREGISTRATION.md to OSF or AsPredicted adds an independent "
          "timestamp. This script never uploads anything.")
    return 0


def verify(root: Path) -> dict:
    root = Path(root)
    fp = root / DIR / "FROZEN.json"
    if not fp.exists():
        return {"ok": False, "reason": f"no {DIR}/FROZEN.json: run `bash run_pilot.sh --v3 --freeze` first", "changed": []}
    try:
        if not _git(root, "tag", "-l", TAG):
            return {"ok": False, "reason": f"git tag {TAG} is missing", "changed": []}
        tagged = _git(root, "show", f"{TAG}:{DIR}/FROZEN.json")
    except RuntimeError as e:
        return {"ok": False, "reason": f"cannot read the frozen record from tag {TAG}: {e}", "changed": []}
    rec = json.loads(fp.read_text())
    if json.loads(tagged) != rec:
        return {"ok": False, "reason": f"{DIR}/FROZEN.json differs from the tagged version", "changed": [f"{DIR}/FROZEN.json"]}
    changed = [f for f, h in rec["files"].items() if not (root / f).exists() or _sha(root / f) != h]
    if changed:
        return {"ok": False, "reason": "frozen files changed since the freeze", "changed": changed, "record": rec}
    return {"ok": True, "reason": "all frozen hashes match", "changed": [], "record": rec}


def record_deviation(root: Path, reason: str, changed: list[str]) -> dict:
    d = {"timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reason": reason,
         "changed_files": changed}
    p = Path(root) / DIR / "deviations.md"
    if not p.exists():
        p.write_text("# Deviations from the frozen v3 preregistration\n\n")
    with open(p, "a") as f:
        f.write(f"- {d['timestamp']} — {reason} — changed files: {', '.join(changed) or 'none'}\n")
    return d


def enforce(root: Path, deviation: str | None, extra_changes: list[str] | None = None) -> tuple[bool, dict, dict | None]:
    """(may_run, verify_result, deviation_record). extra_changes: run-time changes that are not file edits (e.g. a time
    budget override through the environment); they also need --deviation."""
    v = verify(root)
    extra = list(extra_changes or [])
    if v["ok"] and not extra:
        return True, v, (record_deviation(root, deviation, []) if deviation else None)
    if deviation and (v.get("record") is not None or (v["ok"] and extra)):
        return True, v, record_deviation(root, deviation, v.get("changed", []) + extra)
    if v["ok"] and extra:
        v = {**v, "ok": False, "reason": "run-time change after the freeze without --deviation", "changed": extra,
             "record": v.get("record")}
    return False, v, None
