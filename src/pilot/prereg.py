"""Preregistration: render PREREGISTRATION.md from hypotheses.yaml, freeze (hash + commit + tag), and enforce.

Frozen files: prereg/hypotheses.yaml, prereg/PREREGISTRATION.md, config/v2.yaml, the analysis code
(src/pilot/analysis_v2.py) and the metrics code it calls (src/pilot/stats.py). The run plan (priority order and
dropped blocks) lives inside hypotheses.yaml, so it is frozen too.

  freeze   refuse on uncommitted changes to tracked files (or untracked frozen files); write prereg/FROZEN.json
           (SHA-256 per frozen file, the commit, a UTC timestamp); commit it; tag `prereg-v2-frozen`.
  verify   FROZEN.json and the tag exist, the tag's FROZEN.json equals the working copy's, every hash matches.
  --confirm refuses unless verify passes; a mismatch needs --deviation "<reason>", which is appended to
           prereg/deviations.md (reason, timestamp, changed files) and listed at the top of the report.
Nothing is ever uploaded anywhere.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

FROZEN = ["prereg/hypotheses.yaml", "prereg/PREREGISTRATION.md", "config/v2.yaml", "src/pilot/analysis_v2.py",
          "src/pilot/stats.py"]
TAG = "prereg-v2-frozen"


def _git(root: Path, *args, check=True) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- render

def render(root: Path) -> str:
    h = yaml.safe_load((root / "prereg" / "hypotheses.yaml").read_text())
    m = h["metric"]
    L = [f"# {h['title']}", "", f"Status: **{h['status']}** · drafted {h['date_drafted']}"
         + (f" · authors: {', '.join(h['authors'])}" if h.get("authors") else ""), "",
         "Generated from `prereg/hypotheses.yaml` by `bash run_pilot.sh --v2 --render-prereg`; do not edit by hand.", "",
         "## 1. Design", "",
         f"- Primary setting: {h['primary_setting']['depth']:.0%} depth, site {h['primary_setting']['site']}, "
         f"{h['primary_setting']['strategy']} ablation.",
         f"- Bases: {', '.join(h['bases'])}. Readout rules: {', '.join(h['readout_rules'])}. Causal rule: {h['causal_rule']}.",
         f"- Metric: {m['name']}. {m['definition']}",
         f"- Damage: single-token tasks — {m['damage']['single_token']}; refusal — {m['damage']['refusal']}.",
         f"- Cost: {m['cost']}.",
         f"- Primary cost c0 = {m['c0']} nats; secondary {', '.join(str(x) for x in m['c_secondary'])}. Smallest effect of "
         f"interest delta = {m['delta']}. 95% percentile CIs from {m['bootstrap']} bootstrap resamples; Benjamini–Hochberg "
         f"within each hypothesis family (alpha = {m['alpha']})"
         + ("; a per-pair test also requires BH q <= alpha." if m.get("require_bh_q") else "."), "",
         "## 2. Pairs", "", "Confirmatory (counted toward C1–C3):", ""]
    L += [f"- {a} / {b}" for a, b in h["pairs"]["confirmatory"]]
    L += ["", "Development (reported, never counted):", ""] + [f"- {a} / {b}" for a, b in h["pairs"]["development"]]
    L += ["", "## 3. Inclusion rules (mechanical, preregistered)", ""]
    ir = h["inclusion_rules"]
    L += [f"- Task gate, single-token: {ir['task_gate']['single_token']}.", f"- Task gate, refusal: {ir['task_gate']['refusal']}.",
          f"- Substitution: {ir['substitution']}", f"- SAE basis: {ir['sae']}", "", "## 4. Hypotheses", ""]
    for hid, v in h["hypotheses"].items():
        L += [f"### {hid} ({v['role']}): {v['claim']}", "", f"- Statistic: {v['statistic']}"]
        if "pair_supported_if" in v:
            L.append(f"- A pair supports it if: {v['pair_supported_if']}")
        if "overall" in v:
            o = v["overall"]
            L.append(f"- Supported overall if: in at least {o['min_fraction_of_pairs']:.0%} of included confirmatory pairs "
                     "(rounded up)" + (" and in at least one pair per model" if o.get("at_least_one_pair_per_model") else "") + ".")
        if "supported_if" in v:
            L.append(f"- Supported if: {v['supported_if']}")
        L += [f"- Rationale: {v['rationale']}", ""]
    L += ["## 5. Central-claim label", ""] + [f"- `{k}`: {v}" for k, v in h["central_claim_label"].items() if k != "note"]
    L += [f"- {h['central_claim_label']['note']}", "", "## 6. Secondary descriptive analyses", ""]
    for k, v in h["secondary"].items():
        if isinstance(v, dict):
            L += [f"- {k}:"] + [f"  - ({kk}) {vv}" for kk, vv in v.items()]
        else:
            L.append(f"- {k}: {v}")
    L += ["", "## 7. H0 and anchors", "",
          f"- Refusal anchor drop >= {h['h0']['refusal_anchor_drop']} in generated replies; random <= "
          f"{h['h0']['random_frac_of_anchor']:.0%} of the anchor's damage at c*. H0 is shown beside every result as a caveat "
          "and never changes a verdict, a label or the decision logic.",
          f"- Gap closure G is computed only where the anchor exceeds random by >= {h['g_min_gap']} at c*.", "",
          "## 8. Run plan (frozen)", "",
          f"- Priority order: {' → '.join(h['run_plan']['priority'])}.",
          f"- Dropped before the run (from the smoke projection): {', '.join(h['run_plan']['dropped_blocks']) or 'none'}.",
          f"- Time guard: {h['run_plan']['time_guard']}.", "", "## 9. Known limitations", ""]
    L += [f"- {x}" for x in h["known_limitations"]]
    L += ["", "## 10. Frozen files", "", "SHA-256 hashes are recorded in `prereg/FROZEN.json` at the commit tagged "
          f"`{TAG}`:", ""] + [f"- `{f}`" for f in FROZEN]
    return "\n".join(L) + "\n"


def write_render(root: Path) -> Path:
    p = root / "prereg" / "PREREGISTRATION.md"
    p.write_text(render(root))
    return p


# ----------------------------------------------------------------------------- freeze / verify

def freeze(root: Path) -> int:
    root = Path(root)
    try:
        _git(root, "rev-parse", "HEAD")
    except RuntimeError as e:
        print(f"[freeze] REFUSED: not a git repository with a commit ({e})")
        return 2
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("[freeze] REFUSED: the git working tree has uncommitted changes:\n" + dirty
              + "\nCommit (or revert) them first.")
        return 2
    tracked = set(_git(root, "ls-files").splitlines())
    missing = [f for f in FROZEN if f not in tracked]
    if missing:
        print(f"[freeze] REFUSED: frozen files are not committed: {missing}")
        return 2
    if (root / "prereg" / "PREREGISTRATION.md").read_text() != render(root):
        print("[freeze] REFUSED: prereg/PREREGISTRATION.md is out of date with hypotheses.yaml. Run "
              "`bash run_pilot.sh --v2 --render-prereg`, review, commit, then freeze.")
        return 2
    if _git(root, "tag", "-l", TAG):
        print(f"[freeze] REFUSED: tag {TAG} already exists (a freeze happened). Changes after a freeze need "
              "`--confirm --deviation \"<reason>\"`; to re-freeze deliberately, delete the tag yourself first.")
        return 2
    untracked = _git(root, "status", "--porcelain").splitlines()
    if untracked:
        print(f"[freeze] note: {len(untracked)} untracked paths are not part of the freeze (e.g. {untracked[:3]})")
    commit = _git(root, "rev-parse", "HEAD")
    rec = {"files": {f: _sha(root / f) for f in FROZEN}, "git_commit": commit,
           "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tag": TAG}
    (root / "prereg" / "FROZEN.json").write_text(json.dumps(rec, indent=2) + "\n")
    _git(root, "add", "prereg/FROZEN.json")
    _git(root, "commit", "-m", f"Freeze v2 preregistration ({rec['utc']})", "--", "prereg/FROZEN.json")
    _git(root, "tag", "-a", TAG, "-m", f"v2 preregistration frozen at {rec['utc']}")
    print(f"[freeze] frozen {len(FROZEN)} files at commit {commit[:12]}; FROZEN.json committed and tagged {TAG}.")
    for f, h in rec["files"].items():
        print(f"           {h[:16]}  {f}")
    print("[freeze] Reminder: uploading prereg/PREREGISTRATION.md to OSF or AsPredicted adds an independent timestamp. "
          "This script never uploads anything.")
    return 0


def verify(root: Path) -> dict:
    """{ok, reason, changed: [files]} for the current working copy."""
    root = Path(root)
    fp = root / "prereg" / "FROZEN.json"
    if not fp.exists():
        return {"ok": False, "reason": "no prereg/FROZEN.json: run `bash run_pilot.sh --v2 --freeze` first", "changed": []}
    try:
        if not _git(root, "tag", "-l", TAG):
            return {"ok": False, "reason": f"git tag {TAG} is missing", "changed": []}
        tagged = _git(root, "show", f"{TAG}:prereg/FROZEN.json")
    except RuntimeError as e:
        return {"ok": False, "reason": f"cannot read the frozen record from tag {TAG}: {e}", "changed": []}
    rec = json.loads(fp.read_text())
    if json.loads(tagged) != rec:
        return {"ok": False, "reason": "prereg/FROZEN.json differs from the tagged version", "changed": ["prereg/FROZEN.json"]}
    changed = [f for f, h in rec["files"].items() if not (root / f).exists() or _sha(root / f) != h]
    if changed:
        return {"ok": False, "reason": "frozen files changed since the freeze", "changed": changed, "record": rec}
    return {"ok": True, "reason": "all frozen hashes match", "changed": [], "record": rec}


def record_deviation(root: Path, reason: str, changed: list[str]) -> dict:
    d = {"timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reason": reason,
         "changed_files": changed}
    p = root / "prereg" / "deviations.md"
    if not p.exists():
        p.write_text("# Deviations from the frozen v2 preregistration\n\n")
    with open(p, "a") as f:
        f.write(f"- {d['timestamp']} — {reason} — changed files: {', '.join(changed) or 'none'}\n")
    return d


def enforce(root: Path, deviation: str | None) -> tuple[bool, dict, dict | None]:
    """Returns (may_run, verify_result, deviation_record)."""
    v = verify(root)
    if v["ok"]:
        dev = record_deviation(root, deviation, []) if deviation else None
        return True, v, dev
    if deviation and v.get("record") is not None:
        return True, v, record_deviation(root, deviation, v["changed"])
    return False, v, None
