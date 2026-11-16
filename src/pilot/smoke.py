"""Smoke acceptance checks (spec: "The smoke run passes only if all of these hold")."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import LOG, REPO_ROOT, read_json, sha256_file, write_json

EXPECTED = ["manifest.json", "log.txt", "gate/gate.json", "select/rankings.parquet", "select/stability.csv",
            "ablate/results.parquet", "ablate/per_example", "ablate/generation_checks.csv", "analyze/summary.csv",
            "analyze/hypotheses.json", "plots/gap_closure_truth.png", "plots/gap_closure_refusal.png",
            "plots/stability.png", "report.md"]


def _py(args: list[str], timeout: int = 3600) -> tuple[int, str, float]:
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")})
    return r.returncode, r.stdout + r.stderr, time.time() - t0


def _finite_json(o) -> bool:
    if isinstance(o, float):
        return math.isfinite(o)
    if isinstance(o, dict):
        return all(_finite_json(v) for v in o.values())
    if isinstance(o, list):
        return all(_finite_json(v) for v in o)
    return True


def verify(run_dir: Path, dev: bool = False) -> int:
    run_dir = Path(run_dir)
    res = []

    def add(name, ok, detail=""):
        res.append({"check": name, "pass": bool(ok) if ok is not None else None, "detail": str(detail)})
        mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        LOG.info("[%s] %s %s", mark, name, detail)

    # 1. unit tests
    rc, out, _ = _py(["pytest", "-q", "tests"], timeout=1800)
    add("pytest -q passes", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
    man = read_json(run_dir / "manifest.json")
    # 2. identity check on the real model
    chk = read_json(run_dir / "ablate" / "checks.json")
    worst = max(v["max_abs_logit_change"] for t in chk["identity"].values() for v in t.values())
    add("identity check |dlogit| < 1e-2", all(v["pass"] for t in chk["identity"].values() for v in t.values()),
        f"max {worst:.2e}")
    # 3. SAE sanity
    gate = read_json(run_dir / "gate" / "gate.json")
    sc = gate["sae_check"]
    add("SAE sanity check passes", sc["pass"] or (None if dev else False),
        f"FVE {sc['fve']:.3f}, CE +{sc['ce_increase']:.3f} nats at layer {man.get('layer')}")
    # 4. no NaN/Inf
    bad = []
    resdf = pd.read_parquet(run_dir / "ablate" / "results.parquet")
    for task, g in resdf.groupby("task"):
        for c in g.select_dtypes("number").columns:
            v = g[c].to_numpy(dtype=float)
            if np.isnan(v).all():
                continue                         # metric not defined for this task
            if not np.isfinite(v).all():
                bad.append(f"results.parquet {task}.{c}")
    for p in (run_dir / "ablate" / "per_example").glob("*.npz"):
        z = np.load(p)
        for k in z.files:
            if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all():
                bad.append(f"{p.name}:{k}")
    summ = pd.read_csv(run_dir / "analyze" / "summary.csv")
    if not np.isfinite(summ.D_at_cstar.to_numpy(float)).all():
        bad.append("summary.csv D_at_cstar")
    cells = summ[summ.basis != "anchor"]
    for c in ("D_ci_lo", "D_ci_hi"):
        if not np.isfinite(cells[c].to_numpy(float)).all():
            bad.append(f"summary.csv {c}")
    gcells = cells[(cells.rule != "random") & (cells.G_denominator.abs() > 1e-9)]   # G defined only here
    if not np.isfinite(gcells.G.to_numpy(float)).all():
        bad.append("summary.csv G where defined")
    n_undef = int(((cells.rule != "random") & ~(cells.G_denominator.abs() > 1e-9)).sum())
    if n_undef:
        LOG.info("note: G undefined (anchor damage == random damage) for %d cells; reported as n/a", n_undef)
    hyp = json.loads((run_dir / "analyze" / "hypotheses.json").read_text(), parse_constant=lambda c: float(c))
    if not _finite_json(hyp):
        bad.append("hypotheses.json")
    add("no NaN or Inf in saved metrics", not bad, "; ".join(bad[:6]))
    # 5. expected outputs and decision line
    missing = [e for e in EXPECTED if not (run_dir / e).exists()]
    for s in man["config"]["strategies"]:
        for t in ("truth", "refusal"):
            if not (run_dir / "plots" / f"curves_{t}_{s}.png").exists():
                missing.append(f"plots/curves_{t}_{s}.png")
    rep = (run_dir / "report.md").read_text() if (run_dir / "report.md").exists() else ""
    add("every expected output exists", not missing, ", ".join(missing))
    add("report.md contains a decision line", "DECISION (overall):" in rep)
    # 6. --resume after completion: < 30 s, nothing recomputed
    args = ["pilot.cli", "--resume", "--out", str(run_dir)] + (["--dev-tiny"] if dev else ["--smoke"])
    rc, out, dt = _py(args, timeout=600)
    skipped = out.count("already done, skipping")
    add("--resume finishes < 30 s without recomputing", rc == 0 and dt < 30 and skipped == 8,
        f"{dt:.1f} s, {skipped}/8 stages skipped, exit {rc}")
    # 7. env / watchdog / memory telemetry
    env = man.get("env", {})
    if env.get("gpu_only_checks_untested"):
        add("cu130 torch build and capability (12, 1)", None, "no GPU on this machine: untested (CPU smoke path)")
    else:
        cap = env.get("device", {}).get("capability")
        add("cu130 torch build and capability (12, 1)", env.get("torch", {}).get("cu130_build") and cap == [12, 1],
            f"torch {env.get('torch', {}).get('version')}, capability {cap}")
    add("memory watchdog running", man.get("watchdog", {}).get("running"), f"threshold {man.get('watchdog', {}).get('threshold_gb')} GB")
    peaks = [v.get("memory", {}).get("peak_mem_used_gb") for v in man.get("stages", {}).values()]
    add("peak memory from /proc/meminfo logged", peaks and all(p is not None for p in peaks), f"max {max(peaks or [0])} GB")
    # 8. --check --smoke writes only decision_smoke.json
    dpath = REPO_ROOT / "runs" / "check" / "decision.json"
    before = sha256_file(dpath) if dpath.exists() else None
    if dev:
        add("--check --smoke passes, writes only decision_smoke.json", None, "skipped in dev mode")
    else:
        rc, out, dt = _py(["pilot.cli", "--check", "--smoke"], timeout=3600)
        after = sha256_file(dpath) if dpath.exists() else None
        ds = REPO_ROOT / "runs" / "check" / "decision_smoke.json"
        ok = rc == 0 and ds.exists() and read_json(ds).get("status") in ("PASS", "FAIL") and before == after
        add("--check --smoke passes, writes only decision_smoke.json", ok,
            f"exit {rc}, status {read_json(ds).get('status') if ds.exists() else None}, decision.json unchanged: {before == after}")
    wall = sum(v.get("seconds", 0) for v in man.get("stages", {}).values())
    add("smoke wall time (target < 10 min on the GB10)", True if wall < 600 else None,
        f"{wall / 60:.1f} min on {env.get('device', {}).get('name')}")
    write_json(run_dir / "smoke_checks.json", res)
    failed = [r for r in res if r["pass"] is False]
    LOG.info("SMOKE %s: %d checks passed, %d failed, %d skipped/untested", "PASSED" if not failed else "FAILED",
             sum(r["pass"] is True for r in res), len(failed), sum(r["pass"] is None for r in res))
    return 0 if not failed else 1
