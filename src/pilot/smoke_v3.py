"""v3 smoke checks (plan Section 12): in-run model checks and the post-run acceptance verifier.

In-run (per model, `checks` stage): cached example prefix == full prompt (scores), identity (setting every dims unit
to its own value changes nothing, also with values taken from a separate capture), zero cost for a no-op, the
fast-path equivalence, and a resample intervention on the reference cell.
Post-run (`--v3 --verify-smoke`): pytest, models load with the right attention, SAE fidelity at every layer for both
files and both input scalings, layer-index check, the in-run checks, no NaN / Inf, expected outputs, the report's
label, every block ran, no curve past the ladder, --resume, freeze enforcement on a temporary copy, prefetch,
tuning and the projection against the frozen budget; writes smoke_report.md.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bases as BZ
from . import sites as ST
from .utils import LOG, REPO_ROOT, fmt_secs, read_json, write_json

PREFIX_TOL_BF16 = 1.0   # max |score diff| cached prefix vs full prompt: 2 bf16 steps at |logit| 64-128


@torch.inference_mode()
def model_checks(run, mc) -> dict:
    from .engine_v3 import TaskRunner, forward_query, query_masks, scores_all
    from .v3common import ref

    cfg = run.cfg
    r = ref(cfg)
    depth = float(r["depth"]) if any(abs(float(r["depth"]) - d) < 1e-9 for d in mc.depths()) else mc.depths()[0]
    L = mc.layer(depth)
    tol = float(cfg["identity_tol"])
    out = {"layer": L}
    task = run.ref_task if (mc.dir / "prep" / run.ref_task / f"L{L}.pt").exists() else next(
        (t for t in cfg["tasks"] if (mc.dir / "prep" / t / f"L{L}.pt").exists()), None)
    if task is None:
        return out
    # ---- cached example prefix == full prompt
    for t in [x for x in cfg["tasks"] if (mc.dir / "data" / f"{x}_eval.parquet").exists()]:
        s1, _ = scores_all(mc, t, "eval", use_prefix=True)
        s0, _ = scores_all(mc, t, "eval", use_prefix=False)
        d = np.abs(s1 - s0)
        # bf16: the cached-prefix and full-prompt passes run different kernels, so a logit can land one or two bf16
        # steps apart (0.125-0.5 at |logit| 16-128). Pass: typical example exact (median < tol), worst < 2 steps at 128.
        med, mx = float(np.median(d)), float(d.max())
        out[f"prefix_cache_{t}"] = {"max_abs_score_diff": mx, "median_abs_score_diff": med, "tol_median": tol,
                                    "tol_max": PREFIX_TOL_BF16, "pass": med < tol and mx <= PREFIX_TOL_BF16}
    # ---- identity and zero cost
    rn = TaskRunner(mc, task, depth)
    out["fast_path"] = rn.fast_path_check(float(cfg["fast_path_tol"]))
    out["fast_path"]["pass"] = out["fast_path"]["enabled"]
    basis = BZ.DimsBasis(mc.d_model, mc.run.device)
    idx = torch.arange(mc.d_model, device=mc.run.device)
    W_all = basis.directions(idx)
    mk = lambda which: ST.SiteHook(W_all, basis.reader(idx), rn.sink_thr, "S1", ST.IdentityValues())  # noqa: E731
    s, _ = rn._score_eval(mk)
    kl9, kl1, _ = rn._cost(mk)
    out["identity_score"] = {"max_abs_score_change": float(np.abs(s - rn.clean["score"]).max()),
                             "pass": bool(np.abs(s - rn.clean["score"]).max() <= 1e-6)}
    out["noop_cost"] = {"max_kl9": float(kl9.max()), "max_kl1": float(kl1.max()), "pass": bool(kl9.max() == 0.0)}
    b = rn.eval_b[0]
    cap = ST.MultiCapture(mc.layers, [L])
    try:
        forward_query(mc, rn.pkv, b)
        X = cap.h[L].float()
    finally:
        cap.remove()
    pm, pos = query_masks(b)
    table = torch.zeros(int(b.idx.max()) + 1, b.S + 1, mc.d_model, device=mc.run.device)
    for i, ex in enumerate(b.idx):
        table[int(ex), pos[i][pm[i]]] = X[i][pm[i]]   # real tokens only: pad slots clamp to pos 0 (CUDA write race)
    hook = ST.SiteHook(W_all, basis.reader(idx), rn.sink_thr, "S1", ST.ExplicitValues(table))
    hook.set_batch(torch.as_tensor(b.idx, dtype=torch.long), pm, pos)
    d = float((rn._forward(b, hook, 1, True, fast=False) - rn._forward(b, None, 1, True, fast=False)).abs().max())
    out["identity_cached_activations"] = {"max_abs_logit_change": d, "tol": tol, "pass": d < tol, "edited": hook.n_edit}
    # ---- resample on the reference cell
    prep = mc.prep(task, L)
    if prep.get("reservoir") is not None:
        u = np.arange(min(8, mc.d_model))
        o = rn.run(basis, u, "resample", {ST.PROMPT: prep["means"]["dims"]["prompt"]}, prep["reservoir"])
        ok = all(np.isfinite(v).all() for v in o.values() if isinstance(v, np.ndarray))
        out["resample_runs"] = {"reservoir_tokens": int(prep["reservoir"].shape[0]), "pass": bool(ok)}
    del rn
    LOG.info("%s checks: %s", mc.slug, {k: v.get("pass") for k, v in out.items() if isinstance(v, dict)})
    return out


# ============================================================================= post-run acceptance

def _py(args, timeout=3600, cwd=None, extra_env=None):
    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), **(extra_env or {})}
    p = subprocess.run([sys.executable, "-m", *args], cwd=cwd or REPO_ROOT, env=env, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def _finite(o) -> bool:
    if isinstance(o, dict):
        return all(_finite(v) for v in o.values())
    if isinstance(o, list):
        return all(_finite(v) for v in o)
    if isinstance(o, float):
        return np.isfinite(o)
    return True


def prereg_enforcement_test(work: Path) -> dict:
    """--v3 --confirm refuses without a freeze and after a frozen file is edited, on a temporary git copy."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for d in ("src", "config", "prereg"):
        shutil.copytree(REPO_ROOT / d, work / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "FROZEN.json",
                                                                                 "deviations.md"))
    if not (work / "config" / "v3_sae_input_norm.yaml").exists():
        (work / "config" / "v3_sae_input_norm.yaml").write_text("{}\n")
    env = {"GIT_AUTHOR_NAME": "pilot-smoke", "GIT_AUTHOR_EMAIL": "smoke@localhost", "GIT_COMMITTER_NAME": "pilot-smoke",
           "GIT_COMMITTER_EMAIL": "smoke@localhost"}
    genv = {**os.environ, **env}
    git = lambda *a: subprocess.run(["git", "-C", str(work), *a], capture_output=True, text=True, env=genv)  # noqa: E731
    git("init", "-q")
    (work / ".gitignore").write_text("runs/\n__pycache__/\n")
    from .prereg_v3 import write_render

    write_render(work)
    git("add", "-A")
    git("commit", "-q", "-m", "smoke copy")
    res = {}
    rc, out, _ = _py(["pilot.cli_v3", "--confirm", "--enforce-only"], cwd=work, extra_env=env, timeout=600)
    res["refuses_without_freeze"] = rc != 0
    rc, out, _ = _py(["pilot.cli_v3", "--freeze"], cwd=work, extra_env=env, timeout=600)
    res["freeze_ok"] = rc == 0 and bool(git("tag", "-l", "prereg-v3-frozen").stdout.strip())
    rc, out, _ = _py(["pilot.cli_v3", "--confirm", "--enforce-only"], cwd=work, extra_env=env, timeout=600)
    res["accepts_valid_freeze"] = rc == 0
    rc, out, _ = _py(["pilot.cli_v3", "--confirm", "--enforce-only"], cwd=work,
                     extra_env={**env, "PILOT_V3_CONFIRM_HOURS": "9"}, timeout=600)
    res["budget_override_needs_deviation"] = rc != 0
    with open(work / "config" / "v3.yaml", "a") as f:
        f.write("\n# edited after the freeze (smoke test)\n")
    rc, out, _ = _py(["pilot.cli_v3", "--confirm", "--enforce-only"], cwd=work, extra_env=env, timeout=600)
    res["refuses_after_edit"] = rc != 0 and "config/v3.yaml" in out
    rc, out, _ = _py(["pilot.cli_v3", "--confirm", "--enforce-only", "--deviation", "smoke test edit"], cwd=work,
                     extra_env=env, timeout=600)
    dv = work / "prereg" / "v3" / "deviations.md"
    res["deviation_recorded"] = rc == 0 and dv.exists() and "smoke test edit" in dv.read_text()
    res["pass"] = all(res.values())
    shutil.rmtree(work, ignore_errors=True)
    return res


def verify(run_dir: Path, dev_tiny: bool) -> int:
    from .budget_v3 import project, projection_md
    from .prefetch_v3 import missing
    from .v3common import TUNING3_PATH, load_v3_config, slug

    run_dir = Path(run_dir)
    man = read_json(run_dir / "manifest.json")
    cfg = man["config"]
    res = []

    def add(name, ok, detail=""):
        res.append({"check": name, "pass": None if ok is None else bool(ok), "detail": str(detail)})
        LOG.info("[%s] %s %s", "PASS" if ok else ("SKIP" if ok is None else "FAIL"), name, detail)

    rc, out, _ = _py(["pytest", "-q", "tests"], timeout=3600)
    add("pytest -q passes (v1 + v2 + v3 unit tests)", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
    for m in cfg["models"]:
        mi = man.get(f"model:{m}")
        want = cfg["models"][m].get("attn_implementation", "sdpa")
        add(f"model loads with {want} attention: {m}", mi is not None and mi.get("attn_implementation") == want,
            f"{mi.get('n_layers')} layers, d={mi.get('d_model')}, attn={mi.get('attn_implementation')}, "
            f"final softcap={mi.get('final_logit_softcapping')}" if mi else "not loaded")
    for m in cfg["models"]:
        gp = run_dir / "models" / slug(m) / "gate" / "gate.json"
        g = read_json(gp) if gp.exists() else {}
        sae = g.get("sae", {})
        det, ok, idx_ok = [], bool(sae), True
        for L, v in sae.items():
            for slot in ("main", "extra"):
                f = v["files"].get(slot, {})
                if f.get("info") is None:
                    continue
                bm = f.get("by_mode", {})
                ok &= set(bm) >= set(cfg["input_norm_options"])
                det.append(f"L{L} {slot}: " + ", ".join(f"{k} FVE {x['fve']:.3f}" for k, x in bm.items())
                           + f" -> {f['input_norm']}, loss {100 * f['ce_rel_increase']:+.1f}%")
            idx_ok &= bool(v["files"]["main"].get("layer_index_ok", False)) if v["files"]["main"].get("info") else True
        add(f"SAE fidelity at every layer, both files, both input scalings ({m})", ok, "; ".join(det)[:900])
        add(f"layer-index check picks N at every layer ({m})", idx_ok if sae else False,
            {L: (v["files"]["main"].get("fve"), v["files"]["main"].get("by_mode", {}).get(
                v["files"]["main"].get("input_norm"), {}).get("fve_neighbours")) for L, v in sae.items()})
    normp = Path(cfg["sae_input_norm_file"])
    add("measured input scaling written for the freeze", normp.exists(), str(normp))
    ck = read_json(run_dir / "checks" / "checks.json") if (run_dir / "checks" / "checks.json").exists() else {}
    for m, c in ck.items():
        for name, v in c.items():
            if isinstance(v, dict) and "pass" in v:
                add(f"{name} ({m})", v["pass"], json.dumps({k: x for k, x in v.items() if k != "pass"})[:240])
    bad = []
    rp = run_dir / "ablate" / "results.parquet"
    r = pd.read_parquet(rp) if rp.exists() else pd.DataFrame()
    for c in ("damage", "cost_kl", "cost_kl_v1"):
        if c in r and not np.isfinite(r[c].to_numpy(float)).all():
            bad.append(f"results.parquet:{c}")
    for p in (run_dir / "ablate" / "chunks").rglob("*.npz"):
        z = np.load(p)
        bad += [f"{p.name}:{k}" for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    for j in ("tests.json", "secondary.json"):
        p = run_dir / "analyze" / j
        if p.exists() and not _finite(json.loads(p.read_text(), parse_constant=float)):
            bad.append(j)
    add("no NaN or Inf in saved metrics", not bad, "; ".join(bad[:6]))
    ladder = max(int(k) for k in cfg["cell"]["budgets"])
    add("no curve goes past the budget ladder (no doubling)", r.empty or int(r[r.rule.isin(
        ["meandiff", "l1probe", "attrib", "random"])].k.max()) <= ladder, f"max k {int(r.k.max()) if not r.empty else None}")
    exp = ["manifest.json", "log.txt", "inclusion.json", "ablate/results.parquet", "analyze/tests.json",
           "analyze/secondary.json", "report_v3.md", "ablate/blocks.json"]
    miss = [e for e in exp if not (run_dir / e).exists()]
    add("every expected output exists", not miss, ", ".join(miss))
    rep = (run_dir / "report_v3.md").read_text() if (run_dir / "report_v3.md").exists() else ""
    add("report_v3.md opens with the headline label", "## Headline:" in rep[:400])
    blocks = read_json(run_dir / "ablate" / "blocks.json")["blocks"] if (run_dir / "ablate" / "blocks.json").exists() else {}
    add("every smoke block ran", bool(blocks) and all(v.get("status") == "done" for v in blocks.values()),
        {k: v.get("status") for k, v in blocks.items()})
    args = ["pilot.cli_v3", "--smoke", "--resume", "--out", str(run_dir)] + (["--dev-tiny"] if dev_tiny else [])
    rc, out, dt = _py(args, timeout=900)
    n_stages = len(man.get("stages", {}))
    skipped = out.count("already done, skipping")
    add("--resume finishes quickly without recomputing", rc == 0 and skipped >= n_stages,
        f"{dt:.1f} s, {skipped}/{n_stages} stages skipped, exit {rc}")
    pe = prereg_enforcement_test(run_dir / "prereg_test_repo")
    add("--confirm refuses without a freeze, after an edit, and on a budget override without --deviation", pe["pass"], pe)
    try:
        st = missing(load_v3_config("prefetch", dev_tiny=dev_tiny), dev_tiny)
        detail = f"{st['n_files']} files ({st['bytes_total'] / 1e9:.1f} GB); missing {[e['file'] for e in st['missing']][:5]}"
        if dev_tiny:
            add("prefetch: every dataset present", not [e for e in st["missing"] if e["kind"] == "dataset"], detail)
            add("prefetch: model and SAE weights present", None,
                f"dev-tiny: {len(st['skipped_dev_tiny'])} weight files ({st['bytes_skipped'] / 1e9:.1f} GB) not downloaded here")
        else:
            add("prefetch downloaded every model, SAE file and dataset", not st["missing"], detail)
    except Exception as e:  # noqa: BLE001
        add("prefetch completeness", None if dev_tiny else False, repr(e)[:200])
    add("memory watchdog running", man.get("watchdog", {}).get("running"))
    tun = read_json(TUNING3_PATH) if TUNING3_PATH.exists() else {}
    keys = [("dev_tiny:" if dev_tiny else "") + m for m in cfg["models"]]
    add("batch size auto-tuned under the memory cap", all(k in tun for k in keys),
        {k: {kk: tun.get(k, {}).get(kk) for kk in ("max_batch_tokens", "grad_batch_tokens", "fwd_tokens_per_s")} for k in keys})
    proj = project(dev_tiny)
    write_json(run_dir / "projection.json", proj)
    add("throughput measured and projections written", bool(proj["blocks"]),
        f"dev + confirm {fmt_secs(proj['total_seconds'])} vs budget {fmt_secs(proj['budget_seconds'])}; "
        f"not fitting: {proj['not_fitting'] or 'none'}")
    wall = sum(v.get("seconds", 0) for k, v in man.get("stages", {}).items())
    add("smoke wall time (target < 25 min on the GB10)", True if wall < 1500 else None,
        f"{wall / 60:.1f} min on {man.get('env', {}).get('device', {}).get('name')}")
    write_json(run_dir / "smoke_checks.json", res)
    _smoke_report(run_dir, man, res, proj, dev_tiny, r)
    print("\n".join(projection_md(proj)))
    failed = [x for x in res if x["pass"] is False]
    LOG.info("SMOKE %s: %d passed, %d failed, %d skipped/untested", "PASSED" if not failed else "FAILED",
             sum(x["pass"] is True for x in res), len(failed), sum(x["pass"] is None for x in res))
    return 0 if not failed else 1


def _smoke_report(run_dir: Path, man: dict, res: list, proj: dict, dev_tiny: bool, r: pd.DataFrame):
    from .budget_v3 import projection_md

    L = ["# v3 smoke report" + (" — DEV TINY random models (plumbing only)" if dev_tiny else ""), "",
         f"Run `{run_dir.name}` on {man.get('env', {}).get('device', {}).get('name')}. "
         f"{sum(x['pass'] is True for x in res)} checks passed, {sum(x['pass'] is False for x in res)} failed, "
         f"{sum(x['pass'] is None for x in res)} skipped/untested.", "", "## Checks", "", "| check | result | detail |",
         "|---|---|---|"]
    for x in res:
        d = x["detail"].replace("|", "/")
        L.append(f"| {x['check']} | {'PASS' if x['pass'] else ('SKIP' if x['pass'] is None else '**FAIL**')} | {d[:400]} |")
    L += ["", "## Projected run times (full-run set sizes, measured throughput)", ""] + projection_md(proj)
    if dev_tiny or man.get("env", {}).get("gpu_only_checks_untested"):
        L += ["", "These projections were measured on this machine (CPU and/or tiny models) and do not transfer to the "
                  "GB10. Rerun `bash run_pilot.sh --v3 --smoke` on the GB10 for the numbers to freeze."]
    if not r.empty:
        L += ["", "## Measured seconds per intervention in this smoke run (tiny eval sets)", ""]
        for (m, t), g in r.groupby(["model", "task"]):
            L.append(f"- {m}/{t}: {len(g)} interventions, {g.seconds.mean():.3f} s each")
    (run_dir / "smoke_report.md").write_text("\n".join(L) + "\n")
