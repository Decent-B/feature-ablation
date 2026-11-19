"""v2 ablation stage: blocks in the preregistered priority order, cells, curves with extension, controls,
resumable chunk files, and the running time guard.

Cell = (model, task, depth, site, strategy). A grid cell runs its anchors first (massmean + random_dir) to get
c*, then every (basis, rule, resample|draw) curve along the ladder; a curve whose largest cost is still below
c_target = max(c*, 0.5) keeps doubling k (1,000, 2,000, ...) until it reaches c_target or its cap (dims / pca:
all units; sae random: latents firing on >= 1%; sae meandiff / attrib: all ranked latents; l1probe: its 8,192
prefiltered latents). A curve that hits its cap below c_target is flagged.
Chunks: ablate/chunks/<model>/<task>/L<L>/<site>_<strategy>/<group>.{npz,parquet}; a finished chunk is skipped on
--resume. Time guard: before each block the projected block time is added to the elapsed run time; if that would
exceed the budget the run stops, and this block and all later ones are listed as skipped (speed only).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bases as BZ
from . import sites as ST
from .engine_v2 import MCtx, RunCtx, TaskRunner
from .prep_v2 import load_orders
from .utils import LOG, REPO_ROOT, fmt_secs, read_json, write_json
from .v2common import HarnessError, depth_tag, load_hypotheses, models_in

RULES3 = ("meandiff", "l1probe", "attrib")
# per-intervention seconds on the GB10 from the amendment's table (fallback when nothing is measured yet)
SPEC_SECONDS = {("Qwen/Qwen3-1.7B", "truth"): 1.8, ("Qwen/Qwen3-1.7B", "sentiment"): 1.9,
                ("Qwen/Qwen3-1.7B", "answerability"): 3.7, ("Qwen/Qwen3-1.7B", "refusal"): 2.7,
                ("Qwen/Qwen3-1.7B", "toxicity"): 1.9,
                ("Qwen/Qwen3.5-4B", "truth"): 3.8, ("Qwen/Qwen3.5-4B", "sentiment"): 4.1,
                ("Qwen/Qwen3.5-4B", "answerability"): 8.5, ("Qwen/Qwen3.5-4B", "refusal"): 5.2,
                ("Qwen/Qwen3.5-4B", "toxicity"): 4.1}


# ============================================================================= planning

def included_pairs(run: RunCtx) -> list[dict]:
    inc = read_json(run.run_dir / "inclusion.json")
    order = {m: i for i, m in enumerate(models_in(run.cfg))}
    pairs = sorted(inc["pairs"], key=lambda p: (order.get(p["model"], 99), p["role"] != "confirmatory"))
    return pairs


def partner(run: RunCtx, model: str, task: str) -> str | None:
    pairs = {p["task"] for p in included_pairs(run) if p["model"] == model}
    inc = read_json(run.run_dir / "inclusion.json")
    subs = {s["failed"]: s["substitute"] for s in inc["substitutions"] if s["model"] == model}
    for a, b in run.cfg["pairs"]["cross_task"]:
        a, b = subs.get(a, a), subs.get(b, b)
        if task == a and b in pairs:
            return b
        if task == b and a in pairs:
            return a
    return None


def plan_cells(run: RunCtx, block: dict) -> list[dict]:
    cfg = run.cfg
    prim = float(cfg["primary_depth"])
    cells = []
    pairs = included_pairs(run)
    for m in models_in(cfg):
        mds = [float(d) for d in cfg["models"][m]["depths"]]
        if cfg.get("depths_only"):
            mds = [d for d in mds if any(abs(d - float(o)) < 1e-9 for o in cfg["depths_only"])]
        mp = [p for p in pairs if p["model"] == m]
        base = lambda p, d, site, strat, tier, kind: {  # noqa: E731
            "model": m, "task": p["task"], "role": p["role"], "depth": d, "site": site, "strategy": strat,
            "tier": tier, "kind": kind, "block": block["id"]}
        if block["cells"] == "primary":
            cells += [base(p, prim, "S1", "mean", "tier1", "grid") for p in mp if prim in mds]
        elif block["cells"] == "controls":
            cells += [base(p, prim, "S1", "mean", "tier1", "controls") for p in mp if prim in mds]
        elif block["cells"] == "tier2":
            for d in block.get("depths", []):
                if any(abs(d - x) < 1e-9 for x in mds) and abs(d - prim) > 1e-9:
                    cells += [base(p, d, "S1", "mean", "tier2", "grid") for p in mp]
        elif block["cells"] == "refusal_extra":
            for p in mp:
                if p["task"] == "refusal" and prim in mds:
                    cells.append(base(p, prim, "S1", "resample", "tier1", "grid"))
                    cells.append(base(p, prim, "S2", "mean", "tier1", "grid"))
        else:
            raise ValueError(block["cells"])
    return cells


def extension_rate(run: RunCtx) -> float | None:
    """Measured extra curve points per ladder point in this run so far (grid curves only)."""
    root = run.run_dir / "ablate" / "chunks"
    n_ext = n_lad = 0
    for p in root.rglob("*.parquet") if root.exists() else []:
        d = pd.read_parquet(p, columns=["kind", "rule", "ext"])
        d = d[(d.kind == "grid") & (d.rule != "massmean") & (d.rule != "random_dir")]
        n_ext += int(d.ext.sum())
        n_lad += int((~d.ext.astype(bool)).sum())
    return n_ext / n_lad if n_lad else None


def n_interventions(run: RunCtx, cell: dict, n_bases: int = 3, ext_rate: float | None = None) -> float:
    cfg = run.cfg
    t = cfg["tiers"][cell["tier"]]
    lad = len(t["budgets"])
    ext = 1.0 + max(float(cfg["budget"]["extension_factor"]), ext_rate or 0.0)
    if cell["kind"] == "controls":
        c3 = cfg["controls"]["c3"]
        return n_bases * (lad * ext + len(cfg["controls"]["cross_task_k"])
                          + int(c3["n_quintiles"]) * int(c3["groups_per_quintile"]))
    curves = n_bases * (3 * int(t["discovery_resamples"]) + int(t["random_draws"]))
    return curves * lad * ext + 1 + int(t["random_dir"])


def sec_per_intervention(run: RunCtx, model: str, task: str) -> tuple[float, str]:
    meas = _measured(run).get((model, task))
    if meas:
        return meas, "measured in this run"
    from .budget_v2 import projected_seconds

    p = projected_seconds(run.cfg, model, task)
    if p:
        return p, "smoke projection (runs/v2/tuning.json)"
    return SPEC_SECONDS.get((model, task), 5.0), "amendment table"


def _measured(run: RunCtx) -> dict:
    out = {}
    root = run.run_dir / "ablate" / "chunks"
    if not root.exists():
        return out
    acc = {}
    for p in root.rglob("*.parquet"):
        d = pd.read_parquet(p, columns=["model", "task", "seconds"])
        for (m, t), g in d.groupby(["model", "task"]):
            a = acc.setdefault((m, t), [0.0, 0])
            a[0] += float(g.seconds.sum())
            a[1] += len(g)
    for k, (s, n) in acc.items():
        out[k] = s / n + float(run.cfg["budget"]["overhead_s_per_intervention"]) if n else None
    return out


# ============================================================================= execution

class Ablator:
    def __init__(self, run: RunCtx):
        self.run, self.cfg = run, run.cfg
        self.root = run.d("ablate", "chunks")
        self._runners = {}
        self.checks = {"fast_path": {}}

    def runner(self, mc: MCtx, task: str, depth: float) -> TaskRunner:
        key = (mc.model_id, task, depth)
        if key not in self._runners:
            for k in [k for k in self._runners if k[0] != mc.model_id or k[1] != task]:
                del self._runners[k]                      # keep one model/task at a time
            r = TaskRunner(mc, task, depth)
            if r.fast:
                self.checks["fast_path"][f"{mc.slug}/{task}/L{r.L}"] = r.fast_path_check(float(self.cfg["fast_path_tol"]))
            cp = self.root / mc.slug / task / f"L{r.L}" / "clean.npz"
            cp.parent.mkdir(parents=True, exist_ok=True)
            r.save_clean(cp)
            self._runners[key] = r
        return self._runners[key]

    def drop_runners(self):
        self._runners.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cell_dir(self, cell: dict, L: int) -> Path:
        from .v2common import slug

        return self.root / slug(cell["model"]) / cell["task"] / f"L{L}" / f"{cell['site']}_{cell['strategy']}"

    # ---------------------------------------------------------------- chunk io
    @staticmethod
    def _save(stem: Path, rows: list[dict], arrays: list[dict]):
        stem.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for a in arrays for k in a if isinstance(a[k], np.ndarray)})
        np.savez(stem.with_suffix(".npz"), **{k: np.stack([a[k] for a in arrays]) for k in keys})
        pd.DataFrame(rows).to_parquet(stem.with_suffix(".parquet"), index=False)

    @staticmethod
    def _done(stem: Path) -> bool:
        return stem.with_suffix(".npz").exists() and stem.with_suffix(".parquet").exists()

    # ---------------------------------------------------------------- one intervention
    def _one(self, runner, basis, units, cell, means, reservoir, meta: dict, rows, arrays):
        out = runner.run(basis, units, cell["strategy"], cell["site"], means, reservoir)
        row = {"model": cell["model"], "task": cell["task"], "role": cell["role"], "depth": cell["depth"],
               "layer": runner.L, "site": cell["site"], "strategy": cell["strategy"], "tier": cell["tier"],
               "block": cell["block"], "kind": cell["kind"], "cost_kl": float(out["kl9"].mean()),
               "cost_kl_v1": float(out["kl1"].mean()), "rel_change": float(out["rel"].mean()),
               "seconds": out["seconds"], **meta}
        rows.append(row)
        arrays.append({k: v for k, v in out.items() if isinstance(v, np.ndarray)})
        return row

    def _means(self, prep: dict, key: str):
        m = prep["means"][key]
        return {ST.PROMPT: m["prompt"], ST.GEN: m["generated"]}

    def _anchor_basis(self, prep: dict, which: str, j: int):
        U = prep["anchors"][which]
        U = U[None] if U.ndim == 1 else U[j:j + 1]
        basis = BZ.DirectionBasis(U.to(self.run.device))
        dm = prep["dims_mean"]
        means = {ST.PROMPT: (basis.U.cpu() @ dm["prompt"].float()),
                 ST.GEN: None if dm["generated"] is None else basis.U.cpu() @ dm["generated"].float()}
        return basis, means

    def _reservoir(self, prep: dict, strategy: str):
        if strategy != "resample":
            return None
        r = prep.get("reservoir")
        if not r:
            raise HarnessError("resample strategy requested where no reservoir was built")
        return {ST.PROMPT: r["prompt"], ST.GEN: r["generated"]}

    # ---------------------------------------------------------------- grid cell
    def anchors(self, cell: dict) -> float:
        mc = self.run.mctx(cell["model"])
        L = mc.layer(cell["depth"])
        stem = self.cell_dir(cell, L) / "anchor"
        if not self._done(stem):
            runner = self.runner(mc, cell["task"], cell["depth"])
            prep = mc.prep(cell["task"], L)
            res = self._reservoir(prep, cell["strategy"])
            rows, arrays = [], []
            t = self.cfg["tiers"][cell["tier"]]
            todo = [("massmean", 0)] + [("random_dir", j) for j in range(int(t["random_dir"]))]
            for which, j in todo:
                basis, means = self._anchor_basis(prep, which, j)
                self._one(runner, basis, np.array([0]), cell, means, res,
                          {"basis": "anchor", "rule": which, "resample": -1, "draw": j, "k": 1, "k_eff": 1, "ext": False,
                           "cap": 1, "curve_flag": "anchor"}, rows, arrays)
            self._save(stem, rows, arrays)
        d = pd.read_parquet(stem.with_suffix(".parquet"))
        return float(d[d.rule == "massmean"].cost_kl.iloc[0])

    def curve(self, runner, basis, order, cap, ladder, c_target, cell, means, res, meta, rows, arrays,
              max_steps=None):
        ext_cfg = self.cfg["extension"]
        max_steps = ext_cfg.get("max_steps") if max_steps is None else max_steps
        seen, last_cost, ke, k = set(), None, 0, 0
        first = len(rows)
        for k in ladder:
            ke = min(int(k), cap)
            if ke <= 0 or ke in seen:
                continue
            seen.add(ke)
            last_cost = self._one(runner, basis, order[:ke], cell, means, res,
                                  {**meta, "k": int(k), "k_eff": ke, "ext": False, "cap": cap}, rows, arrays)["cost_kl"]
        steps = 0
        k = max(int(x) for x in ladder)
        flag = "ok"
        while last_cost is not None and last_cost < c_target:
            if ke >= cap:
                flag = "cap_hit"
                break
            if max_steps is not None and steps >= int(max_steps):
                flag = "max_steps"
                break
            k = int(k * float(ext_cfg["factor"]))
            ke = min(k, cap)
            steps += 1
            last_cost = self._one(runner, basis, order[:ke], cell, means, res,
                                  {**meta, "k": k, "k_eff": ke, "ext": True, "cap": cap}, rows, arrays)["cost_kl"]
        for r in rows[first:]:
            r["curve_flag"] = flag
            r["c_target"] = c_target
        return flag

    def grid_cell(self, cell: dict):
        mc = self.run.mctx(cell["model"])
        L = mc.layer(cell["depth"])
        cstar = self.anchors(cell)
        c_target = max(cstar, float(self.cfg["extension"]["c_floor"]))
        prep = mc.prep(cell["task"], L)
        orders = load_orders(mc, cell["task"], L)
        t = self.cfg["tiers"][cell["tier"]]
        res = self._reservoir(prep, cell["strategy"])
        for key, basis in mc.bases(L).items():
            stem = self.cell_dir(cell, L) / key
            if self._done(stem):
                continue
            runner = self.runner(mc, cell["task"], cell["depth"])
            means = self._means(prep, key)
            rows, arrays = [], []
            curves = [(r, i, -1) for r in RULES3 for i in range(int(t["discovery_resamples"]))]
            curves += [("random", -1, j) for j in range(int(t["random_draws"]))]
            for rule, rs, dr in curves:
                ok = f"{key}|{rule}|{rs}|{dr}"
                if ok not in orders:
                    continue
                self.curve(runner, basis, orders[ok], int(prep["caps"][ok]), [int(k) for k in t["budgets"]], c_target,
                           cell, means, res, {"basis": key, "rule": rule, "resample": rs, "draw": dr}, rows, arrays)
            self._save(stem, rows, arrays)
            LOG.info("%s/%s L%d %s/%s %s: %d interventions (c* = %.4f, target %.3f)", mc.slug, cell["task"], L,
                     cell["site"], cell["strategy"], key, len(rows), cstar, c_target)

    # ---------------------------------------------------------------- controls
    def controls_cell(self, cell: dict):
        mc = self.run.mctx(cell["model"])
        L = mc.layer(cell["depth"])
        prim = dict(cell, kind="grid")
        cstar = self.anchors(prim) if self._done(self.cell_dir(prim, L) / "anchor") else self.anchors(prim)
        c_target = max(cstar, float(self.cfg["extension"]["c_floor"]))
        prep = mc.prep(cell["task"], L)
        orders = load_orders(mc, cell["task"], L)
        src_task = partner(self.run, cell["model"], cell["task"])
        src_orders = load_orders(mc, src_task, L) if src_task else None
        t1 = self.cfg["tiers"]["tier1"]
        for key, basis in mc.bases(L).items():
            stem = self.cell_dir(cell, L) / f"controls_{key}"
            if self._done(stem):
                continue
            runner = self.runner(mc, cell["task"], cell["depth"])
            means = self._means(prep, key)
            rows, arrays = [], []
            sk = f"{key}|attrib_shuffled|0|-1"
            if sk in orders:
                self.curve(runner, basis, orders[sk], int(prep["caps"][sk]), [int(k) for k in t1["budgets"]], c_target,
                           cell, means, None, {"basis": key, "rule": "attrib_shuffled", "resample": 0, "draw": -1},
                           rows, arrays)
            if src_orders is not None and f"{key}|attrib|0|-1" in src_orders:
                for k in self.cfg["controls"]["cross_task_k"]:
                    u = src_orders[f"{key}|attrib|0|-1"][: int(k)]
                    self._one(runner, basis, u, cell, means, None,
                              {"basis": key, "rule": "cross_task", "source_task": src_task, "resample": 0, "draw": -1,
                               "k": int(k), "k_eff": int(len(u)), "ext": False, "cap": int(k), "curve_flag": "control"},
                              rows, arrays)
            for gi, g in enumerate(prep["groups"].get(key, [])):
                self._one(runner, basis, np.asarray(g["units"]), cell, means, None,
                          {"basis": key, "rule": "c3_group", "resample": 0, "draw": gi, "k": int(len(g["units"])),
                           "k_eff": int(len(g["units"])), "ext": False, "cap": int(len(g["units"])),
                           "curve_flag": "control", "quintile": int(g["quintile"]), "strength": float(g["strength"]),
                           "attr_sum": float(g["attr_sum"])}, rows, arrays)
            self._save(stem, rows, arrays)
            LOG.info("%s/%s L%d controls %s: %d interventions (cross-task source: %s)", mc.slug, cell["task"], L, key,
                     len(rows), src_task)

    def run_cell(self, cell: dict):
        (self.controls_cell if cell["kind"] == "controls" else self.grid_cell)(cell)


# ============================================================================= stage

def _elapsed_before_ablate(run: RunCtx) -> float:
    st = run.manifest().get("stages", {})
    return float(sum(v.get("seconds", 0) for k, v in st.items() if k not in ("ablate", "analyze")))


def _partial_block_seconds(run: RunCtx, blocks: set) -> float:
    """Intervention time already spent in blocks that did not finish (a resumed run keeps its clock)."""
    root = run.run_dir / "ablate" / "chunks"
    tot = 0.0
    for p in root.rglob("*.parquet") if root.exists() and blocks else []:
        d = pd.read_parquet(p, columns=["block", "seconds"])
        tot += float(d[d.block.isin(blocks)].seconds.sum())
    return tot


def _launch_worker(run: RunCtx):
    if not run.cfg["analysis"].get("background_workers"):
        return
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    log = open(run.run_dir / "analyze_worker.log", "a")
    p = subprocess.Popen(["nice", "-n", "10", sys.executable, "-m", "pilot.analysis_v2", "--precompute",
                          str(run.run_dir)], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    pids = run.run_dir / "analyze_worker.pids"
    with open(pids, "a") as f:
        f.write(f"{p.pid}\n")
    LOG.info("background analysis worker started (pid %d): bootstrap caches for finished chunks", p.pid)


def budget_seconds(run: RunCtx) -> tuple[float, str]:
    b = run.cfg["budget"]
    if run.phase != "confirm":
        return float(b["total_hours"]) * 3600, f"total_hours={b['total_hours']}"
    if b.get("confirm_hours") is not None:
        return float(b["confirm_hours"]) * 3600, f"confirm_hours={b['confirm_hours']}"
    from .v2common import V2_RUNS

    dev_s, src = None, None
    for d in sorted((V2_RUNS / "dev").glob("*/manifest.json")) if (V2_RUNS / "dev").exists() else []:
        m = read_json(d)
        if m.get("finished") and bool(m.get("dev_tiny")) == run.dev_tiny:
            dev_s = sum(v.get("seconds", 0) for v in m.get("stages", {}).values())
            src = d.parent.name
    if dev_s is None:
        dev_s, src = float(b["dev_hours_assumed"]) * 3600, "assumed (no finished dev run)"
    return float(b["total_hours"]) * 3600 - dev_s, f"{b['total_hours']} h minus dev run {fmt_secs(dev_s)} ({src})"


def stage_ablate(run: RunCtx) -> dict:
    cfg = run.cfg
    out_dir = run.d("ablate")
    ab = Ablator(run)
    hyp = load_hypotheses() if run.phase == "confirm" else {}
    dropped = set((hyp.get("run_plan") or {}).get("dropped_blocks") or [])
    budget, why = budget_seconds(run)
    blocks_path = out_dir / "blocks.json"
    state = read_json(blocks_path) if blocks_path.exists() else {"blocks": {}}
    t_stage = time.time()
    # time spent in earlier sessions of this stage (a resumed run keeps its clock); this session is t_stage
    prior = sum(v.get("seconds", 0) for v in state["blocks"].values() if v.get("status") == "done")
    prior += _partial_block_seconds(run, {b for b, v in state["blocks"].items() if v.get("status") != "done"})
    stop = False
    for blk in cfg["blocks"]:
        bid = blk["id"]
        rec = state["blocks"].setdefault(bid, {})
        if rec.get("status") == "done":
            continue
        cells = plan_cells(run, blk)
        if bid in dropped:
            rec.update(status="dropped (preregistered run plan)", n_cells=len(cells))
            continue
        if stop:
            rec.update(status="skipped (time budget)", n_cells=len(cells))
            continue
        proj = 0.0
        er = extension_rate(run)
        for c in cells:
            mc = run.mctx(c["model"])
            nb = len(mc.bases(mc.layer(c["depth"])))
            s, src = sec_per_intervention(run, c["model"], c["task"])
            proj += n_interventions(run, c, nb, er) * s
        rec["extension_rate_used"] = max(float(cfg["budget"]["extension_factor"]), er or 0.0)
        elapsed = _elapsed_before_ablate(run) + prior + (time.time() - t_stage)
        rec.update(projected_seconds=proj, n_cells=len(cells), elapsed_at_start=elapsed, budget_seconds=budget,
                   budget_source=why)
        if elapsed + proj > budget:
            LOG.warning("TIME GUARD: block %s needs ~%s but only %s of the %s budget remain: stopping here; this and "
                        "later blocks are skipped (speed only, never results)", bid, fmt_secs(proj),
                        fmt_secs(max(0, budget - elapsed)), fmt_secs(budget))
            rec["status"] = "skipped (time budget)"
            stop = True
            write_json(blocks_path, state)
            continue
        LOG.info("[block %s] %d cells, projected %s (elapsed %s of %s)", bid, len(cells), fmt_secs(proj),
                 fmt_secs(elapsed), fmt_secs(budget))
        tb = time.time()
        for c in cells:
            ab.run_cell(c)
        ab.drop_runners()
        rec.update(status="done", seconds=time.time() - tb)
        write_json(blocks_path, state)
        _launch_worker(run)
    write_json(out_dir / "checks.json", ab.checks)
    write_json(blocks_path, state)
    res = collect_results(run)
    res.to_parquet(out_dir / "results.parquet", index=False)
    return {"interventions": int(len(res)), "blocks": {k: v.get("status") for k, v in state["blocks"].items()}}


def collect_results(run: RunCtx) -> pd.DataFrame:
    from .analysis_v2 import results_table

    return results_table(run.run_dir)


def agreement_check(run: RunCtx) -> dict:
    """Dev run: refusal labels from 32 vs 64 generated tokens on up to 100 clean-refused harmful eval prompts, for the
    clean model and the massmean anchor at both sites (one 64-token generation; its first 32 tokens are the
    32-token greedy reply)."""
    from .engine_v2 import generate, refusal_labels

    cfg = run.cfg
    ac = cfg["refusal_generation"]["agreement_check"]
    out = {}
    for p in [p for p in included_pairs(run) if p["task"] == "refusal"]:
        mc = run.mctx(p["model"])
        depth = float(cfg["primary_depth"])
        L = mc.layer(depth)
        r = TaskRunner(mc, "refusal", depth)
        ev = r.ev[r.ev.label == 1].iloc[: int(ac["n_prompts"])]
        toks = list(ev.input_ids)
        short, long_ = int(cfg["refusal_generation"]["max_new_tokens"]), int(ac["long_tokens"])
        marker = mc.gate["tasks"]["refusal"].get("dev_tiny_marker")
        prep = mc.prep("refusal", L)
        ab = Ablator(run)
        for name, site in (("clean", None), ("massmean_S1", "S1"), ("massmean_S2", "S2")):
            hook = None
            if site:
                basis, means = ab._anchor_basis(prep, "massmean", 0)
                vals = ST.MeanValues(means[ST.PROMPT], means[ST.GEN])
                hook = ST.SiteHook(basis.directions(torch.tensor([0], device=run.device)).float(),
                                   basis.reader(torch.tensor([0], device=run.device)), r.sink_thr, site, vals)
            gens, _ = generate(mc, toks, long_, hook=hook, L=L)
            l32 = refusal_labels(mc, [g[:short] for g in gens], marker)
            l64 = refusal_labels(mc, gens, marker)
            out[f"{mc.slug}/{name}"] = {"n": len(toks), "agreement": float(np.mean(l32 == l64)) if len(toks) else None,
                                        "rate_32": float(l32.mean()) if len(toks) else None,
                                        "rate_64": float(l64.mean()) if len(toks) else None}
            LOG.info("32 vs 64 tokens (%s): agreement %.3f", name, out[f"{mc.slug}/{name}"]["agreement"] or float("nan"))
    write_json(run.run_dir / "ablate" / "agreement_32_64.json", out)
    return out
