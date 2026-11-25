"""v3 ablation stage: blocks in the preregistered priority order, one-factor-at-a-time cells, fixed budget ladders
(no doubling), controls, resumable chunk files, and the running time guard.

Cell = (model, task, depth, strategy). A grid cell runs its anchors (massmean + random_dir), then every
(basis, rule, resample|draw) curve along the fixed ladder k = 1, 5, 20, 100, 200, 500 (k capped at the basis size:
dims / pca all units; sae random: latents firing on >= 1%; sae ranked: all ranked latents; l1probe: its 8,192
prefiltered latents). `sae_extra` cells run the Gemma 16k SAE (`sae_x`) curves only (width test D6). Controls:
cross-task (top-k attribution units of every other task at the same depth, ablated on this task) and, in the
reference cell, the readout groups.
Chunks: ablate/chunks/<model>/<task>/L<L>/S1_<strategy>/<group>.{npz,parquet}; a finished chunk is skipped on
--resume. Time guard: before each block the projected block time is added to the elapsed time (dev run included);
if that exceeds the frozen budget the run stops and this and every later block are listed as skipped.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bases as BZ
from . import sites as ST
from .engine_v3 import MCtx, RunCtx, TaskRunner
from .prep_v3 import load_orders
from .utils import LOG, fmt_secs, read_json, write_json
from .v3common import V3_RUNS, HarnessError, load_hypotheses3, plan_cells, slug

RULES3 = ("meandiff", "l1probe", "attrib")
# per-intervention seconds on the GB10 (plan Section 10) when nothing is measured yet
SPEC_SECONDS = {("google/gemma-2-2b", "truth"): 3.6, ("google/gemma-2-2b", "sentiment"): 6.3,
                ("google/gemma-2-2b", "answerability"): 10.6, ("google/gemma-2-2b", "toxicity"): 7.9,
                ("google/gemma-2-2b", "ioi"): 3.0, ("Qwen/Qwen3-1.7B-Base", "truth"): 2.4,
                ("Qwen/Qwen3-1.7B-Base", "sentiment"): 4.2, ("Qwen/Qwen3-1.7B-Base", "answerability"): 7.0,
                ("Qwen/Qwen3-1.7B-Base", "toxicity"): 5.3, ("Qwen/Qwen3-1.7B-Base", "ioi"): 2.0}


def gates_of(run: RunCtx) -> dict:
    from .v3common import models_in

    return {m: read_json(run.run_dir / "models" / slug(m) / "gate" / "gate.json") for m in models_in(run.cfg)}


def n_interventions(run: RunCtx, cell: dict, n_bases: int = 3) -> float:
    c = run.cfg["cell"]
    lad = len(c["budgets"])
    if cell["kind"] == "sae_extra":
        return (3 * int(c["discovery_resamples"]) + int(c["random_draws"])) * lad
    if cell["kind"] == "controls":
        rg = run.cfg["controls"]["readout_groups"]
        n_src = max(1, len(run.cfg["tasks"]) - 1)
        groups = int(rg["n_bands"]) * int(rg["groups_per_band"]) if cell["task"] == run.ref_task else 0
        return n_bases * (n_src * len(run.cfg["controls"]["cross_task_k"]) + groups)
    curves = n_bases * (3 * int(c["discovery_resamples"]) + int(c["random_draws"]))
    return curves * lad + 1 + int(c["random_dir"])


def _measured(run: RunCtx) -> dict:
    out, acc = {}, {}
    root = run.run_dir / "ablate" / "chunks"
    for p in root.rglob("*.parquet") if root.exists() else []:
        d = pd.read_parquet(p, columns=["model", "task", "seconds"])
        for (m, t), g in d.groupby(["model", "task"]):
            a = acc.setdefault((m, t), [0.0, 0])
            a[0] += float(g.seconds.sum())
            a[1] += len(g)
    for k, (s, n) in acc.items():
        out[k] = s / n + float(run.cfg["budget"]["overhead_s_per_intervention"]) if n else None
    return out


def sec_per_intervention(run: RunCtx, model: str, task: str) -> tuple[float, str]:
    meas = _measured(run).get((model, task))
    if meas:
        return meas, "measured in this run"
    from .budget_v3 import projected_seconds

    p = projected_seconds(run.cfg, model, task, run.dev_tiny)
    if p:
        return p, "smoke projection (runs/v3/tuning.json)"
    return SPEC_SECONDS.get((model, task), 5.0), "plan table"


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
                del self._runners[k]
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
        return self.root / slug(cell["model"]) / cell["task"] / f"L{L}" / f"S1_{cell['strategy']}"

    @staticmethod
    def _save(stem: Path, rows: list[dict], arrays: list[dict]):
        stem.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for a in arrays for k in a if isinstance(a[k], np.ndarray)})
        np.savez(stem.with_suffix(".npz"), **{k: np.stack([a[k] for a in arrays]) for k in keys})
        pd.DataFrame(rows).to_parquet(stem.with_suffix(".parquet"), index=False)

    @staticmethod
    def _done(stem: Path) -> bool:
        return stem.with_suffix(".npz").exists() and stem.with_suffix(".parquet").exists()

    def _one(self, runner, basis, units, cell, means, reservoir, meta: dict, rows, arrays):
        out = runner.run(basis, units, cell["strategy"], means, reservoir)
        row = {"model": cell["model"], "task": cell["task"], "depth": cell["depth"], "layer": runner.L, "site": "S1",
               "strategy": cell["strategy"], "block": cell["block"], "kind": cell["kind"], "test": cell.get("test"),
               "cost_kl": float(out["kl9"].mean()), "cost_kl_v1": float(out["kl1"].mean()),
               "rel_change": float(out["rel"].mean()), "seconds": out["seconds"], **meta}
        rows.append(row)
        arrays.append({k: v for k, v in out.items() if isinstance(v, np.ndarray)})
        return row

    @staticmethod
    def _means(prep: dict, key: str):
        return {ST.PROMPT: prep["means"][key]["prompt"]}

    def _anchor_basis(self, prep: dict, which: str, j: int):
        U = prep["anchors"][which]
        U = U[None] if U.ndim == 1 else U[j:j + 1]
        basis = BZ.DirectionBasis(U.to(self.run.device))
        return basis, {ST.PROMPT: basis.U.cpu() @ prep["dims_mean"]["prompt"].float()}

    def _reservoir(self, prep: dict, strategy: str):
        if strategy != "resample":
            return None
        r = prep.get("reservoir")
        if r is None:
            raise HarnessError("resample strategy requested where no reservoir was built")
        return r

    def anchors(self, cell: dict):
        mc = self.run.mctx(cell["model"])
        L = mc.layer(cell["depth"])
        stem = self.cell_dir(cell, L) / "anchor"
        if self._done(stem):
            return
        runner = self.runner(mc, cell["task"], cell["depth"])
        prep = mc.prep(cell["task"], L)
        res = self._reservoir(prep, cell["strategy"])
        rows, arrays = [], []
        todo = [("massmean", 0)] + [("random_dir", j) for j in range(int(self.cfg["cell"]["random_dir"]))]
        for which, j in todo:
            basis, means = self._anchor_basis(prep, which, j)
            self._one(runner, basis, np.array([0]), cell, means, res,
                      {"basis": "anchor", "rule": which, "resample": -1, "draw": j, "k": 1, "k_eff": 1, "cap": 1,
                       "curve_flag": "anchor"}, rows, arrays)
        self._save(stem, rows, arrays)

    def curve(self, runner, basis, order, cap, ladder, cell, means, res, meta, rows, arrays):
        seen = set()
        first = len(rows)
        for k in ladder:
            ke = min(int(k), cap)
            if ke <= 0 or ke in seen:
                continue
            seen.add(ke)
            self._one(runner, basis, order[:ke], cell, means, res, {**meta, "k": int(k), "k_eff": ke, "cap": cap},
                      rows, arrays)
        flag = "cap_hit" if cap < max(int(k) for k in ladder) else "ok"
        for r in rows[first:]:
            r["curve_flag"] = flag

    def grid_cell(self, cell: dict, keys=None):
        mc = self.run.mctx(cell["model"])
        L = mc.layer(cell["depth"])
        if cell["kind"] == "grid":
            self.anchors(cell)
        prep = mc.prep(cell["task"], L)
        orders = load_orders(mc, cell["task"], L)
        c = self.cfg["cell"]
        res = self._reservoir(prep, cell["strategy"])
        for key, basis in mc.bases(L, keys).items():
            stem = self.cell_dir(cell, L) / key
            if self._done(stem):
                continue
            runner = self.runner(mc, cell["task"], cell["depth"])
            means = self._means(prep, key)
            rows, arrays = [], []
            curves = [(r, i, -1) for r in RULES3 for i in range(int(c["discovery_resamples"]))]
            curves += [("random", -1, j) for j in range(int(c["random_draws"]))]
            for rule, rs, dr in curves:
                ok = f"{key}|{rule}|{rs}|{dr}"
                if ok not in orders:
                    continue
                self.curve(runner, basis, orders[ok], int(prep["caps"][ok]), [int(k) for k in c["budgets"]], cell,
                           means, res, {"basis": key, "rule": rule, "resample": rs, "draw": dr}, rows, arrays)
            if rows:
                self._save(stem, rows, arrays)
            LOG.info("%s/%s L%d %s %s: %d interventions", mc.slug, cell["task"], L, cell["strategy"], key, len(rows))

    def controls_cell(self, cell: dict):
        mc = self.run.mctx(cell["model"])
        L = mc.layer(cell["depth"])
        prep = mc.prep(cell["task"], L)
        others = [cc["task"] for cc in plan_cells(self.cfg, {"id": "x", "cells": "controls"}, self.run.ref_task,
                                                  gates_of(self.run)) if cc["task"] != cell["task"]]
        for key, basis in mc.bases(L).items():
            stem = self.cell_dir(cell, L) / f"controls_{key}"
            if self._done(stem):
                continue
            runner = self.runner(mc, cell["task"], cell["depth"])
            means = self._means(prep, key)
            rows, arrays = [], []
            for src in others:
                so = load_orders(mc, src, L)
                ok = f"{key}|attrib|0|-1"
                if ok not in so:
                    continue
                for k in self.cfg["controls"]["cross_task_k"]:
                    u = so[ok][: int(k)]
                    self._one(runner, basis, u, cell, means, None,
                              {"basis": key, "rule": "cross_task", "source_task": src, "resample": 0, "draw": -1,
                               "k": int(k), "k_eff": int(len(u)), "cap": int(k), "curve_flag": "control"}, rows, arrays)
            own = f"{key}|attrib|0|-1"
            so = load_orders(mc, cell["task"], L)
            if own in so:
                for k in self.cfg["controls"]["cross_task_k"]:
                    u = so[own][: int(k)]
                    self._one(runner, basis, u, cell, means, None,
                              {"basis": key, "rule": "own_task", "source_task": cell["task"], "resample": 0, "draw": -1,
                               "k": int(k), "k_eff": int(len(u)), "cap": int(k), "curve_flag": "control"}, rows, arrays)
            for gi, g in enumerate(prep["groups"].get(key, [])):
                self._one(runner, basis, np.asarray(g["units"]), cell, means, None,
                          {"basis": key, "rule": "readout_group", "resample": 0, "draw": gi, "k": int(len(g["units"])),
                           "k_eff": int(len(g["units"])), "cap": int(len(g["units"])), "curve_flag": "control",
                           "band": int(g["band"]), "strength": float(g["strength"]), "attr_sum": float(g["attr_sum"])},
                          rows, arrays)
            if rows:
                self._save(stem, rows, arrays)
            LOG.info("%s/%s L%d controls %s: %d interventions", mc.slug, cell["task"], L, key, len(rows))

    def run_cell(self, cell: dict):
        if cell["kind"] == "controls":
            self.controls_cell(cell)
        elif cell["kind"] == "sae_extra":
            self.grid_cell(cell, keys=["sae_x"])
        else:
            self.grid_cell(cell)


# ============================================================================= stage

def _elapsed_before_ablate(run: RunCtx) -> float:
    st = run.manifest().get("stages", {})
    return float(sum(v.get("seconds", 0) for k, v in st.items() if k not in ("ablate", "analyze")))


def budget_seconds(run: RunCtx) -> tuple[float, str]:
    """Total budget for dev + confirm, minus the latest finished dev run (confirm) or nothing (dev, smoke)."""
    b = run.cfg["budget"]
    total = float(b["total_hours"]) * 3600
    if run.phase != "confirm":
        return total, f"total_hours={b['total_hours']}"
    dev_s, src = None, None
    for d in sorted((V3_RUNS / "dev").glob("*/manifest.json")) if (V3_RUNS / "dev").exists() else []:
        m = read_json(d)
        if m.get("finished") and bool(m.get("dev_tiny")) == run.dev_tiny:
            dev_s = sum(v.get("seconds", 0) for v in m.get("stages", {}).values())
            src = d.parent.name
    if dev_s is None:
        dev_s, src = float(b["dev_hours_assumed"]) * 3600, "assumed (no finished dev run)"
    return total - dev_s, f"{b['total_hours']} h minus dev run {fmt_secs(dev_s)} ({src})"


def stage_ablate(run: RunCtx) -> dict:
    cfg = run.cfg
    out_dir = run.d("ablate")
    ab = Ablator(run)
    gates = gates_of(run)
    hyp = load_hypotheses3() if run.phase == "confirm" else {}
    dropped = set((hyp.get("run_plan") or {}).get("dropped_blocks") or [])
    budget, why = budget_seconds(run)
    blocks_path = out_dir / "blocks.json"
    state = read_json(blocks_path) if blocks_path.exists() else {"blocks": {}}
    t_stage = time.time()
    prior = sum(v.get("seconds", 0) for v in state["blocks"].values() if v.get("status") == "done")
    stop = False
    for blk in cfg["blocks"]:
        bid = blk["id"]
        rec = state["blocks"].setdefault(bid, {})
        if rec.get("status") == "done":
            continue
        cells = plan_cells(cfg, blk, run.ref_task, gates)
        if bid in dropped:
            rec.update(status="dropped (preregistered run plan)", n_cells=len(cells))
            continue
        if stop:
            rec.update(status="skipped (time budget)", n_cells=len(cells))
            continue
        proj = 0.0
        for c in cells:
            mc = run.mctx(c["model"])
            nb = len(mc.bases(mc.layer(c["depth"])))
            s, _ = sec_per_intervention(run, c["model"], c["task"])
            proj += n_interventions(run, c, nb) * s
        elapsed = _elapsed_before_ablate(run) + prior + (time.time() - t_stage)
        rec.update(projected_seconds=proj, n_cells=len(cells), elapsed_at_start=elapsed, budget_seconds=budget,
                   budget_source=why, cells=[{k: c[k] for k in ("model", "task", "depth", "strategy", "kind")} for c in cells])
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
    write_json(out_dir / "checks.json", ab.checks)
    write_json(blocks_path, state)
    from .analysis_v3 import results_table

    res = results_table(run.run_dir)
    res.to_parquet(out_dir / "results.parquet", index=False)
    return {"interventions": int(len(res)), "blocks": {k: v.get("status") for k, v in state["blocks"].items()}}
