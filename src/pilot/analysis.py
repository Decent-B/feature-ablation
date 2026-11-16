"""Curves, D@c*, gap closure G, paired bootstrap CIs, hypotheses H0-H6 and X1, decision.

Every configuration contributes points (cost, damage) over budgets k. A curve is one
(task, strategy, basis, rule, resample|draw); D@c* interpolates its monotone envelope
(linear in log cost) at c* = the massmean anchor's cost for that task and strategy.
Rules are averaged over discovery resamples, the random rule over its draws.
Uncertainty: B paired bootstrap replicates of eval examples (stratified by label) and
generic prompts; the same replicate weights are used for every configuration, so
between-cell differences are paired. Hypothesis families are BH-corrected.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as MET
from . import stats
from .utils import LOG, rng_for

HYP_BASES = ("sae", "pca", "dims")          # the comparable bases (mlp is a different site)
READOUT = ("meandiff", "l1probe")
RULES3 = ("meandiff", "l1probe", "attrib")


@dataclass
class TaskData:
    task: str
    configs: pd.DataFrame
    score: np.ndarray
    aux: np.ndarray
    kl: np.ndarray
    rel: np.ndarray
    clean_score: np.ndarray
    clean_aux: np.ndarray
    labels: np.ndarray
    refused: np.ndarray
    tau: float | None
    mode: str
    gen_refusal: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def load_task(run_dir: Path, task: str, gate: dict) -> TaskData:
    pe = Path(run_dir) / "ablate" / "per_example"
    clean = np.load(pe / f"{task}__clean.npz")
    tabs, S, A, K, R, G = [], [], [], [], [], []
    for p in sorted(pe.glob(f"{task}__*__*.parquet")):
        tabs.append(pd.read_parquet(p))
        z = np.load(p.with_suffix(".npz"))
        S.append(z["score"]), A.append(z["aux"]), K.append(z["kl"]), R.append(z["rel"])
        G.append(z["gen_refusal"] if "gen_refusal" in z.files else None)
    cfgs = pd.concat(tabs, ignore_index=True)
    mode = gate["refusal"]["damage_mode"] if task == "refusal" else "auroc"
    gen = np.concatenate(G) if task == "refusal" and mode == "generation" and all(g is not None for g in G) else None
    return TaskData(task, cfgs, np.concatenate(S), np.concatenate(A), np.concatenate(K), np.concatenate(R),
                    clean["score"], clean["aux"], clean["labels"], clean["refused"],
                    gate["refusal"]["threshold"] if task == "refusal" else None, mode, gen)


def results_table(ctx, td: TaskData) -> pd.DataFrame:
    rows = []
    for i, c in td.configs.iterrows():
        if td.task == "truth":
            m = MET.truth_point(td.score[i], td.clean_score, td.labels, td.aux[i])
        else:
            m = MET.refusal_point(td.score[i], td.clean_score, td.labels, td.refused, td.tau,
                                  None if td.gen_refusal is None else td.gen_refusal[i])
        m.update({"cost_kl": float(td.kl[i].mean()), "cost_rel": float(td.rel[i].mean())})
        rows.append({**{k: c[k] for k in ("task", "strategy", "basis", "rule", "resample", "draw", "k", "k_eff")},
                     "site": "mlp_down_proj_input (different site)" if c["basis"] == "mlp" else "resid_post", **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- bootstrap core

def make_weights(td: TaskData, B: int, seed: int):
    rng = rng_for(seed, td.task, "bootstrap")
    We = stats.bootstrap_counts(rng, len(td.labels), B, strata=td.labels)
    Wg = stats.bootstrap_counts(rng, td.kl.shape[1], B)
    return We, Wg


def config_samples(td: TaskData, We, Wg):
    """damage [C, B'] and cost [C, B'] for B' replicates (B'=1 with unit weights = point estimate)."""
    dmg = MET.damage_boot(td.task, td.score, td.clean_score, td.labels, We, td.refused, td.tau, td.gen_refusal)
    cost = stats.weighted_mean(td.kl, Wg)
    return dmg, cost


def curve_values(td: TaskData, dmg, cost, envelope=True, value=None):
    """Interpolate every curve at its strategy's c*. Returns dict of per-curve arrays and anchors."""
    cf = td.configs
    out = {"curves": [], "anchor": {}, "random_dir": {}}
    for strategy in cf.strategy.unique():
        a = cf[(cf.strategy == strategy) & (cf.rule == "massmean")]
        if a.empty:
            continue
        ai = a.index[0]
        cstar = cost[ai]
        out["anchor"][strategy] = {"D": dmg[ai], "cstar": cstar}
        rd = cf[(cf.strategy == strategy) & (cf.rule == "random_dir")].index
        if len(rd):
            out["random_dir"][strategy] = {"D": dmg[rd].mean(0), "cost": cost[rd].mean(0)}
        sub = cf[(cf.strategy == strategy) & (cf.basis != "anchor")]
        for (basis, rule, res, draw), g in sub.groupby(["basis", "rule", "resample", "draw"]):
            g = g.sort_values("k")
            idx = g.index.to_numpy()
            src = dmg if value is None else value
            v, fl = stats.value_at_cost(cost[idx].T, src[idx].T, cstar, envelope=envelope)
            # budget whose cost is nearest c* (point use only)
            c0 = cost[idx][:, 0]
            near = int(g.k.to_numpy()[np.argmin(np.abs(np.log(np.maximum(c0, 1e-12)) - np.log(max(cstar[0], 1e-12))))])
            out["curves"].append({"strategy": strategy, "basis": basis, "rule": rule, "resample": int(res),
                                  "draw": int(draw), "D": v, "flags": fl, "k_at_cstar": near})
    return out


def cell_table(cv: dict) -> dict:
    """Average curves into cells: (strategy, basis, rule) -> dict(D [B], flag_never, flag_below, k)."""
    cells = {}
    for c in cv["curves"]:
        key = (c["strategy"], c["basis"], c["rule"])
        cells.setdefault(key, []).append(c)
    out = {}
    for key, cs in cells.items():
        D = np.mean([c["D"] for c in cs], axis=0)
        fl = np.stack([c["flags"] for c in cs])
        out[key] = {"D": D, "never_frac": float((fl == stats.FLAG_NEVER).mean()),
                    "below_frac": float((fl == stats.FLAG_BELOW).mean()), "n_curves": len(cs),
                    "k_at_cstar": int(np.median([c["k_at_cstar"] for c in cs]))}
    return out


def gap_closure(cells: dict, anchor: dict) -> dict:
    G = {}
    for (s, b, r), v in cells.items():
        if r == "random" or s not in anchor:
            continue
        rnd = cells.get((s, b, "random"))
        if rnd is None:
            continue
        den = anchor[s]["D"] - rnd["D"]
        with np.errstate(invalid="ignore", divide="ignore"):
            G[(s, b, r)] = np.where(np.abs(den) > 1e-9, (v["D"] - rnd["D"]) / np.where(np.abs(den) > 1e-9, den, 1), np.nan)
    return G


def point_cells(ctx, td: TaskData) -> pd.DataFrame:
    """Point-estimate D@c* per cell (used by the generation checks in the ablate stage)."""
    W1e = np.ones((1, len(td.labels)), np.float32)
    W1g = np.ones((1, td.kl.shape[1]), np.float32)
    dmg, cost = config_samples(td, W1e, W1g)
    cells = cell_table(curve_values(td, dmg, cost))
    return pd.DataFrame([{"strategy": s, "basis": b, "rule": r, "D": float(v["D"][0]), "k_at_cstar": v["k_at_cstar"]}
                         for (s, b, r), v in cells.items()])


# ---------------------------------------------------------------------------- tests

def _ci(x, alpha):
    lo, hi = stats.percentile_ci(x, alpha)
    return [float(lo), float(hi)]


def _test(point: float, samples: np.ndarray, alpha: float, one_sided_positive: bool = False) -> dict:
    lo, hi = _ci(samples, alpha)
    excl = (lo > 0) if one_sided_positive else (lo > 0 or hi < 0)
    return {"stat": float(point), "ci": [lo, hi], "p": stats.boot_pvalue(samples), "ci_excludes_0": bool(excl)}


def _bh(tests: list[dict], alpha: float):
    if not tests:
        return
    q = stats.bh_qvalues(np.array([t["p"] for t in tests]))
    for t, qq in zip(tests, q):
        t["q"] = float(qq)
        t["pass"] = bool(t["ci_excludes_0"] and qq <= alpha and t.get("extra_ok", True))


def analyze_task(ctx, td: TaskData, B: int, seed: int) -> dict:
    alpha = float(ctx.cfg["thresholds"]["alpha"])
    thr = ctx.cfg["thresholds"]
    We, Wg = make_weights(td, B, seed)
    dmg_b, cost_b = config_samples(td, We, Wg)
    dmg_p, cost_p = config_samples(td, np.ones((1, len(td.labels)), np.float32), np.ones((1, td.kl.shape[1]), np.float32))
    cv_b, cv_p = curve_values(td, dmg_b, cost_b), curve_values(td, dmg_p, cost_p)
    cells_b, cells_p = cell_table(cv_b), cell_table(cv_p)
    G_b, G_p = gap_closure(cells_b, cv_b["anchor"]), gap_closure(cells_p, cv_p["anchor"])
    # secondary metrics at c* (no envelope): bias shift and coherence
    bias = td.score - td.clean_score[None, :]
    sec = {}
    for name, val in (("bias_shift", bias.mean(1)[:, None]), ("coherence", td.aux.mean(1)[:, None]),
                      ("cost_rel", td.rel.mean(1)[:, None])):
        cvs = cell_table(curve_values(td, dmg_p, cost_p, envelope=False, value=val))
        sec[name] = {k: float(v["D"][0]) for k, v in cvs.items()}
    strategies = [s for s in ctx.cfg["strategies"] if s in cv_p["anchor"]]
    hyp_strats = [s for s in strategies if s in ("mean", "resample")]
    bases = [b for b in HYP_BASES if any(k[1] == b for k in cells_p)]
    rules = [r for r in RULES3 if any(k[2] == r for k in cells_p)]

    def D(s, b, r, boot=True):
        return (cells_b if boot else cells_p)[(s, b, r)]["D"] if boot else cells_p[(s, b, r)]["D"][0]

    hyp = {}
    # ---------------- H0: harness works (random small vs anchor; refusal anchor generation-confirmed)
    h0 = {"random_checks": [], "anchor_generation": None}
    ok = True
    for s in strategies:
        da = float(cv_p["anchor"][s]["D"][0])
        for b in bases:
            if (s, b, "random") not in cells_p:
                continue
            dr = float(cells_p[(s, b, "random")]["D"][0])
            passed = dr <= thr["h0_random_frac"] * da and da > 0
            h0["random_checks"].append({"strategy": s, "basis": b, "D_random": dr, "D_anchor": da,
                                        "ratio": dr / da if da else float("inf"), "pass": bool(passed)})
            ok &= passed
    if td.task == "refusal":
        gpath = ctx.run_dir / "ablate" / "generation_checks.csv"
        gc = pd.read_csv(gpath) if gpath.exists() and gpath.stat().st_size > 1 else pd.DataFrame()
        rows = gc[gc.label == "anchor:massmean"] if not gc.empty else gc
        drops = {r.strategy: float(r.drop) for r in rows.itertuples()} if not rows.empty else {}
        gen_ok = bool(drops) and all(v >= thr["h0_drop"] for v in drops.values())
        h0["anchor_generation"] = {"drops": drops, "threshold": thr["h0_drop"], "pass": gen_ok}
        ok &= gen_ok
    h0["pass"] = bool(ok)
    hyp["H0"] = h0

    # ---------------- H1: bases differ (largest between-basis difference per rule)
    tests = []
    for s in hyp_strats:
        for r in rules:
            vals = {b: D(s, b, r, False) for b in bases if (s, b, r) in cells_p}
            if len(vals) < 2:
                continue
            ordered = sorted(vals, key=vals.get)            # distinct pair even when values tie
            bmin, bmax = ordered[0], ordered[-1]
            t = _test(vals[bmax] - vals[bmin], D(s, bmax, r) - D(s, bmin, r), alpha)
            t.update({"strategy": s, "rule": r, "pair": [bmax, bmin]})
            tests.append(t)
    _bh(tests, alpha)
    hyp["H1"] = {"tests": tests, "pass": any(t["pass"] for t in tests), "aggregate": "any rule, any strategy"}

    # ---------------- H2: attribution beats the best readout rule, in every basis
    tests = []
    for s in hyp_strats:
        for b in bases:
            if not all((s, b, r) in cells_p for r in RULES3):
                continue
            pt = D(s, b, "attrib", False) - max(D(s, b, "meandiff", False), D(s, b, "l1probe", False))
            sm = D(s, b, "attrib") - np.maximum(D(s, b, "meandiff"), D(s, b, "l1probe"))
            t = _test(pt, sm, alpha, one_sided_positive=True)
            t.update({"strategy": s, "basis": b})
            tests.append(t)
    _bh(tests, alpha)
    h2s = {s: bool([t for t in tests if t["strategy"] == s]) and all(t["pass"] for t in tests if t["strategy"] == s)
           for s in hyp_strats}
    hyp["H2"] = {"tests": tests, "by_strategy": h2s, "pass": bool(h2s) and all(h2s.values()),
                 "aggregate": "all bases, all strategies"}

    # ---------------- H3: selection dominates basis (+ gap-closure thresholds)
    tests = []
    for s in hyp_strats:
        if not all((s, b, r) in cells_p for b in bases for r in RULES3) or len(bases) < 2:
            continue

        def spread(boot):
            get = (lambda b, r: D(s, b, r)) if boot else (lambda b, r: np.array([D(s, b, r, False)]))
            rs = np.mean([np.max([get(b, r) for r in RULES3], 0) - np.min([get(b, r) for r in RULES3], 0)
                          for b in bases], 0)
            bs = np.mean([np.max([get(b, r) for b in bases], 0) - np.min([get(b, r) for b in bases], 0)
                          for r in RULES3], 0)
            return rs - bs
        t = _test(float(spread(False)[0]), spread(True), alpha, one_sided_positive=True)
        g_ok = all(G_p.get((s, b, "attrib"), [np.nan])[0] >= thr["g_attrib"] for b in bases) and all(
            G_p.get((s, b, r), [np.nan])[0] <= thr["g_readout"] for b in bases for r in READOUT)
        t.update({"strategy": s, "extra_ok": bool(g_ok),
                  "G": {f"{b}/{r}": float(G_p[(s, b, r)][0]) for b in bases for r in RULES3 if (s, b, r) in G_p}})
        tests.append(t)
    _bh(tests, alpha)
    hyp["H3"] = {"tests": tests, "pass": bool(tests) and all(t["pass"] for t in tests),
                 "aggregate": "all strategies; spread CI > 0 and G thresholds in every basis"}

    # ---------------- H4: basis x rule interaction (a pair of bases swaps order between two rules)
    tests = []
    for s in hyp_strats:
        for b1, b2 in itertools.combinations(bases, 2):
            for r1, r2 in itertools.combinations(rules, 2):
                if not all((s, b, r) in cells_p for b in (b1, b2) for r in (r1, r2)):
                    continue
                d1p, d2p = D(s, b1, r1, False) - D(s, b2, r1, False), D(s, b1, r2, False) - D(s, b2, r2, False)
                t1 = _test(d1p, D(s, b1, r1) - D(s, b2, r1), alpha)
                t2 = _test(d2p, D(s, b1, r2) - D(s, b2, r2), alpha)
                swap = np.sign(d1p) != np.sign(d2p) and d1p != 0 and d2p != 0
                tests.append({"strategy": s, "bases": [b1, b2], "rules": [r1, r2], "stat": [d1p, d2p],
                              "ci": [t1["ci"], t2["ci"]], "p": max(t1["p"], t2["p"]),
                              "ci_excludes_0": bool(swap and t1["ci_excludes_0"] and t2["ci_excludes_0"]),
                              "swap": bool(swap)})
    _bh(tests, alpha)
    hyp["H4"] = {"tests": tests, "pass": any(t["pass"] for t in tests), "aggregate": "any swap, any strategy"}

    # ---------------- H5: strategy interaction (cells swap order between mean and resample)
    tests = []
    tau = None
    if {"mean", "resample"} <= set(hyp_strats):
        cl = [(b, r) for b in bases for r in RULES3 if ("mean", b, r) in cells_p and ("resample", b, r) in cells_p]
        for c1, c2 in itertools.combinations(cl, 2):
            dm = D("mean", *c1, False) - D("mean", *c2, False)
            dr = D("resample", *c1, False) - D("resample", *c2, False)
            t1 = _test(dm, D("mean", *c1) - D("mean", *c2), alpha)
            t2 = _test(dr, D("resample", *c1) - D("resample", *c2), alpha)
            swap = np.sign(dm) != np.sign(dr) and dm != 0 and dr != 0
            tests.append({"cells": [list(c1), list(c2)], "stat": [dm, dr], "ci": [t1["ci"], t2["ci"]],
                          "p": max(t1["p"], t2["p"]), "swap": bool(swap),
                          "ci_excludes_0": bool(swap and t1["ci_excludes_0"] and t2["ci_excludes_0"])})
        if len(cl) >= 2:
            a_p = np.array([D("mean", *c, False) for c in cl])
            b_p = np.array([D("resample", *c, False) for c in cl])
            a_b = np.stack([D("mean", *c) for c in cl], 1)
            b_b = np.stack([D("resample", *c) for c in cl], 1)
            tau = {"tau": float(stats.kendall_tau(a_p, b_p)), "ci": _ci(stats.kendall_tau(a_b, b_b), alpha)}
    _bh(tests, alpha)
    hyp["H5"] = {"tests": tests, "kendall_tau_mean_vs_resample": tau, "pass": any(t["pass"] for t in tests),
                 "aggregate": "any CI-separated reversal (BH over cell pairs)"}

    # ---------------- X1: bias shift per unit AUROC drop (truth, report only)
    if td.task == "truth":
        x1 = []
        for (s, b, r), v in cells_p.items():
            d = float(v["D"][0])
            bs = sec["bias_shift"].get((s, b, r), float("nan"))
            x1.append({"strategy": s, "basis": b, "rule": r, "auroc_drop": d, "bias_shift": bs,
                       "bias_per_auroc_drop": bs / d if abs(d) > 1e-6 else None})
        hyp["X1"] = {"cells": x1, "note": "exploratory, report only"}
    return {"task": td.task, "hyp": hyp, "cells_b": cells_b, "cells_p": cells_p, "G_b": G_b, "G_p": G_p,
            "anchor_p": cv_p["anchor"], "anchor_b": cv_b["anchor"], "random_dir_p": cv_p["random_dir"],
            "sec": sec, "curves_p": cv_p["curves"], "cost_p": cost_p[:, 0], "dmg_p": dmg_p[:, 0]}


def h6(res: dict, strategies: list[str], alpha: float, thr: float) -> dict:
    out = {"by_strategy": {}}
    for s in strategies:
        cl = [(b, r) for b in HYP_BASES for r in RULES3
              if all((s, b, r) in res[t]["cells_p"] for t in ("truth", "refusal"))]
        if len(cl) < 3:
            continue
        a = np.array([res["truth"]["cells_p"][(s, *c)]["D"][0] for c in cl])
        b = np.array([res["refusal"]["cells_p"][(s, *c)]["D"][0] for c in cl])
        ab = np.stack([res["truth"]["cells_b"][(s, *c)]["D"] for c in cl], 1)
        bb = np.stack([res["refusal"]["cells_b"][(s, *c)]["D"] for c in cl], 1)
        n = min(ab.shape[0], bb.shape[0])
        t = float(stats.kendall_tau(a, b))
        out["by_strategy"][s] = {"tau": t, "ci": _ci(stats.kendall_tau(ab[:n], bb[:n]), alpha),
                                 "n_cells": len(cl), "pass": bool(t >= thr)}
    out["pass"] = bool(out["by_strategy"]) and all(v["pass"] for v in out["by_strategy"].values())
    out["threshold"] = thr
    return out


def decide(h: dict) -> tuple[str, list[str]]:
    held = [k for k in ("H1", "H2", "H3", "H4", "H5") if h[k]["pass"]]
    if not h["H0"]["pass"]:
        return "INVALID HARNESS", held
    if h["H4"]["pass"] or h["H5"]["pass"]:
        return "OUTCOME 1: interaction", held
    if h["H2"]["pass"] and h["H3"]["pass"] and not h["H4"]["pass"]:
        return "OUTCOME 2: selection beats basis", held
    if not held:
        return "OUTCOME 3: uniform", held
    return "MIXED", held


def run_analysis(ctx) -> dict:
    from .pipeline import TASKS

    B = int(ctx.cfg["bootstrap"])
    seed = int(ctx.cfg["seed"])
    res = {}
    for task in TASKS:
        td = load_task(ctx.run_dir, task, ctx.gate)
        res[task] = analyze_task(ctx, td, B, seed)
        res[task]["td"] = td
        LOG.info("%s: %d configurations analysed with %d bootstrap replicates", task, len(td.configs), B)
    thr = ctx.cfg["thresholds"]
    res["H6"] = h6(res, [s for s in ctx.cfg["strategies"] if s in ("mean", "resample")], float(thr["alpha"]),
                   float(thr["h6_tau"]))
    decisions = {}
    for task in TASKS:
        out, held = decide(res[task]["hyp"])
        decisions[task] = {"outcome": out, "held": held}
    outs = {d["outcome"] for d in decisions.values()}
    if any(d["outcome"] == "INVALID HARNESS" for d in decisions.values()):
        overall = "INVALID HARNESS"
    elif len(outs) == 1:
        overall = outs.pop()
    else:
        overall = "MIXED (tasks disagree: " + ", ".join(f"{t}={d['outcome']}" for t, d in decisions.items()) + ")"
    res["decisions"] = decisions
    res["overall"] = overall
    return res
