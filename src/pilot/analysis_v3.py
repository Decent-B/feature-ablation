"""v3 analysis (frozen with the preregistration): Ē per method, dependence tests D1-D6, the headline label and the
secondary analyses S1-S7. Every threshold comes from prereg/v3/hypotheses.yaml.

Per intervention: damage (AUROC drop) and cost (generic-prompt KL) per bootstrap replicate, from v2's
`chunk_samples` (count-weighted, stratified by label, paired across every cell of the same (model, task)).
  E_{b,r}(k) = D_{b,r}(k) - D_{b,random}(k)     (same cell, same basis, same nominal budget k)
  Ē_{b,r}    = mean of E over the ladder budgets present in both curves (primary); S4 uses single budgets and S5 the
               value at a fixed cost (linear in log cost on the monotone envelope, both curves must reach it).
Comparisons: a difference is *reliable* if |point| >= delta, its 95% percentile CI excludes 0 and its BH q-value
(within the test) is <= alpha. A *reversal* is a method pair whose difference is reliably positive at one level
of the factor and reliably negative at another.
  D1 basis  reference cell; pairs of bases with the rule fixed (3 rules x 3 pairs)
  D2 rule   reference cell; pairs of rules with the basis fixed
  D3 task   reversal of a method pair between two tasks (Gemma, reference depth)
  D4 model  reversal between Gemma and Qwen (reference task and depth); also reported for sae-only and pca+dims
  D5 depth  reversal between two depths (Gemma, reference task)
  D6 width  reference cell; sae (65k) vs sae_x (16k) with the rule fixed
Label: SUPPORTED if D1-D4 all hold, PARTIALLY SUPPORTED if 1-3, NOT SUPPORTED if none; INCONCLUSIVE per inclusion.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .analysis_v2 import _chunks, _clean, _samples, chunk_samples, spearman
from .utils import load_yaml, read_json

RULES3 = ("meandiff", "l1probe", "attrib")
BASES = ("sae", "pca", "dims")


def results_table(run_dir: Path) -> pd.DataFrame:
    """One row per intervention with point damage, cost and coherence."""
    tabs = []
    for c in _chunks(run_dir):
        rows = pd.read_parquet(c)
        z = np.load(c.with_suffix(".npz"))
        a = {k: z[k] for k in z.files}
        cl = _clean(c)
        dmg, _ = _samples(a, cl, np.ones((1, len(cl["labels"]))), np.ones((1, a["kl9"].shape[1])))
        rows["damage"] = dmg[:, 0]
        rows["coherence"] = a["aux"].mean(1)
        rows["chunk"] = str(c.relative_to(Path(run_dir) / "ablate" / "chunks").with_suffix(""))
        tabs.append(rows)
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()


def load_hyp3(run_dir: Path) -> dict:
    p = Path(run_dir) / "prereg_snapshot" / "hypotheses.yaml"
    if p.exists():
        return load_yaml(p)
    from .v3common import HYPOTHESES3

    return load_yaml(HYPOTHESES3)


# ============================================================================= data

class Data:
    def __init__(self, run_dir: Path, B: int, seed: int, hyp: dict):
        self.run_dir, self.B, self.hyp = Path(run_dir), B, hyp
        m = hyp["metric"]
        self.delta, self.alpha = float(m["delta"]), float(m["alpha"])
        tabs, Db, Cb, Dp, Cp = [], [], [], [], []
        n = 0
        for c in _chunks(run_dir):
            s = chunk_samples(c, B, seed)
            r = s["rows"].copy()
            r["i"] = np.arange(n, n + len(r))
            n += len(r)
            tabs.append(r)
            Db.append(s["dmg_b"])
            Cb.append(s["cost_b"])
            Dp.append(s["dmg_p"])
            Cp.append(s["cost_p"])
        self.rows = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
        self.Db = np.concatenate(Db) if Db else np.zeros((0, B))
        self.Cb = np.concatenate(Cb) if Cb else np.zeros((0, B))
        self.Dp = np.concatenate(Dp) if Dp else np.zeros(0)
        self.Cp = np.concatenate(Cp) if Cp else np.zeros(0)
        self._cache = {}

    def sel(self, cell: dict, basis: str, rule: str) -> pd.DataFrame:
        r = self.rows
        if r.empty:
            return r
        m = (r.model == cell["model"]) & (r.task == cell["task"]) & (r.layer == cell["layer"]) & \
            (r.strategy == cell["strategy"]) & (r.basis == basis) & (r.rule == rule)
        return r[m]

    def curve(self, cell: dict, basis: str, rule: str):
        """{k: (point damage, damage samples [B], point cost, cost samples [B])}, averaged over resamples / draws."""
        key = (tuple(sorted(cell.items())), basis, rule)
        if key not in self._cache:
            d = self.sel(cell, basis, rule)
            out = {}
            for k, g in d.groupby("k"):
                ii = g.i.to_numpy()
                out[int(k)] = (float(self.Dp[ii].mean()), self.Db[ii].mean(0), float(self.Cp[ii].mean()),
                               self.Cb[ii].mean(0))
            self._cache[key] = out
        return self._cache[key]


# ============================================================================= measures

def measure(dat: Data, cell: dict, basis: str, rule: str, mode="avg"):
    """(point, samples [B]) of the method's damage beyond random, or None if not computable."""
    cr, c0 = dat.curve(cell, basis, rule), dat.curve(cell, basis, "random")
    ks = sorted(set(cr) & set(c0))
    if not ks:
        return None
    if mode == "avg":
        pt = np.mean([cr[k][0] - c0[k][0] for k in ks])
        sm = np.mean([cr[k][1] - c0[k][1] for k in ks], axis=0)
        return float(pt), sm
    if isinstance(mode, (int, np.integer)):
        if int(mode) not in ks:
            return None
        k = int(mode)
        return float(cr[k][0] - c0[k][0]), cr[k][1] - c0[k][1]
    if isinstance(mode, tuple) and mode[0] == "cost":
        c = float(mode[1])

        def at(cv):
            kk = sorted(cv)
            Cp = np.array([[cv[k][2] for k in kk]])
            Dp_ = np.array([[cv[k][0] for k in kk]])
            v, f = stats.value_at_cost(Cp, Dp_, c)
            if f[0] == stats.FLAG_NEVER:
                return None
            Cb = np.stack([cv[k][3] for k in kk], 1)
            Db = np.stack([cv[k][1] for k in kk], 1)
            vb, _ = stats.value_at_cost(Cb, Db, c)
            return float(v[0]), vb
        a, b = at(cr), at(c0)
        if a is None or b is None:
            return None
        return a[0] - b[0], a[1] - b[1]
    raise ValueError(mode)


def methods(dat: Data, cell: dict, bases=BASES) -> dict:
    out = {}
    for b in bases:
        for r in RULES3:
            m = measure(dat, cell, b, r, dat.mode)
            if m is not None:
                out[(b, r)] = m
    return out


# ============================================================================= comparisons

def _compare(dat: Data, items: list[dict]) -> list[dict]:
    """items: {label..., a: (pt, smp), b: (pt, smp)} -> adds point, ci, p, q, reliable, sign."""
    for it in items:
        pt = it["a"][0] - it["b"][0]
        s = it["a"][1] - it["b"][1]
        lo, hi = stats.percentile_ci(s)
        it.update(point=float(pt), ci=[float(lo), float(hi)], p=stats.boot_pvalue(s))
    q = stats.bh_qvalues([it["p"] for it in items]) if items else []
    for it, qq in zip(items, q):
        ci_ex = it["ci"][0] > 0 or it["ci"][1] < 0
        it["q"] = float(qq)
        it["reliable"] = bool(abs(it["point"]) >= dat.delta and ci_ex and qq <= dat.alpha)
        it["sign"] = int(np.sign(it["point"])) if it["reliable"] else 0
        it.pop("a")
        it.pop("b")
    return items


def difference_test(dat: Data, cell: dict, factor: str, bases=BASES) -> dict:
    """D1 (factor basis), D2 (factor rule), D6 (factor width: sae vs sae_x)."""
    ms = methods(dat, cell, bases)
    items = []
    if factor == "basis":
        for r in RULES3:
            for b1, b2 in itertools.combinations(bases, 2):
                if (b1, r) in ms and (b2, r) in ms:
                    items.append({"fixed": r, "x": b1, "y": b2, "a": ms[(b1, r)], "b": ms[(b2, r)]})
    elif factor == "rule":
        for b in bases:
            for r1, r2 in itertools.combinations(RULES3, 2):
                if (b, r1) in ms and (b, r2) in ms:
                    items.append({"fixed": b, "x": r1, "y": r2, "a": ms[(b, r1)], "b": ms[(b, r2)]})
    elif factor == "width":
        for r in RULES3:
            if ("sae", r) in ms and ("sae_x", r) in ms:
                items.append({"fixed": r, "x": "sae (65k)", "y": "sae_x (16k)", "a": ms[("sae", r)], "b": ms[("sae_x", r)]})
    else:
        raise ValueError(factor)
    items = _compare(dat, items)
    rel = [it for it in items if it["reliable"]]
    return {"holds": bool(rel), "n_comparisons": len(items), "n_reliable": len(rel), "comparisons": items,
            "largest": max(items, key=lambda x: abs(x["point"])) if items else None,
            "computable": bool(items)}


def reversal_test(dat: Data, cells: dict, bases=BASES) -> dict:
    """cells: {level: cell}. Reversal of a method pair between two levels (D3-D5)."""
    per = {lv: methods(dat, c, bases) for lv, c in cells.items()}
    items = []
    for lv, ms in per.items():
        for m1, m2 in itertools.combinations(sorted(ms), 2):
            items.append({"level": lv, "x": "/".join(m1), "y": "/".join(m2), "a": ms[m1], "b": ms[m2]})
    items = _compare(dat, items)
    by_pair = {}
    for it in items:
        by_pair.setdefault((it["x"], it["y"]), []).append(it)
    revs = []
    for (x, y), its in by_pair.items():
        pos = [it for it in its if it["sign"] > 0]
        neg = [it for it in its if it["sign"] < 0]
        for a, b in itertools.product(pos, neg):
            revs.append({"x": x, "y": y, "x_better_at": a["level"], "y_better_at": b["level"],
                         "diff_at_x_level": a["point"], "diff_at_y_level": b["point"]})
    # largest change of a method pair's difference across levels (S6)
    big = None
    for (x, y), its in by_pair.items():
        if len(its) >= 2:
            pts = [it["point"] for it in its]
            span = max(pts) - min(pts)
            if big is None or span > big["span"]:
                big = {"x": x, "y": y, "span": float(span)}
    return {"holds": bool(revs), "levels": list(cells), "n_comparisons": len(items),
            "n_reliable": int(sum(it["reliable"] for it in items)), "reversals": revs[:50], "n_reversals": len(revs),
            "largest_span": big, "computable": len(per) >= 2 and len(items) > 0}


# ============================================================================= cells of the plan

def plan_cells(dat: Data, inc: dict, cfg: dict) -> dict:
    """Cells present in the data, keyed by role (reference, depth, task, model, resample)."""
    r = dat.rows
    ref = cfg["reference"]
    M, rt = ref["model"], inc.get("reference_task") or ref["task"]
    if r.empty:
        return {}
    gl = r[(r.model == M) & (r.task == rt) & (r.kind == "grid") & (r.strategy == "mean")]
    ref_layer = int(r[(r.model == M) & (r.task == rt) & (np.isclose(r.depth, float(ref["depth"])))].layer.iloc[0]) \
        if ((r.model == M) & (r.task == rt) & np.isclose(r.depth, float(ref["depth"]))).any() else None
    cell = lambda m, t, L, s="mean": {"model": m, "task": t, "layer": int(L), "strategy": s}  # noqa: E731
    out = {"reference": cell(M, rt, ref_layer) if ref_layer is not None else None}
    out["depth"] = {f"{float(d):.2f}": cell(M, rt, L) for (d, L), _ in gl.groupby(["depth", "layer"])}
    tl = r[(r.model == M) & (r.layer == ref_layer) & (r.kind == "grid") & (r.strategy == "mean")]
    out["task"] = {t: cell(M, t, ref_layer) for t in sorted(tl.task.unique())}
    ml = r[(r.task == rt) & np.isclose(r.depth, float(ref["depth"])) & (r.kind == "grid") & (r.strategy == "mean")]
    out["model"] = {m: cell(m, rt, int(g.layer.iloc[0])) for m, g in ml.groupby("model")}
    out["resample"] = cell(M, rt, ref_layer, "resample") if ((r.strategy == "resample") & (r.task == rt)).any() else None
    return out


def run_tests(dat: Data, cells: dict, mode="avg") -> dict:
    dat.mode = mode
    out = {}
    refc = cells.get("reference")
    out["D1"] = difference_test(dat, refc, "basis") if refc else {"holds": False, "computable": False}
    out["D2"] = difference_test(dat, refc, "rule") if refc else {"holds": False, "computable": False}
    out["D3"] = reversal_test(dat, cells.get("task", {}))
    out["D4"] = reversal_test(dat, cells.get("model", {}))
    out["D4_sae_only"] = reversal_test(dat, cells.get("model", {}), bases=("sae",))
    out["D4_pca_dims"] = reversal_test(dat, cells.get("model", {}), bases=("pca", "dims"))
    out["D5"] = reversal_test(dat, cells.get("depth", {}))
    out["D6"] = difference_test(dat, refc, "width", bases=("sae", "sae_x")) if refc else {"holds": False,
                                                                                           "computable": False}
    return out


def label(tests: dict, inc: dict) -> str:
    if inc.get("inconclusive"):
        return "INCONCLUSIVE"
    n = sum(bool(tests[k]["holds"]) for k in ("D1", "D2", "D3", "D4"))
    return "SUPPORTED" if n == 4 else ("PARTIALLY SUPPORTED" if n else "NOT SUPPORTED")


# ============================================================================= secondary

def _fz(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def readout_vs_causal(dat: Data, refc: dict) -> dict:
    """S1 (and S3 attribution accuracy): Spearman over the readout groups in the reference cell."""
    r = dat.rows
    out = {"per_basis": {}}
    zs, zb, za = [], [], []
    for b in BASES:
        g = r[(r.model == refc["model"]) & (r.task == refc["task"]) & (r.layer == refc["layer"]) & (r.basis == b)
              & (r.rule == "readout_group")] if not r.empty else r
        if len(g) < 3:
            continue
        ii = g.i.to_numpy()
        rho_s = spearman(g.strength.to_numpy(), dat.Db[ii].T)
        rho_p = float(spearman(g.strength.to_numpy(), dat.Dp[ii][None])[0])
        rho_a = float(spearman(g.attr_sum.to_numpy(), dat.Dp[ii][None])[0])
        lo, hi = stats.percentile_ci(rho_s)
        out["per_basis"][b] = {"rho_strength": rho_p, "ci": [float(lo), float(hi)], "rho_attribution": rho_a,
                               "n_groups": int(len(g)), "mean_damage": float(dat.Dp[ii].mean())}
        zs.append(_fz(rho_p))
        zb.append(_fz(rho_s))
        za.append(_fz(rho_a))
    if zs:
        pooled_b = np.tanh(np.nanmean(np.stack(zb), 0))
        lo, hi = stats.percentile_ci(pooled_b)
        out["pooled"] = {"rho": float(np.tanh(np.mean(zs))), "ci": [float(lo), float(hi)],
                         "rho_attribution": float(np.tanh(np.mean(za)))}
    return out


def cross_task(dat: Data, refc: dict) -> list[dict]:
    """S3: damage on task B from units attributed for task A vs for B itself, at equal k."""
    r = dat.rows
    if r.empty:
        return []
    c = r[(r.model == refc["model"]) & (r.layer == refc["layer"]) & r.rule.isin(["cross_task", "own_task"])]
    out = []
    for (t, b, k), g in c.groupby(["task", "basis", "k"]):
        own = g[g.rule == "own_task"]
        if own.empty:
            continue
        io = own.i.to_numpy()
        for src, gg in g[g.rule == "cross_task"].groupby("source_task"):
            ic = gg.i.to_numpy()
            diff = dat.Db[ic].mean(0) - dat.Db[io].mean(0)
            lo, hi = stats.percentile_ci(diff)
            out.append({"target": t, "source": src, "basis": b, "k": int(k), "damage_cross": float(dat.Dp[ic].mean()),
                        "damage_own": float(dat.Dp[io].mean()), "diff_ci": [float(lo), float(hi)],
                        "specific": bool(dat.Dp[ic].mean() < 0.5 * dat.Dp[io].mean())})
    return out


def kendall_mean_vs_resample(dat: Data, cells: dict) -> dict | None:
    a, b = cells.get("reference"), cells.get("resample")
    if not a or not b:
        return None
    dat.mode = "avg"
    ma, mb = methods(dat, a), methods(dat, b)
    keys = sorted(set(ma) & set(mb))
    if len(keys) < 3:
        return None
    pa = np.array([ma[k][0] for k in keys])
    pb = np.array([mb[k][0] for k in keys])
    sa = np.stack([ma[k][1] for k in keys], 1)
    sb = np.stack([mb[k][1] for k in keys], 1)
    tb = stats.kendall_tau(sa, sb)
    lo, hi = stats.percentile_ci(np.atleast_1d(tb))
    return {"tau": float(stats.kendall_tau(pa, pb)), "ci": [float(lo), float(hi)], "methods": ["/".join(k) for k in keys]}


def anchors(dat: Data) -> list[dict]:
    r = dat.rows
    if r.empty:
        return []
    a = r[r.basis == "anchor"]
    out = []
    for (m, t, L, s), g in a.groupby(["model", "task", "layer", "strategy"]):
        mm = g[g.rule == "massmean"]
        rd = g[g.rule == "random_dir"]
        out.append({"model": m, "task": t, "layer": int(L), "strategy": s,
                    "massmean_damage": float(dat.Dp[mm.i].mean()) if len(mm) else None,
                    "massmean_cost": float(dat.Cp[mm.i].mean()) if len(mm) else None,
                    "random_dir_damage": float(dat.Dp[rd.i].mean()) if len(rd) else None,
                    "random_dir_cost": float(dat.Cp[rd.i].mean()) if len(rd) else None})
    return out


def methods_table(dat: Data, cells: dict) -> list[dict]:
    """Ē (point, CI) and mean cost per method per cell (report, not the dev report)."""
    dat.mode = "avg"
    rows = []
    seen = set()
    for role, v in cells.items():
        cs = v.values() if isinstance(v, dict) and "model" not in v else ([v] if v else [])
        for c in cs:
            key = tuple(sorted(c.items()))
            if key in seen:
                continue
            seen.add(key)
            for b in BASES + ("sae_x",):
                for r in RULES3:
                    m = measure(dat, c, b, r, "avg")
                    if m is None:
                        continue
                    lo, hi = stats.percentile_ci(m[1])
                    d = dat.sel(c, b, r)
                    rows.append({**c, "basis": b, "rule": r, "E_bar": m[0], "ci": [float(lo), float(hi)],
                                 "ci_width": float(hi - lo), "mean_cost": float(dat.Cp[d.i].mean()),
                                 "max_cost": float(dat.Cp[d.i].max()),
                                 "broken_budgets": int((dat.Cp[d.i] > float(dat.hyp["metric"]["broken_cost"])).sum())})
    return rows


def analyze(run_dir: Path, B: int | None = None) -> dict:
    run_dir = Path(run_dir)
    man = read_json(run_dir / "manifest.json")
    cfg = man["config"]
    hyp = load_hyp3(run_dir)
    B = B or (int(cfg["analysis"]["bootstrap"]) if cfg["phase"] == "smoke" else int(hyp["metric"]["bootstrap"]))
    dat = Data(run_dir, B, int(cfg["seed"]), hyp)
    inc = read_json(run_dir / "inclusion.json")
    cells = plan_cells(dat, inc, cfg)
    if cfg["phase"] == "dev" or dat.rows.empty:
        return {"phase": cfg["phase"], "B": B, "cells": cells, "methods": methods_table(dat, cells) if cells else [],
                "anchors": anchors(dat)}
    tests = run_tests(dat, cells, "avg")
    res = {"phase": cfg["phase"], "B": B, "delta": dat.delta, "alpha": dat.alpha, "cells": cells, "tests": tests,
           "label": label(tests, inc), "inconclusive": bool(inc.get("inconclusive"))}
    sec = {}
    refc = cells.get("reference")
    if refc:
        sec["S1_readout_vs_causal"] = readout_vs_causal(dat, refc)
        sec["S3_cross_task"] = cross_task(dat, refc)
    if cells.get("resample"):
        rc = {**cells, "reference": cells["resample"]}
        t2 = run_tests(dat, {"reference": cells["resample"]}, "avg")
        sec["S2_strategy"] = {"D1_resample": t2["D1"], "D2_resample": t2["D2"],
                              "kendall_mean_vs_resample": kendall_mean_vs_resample(dat, cells)}
        del rc
    sec["S4_single_budgets"] = {}
    for k in hyp["metric"]["single_budgets"]:
        t = run_tests(dat, cells, int(k))
        sec["S4_single_budgets"][str(k)] = {d: t[d]["holds"] for d in ("D1", "D2", "D3", "D4", "D5", "D6")}
    t5 = run_tests(dat, cells, ("cost", float(hyp["metric"]["cost_matched"])))
    sec["S5_cost_matched"] = {"cost": float(hyp["metric"]["cost_matched"]),
                              "holds": {d: t5[d]["holds"] for d in ("D1", "D2", "D3", "D4", "D5", "D6")},
                              "n_comparisons": {d: t5[d].get("n_comparisons", 0) for d in ("D1", "D2", "D3", "D4", "D5", "D6")}}
    sec["S6_size_per_factor"] = {d: (tests[d].get("largest") if d in ("D1", "D2", "D6") else tests[d].get("largest_span"))
                                 for d in ("D1", "D2", "D3", "D4", "D5", "D6")}
    sec["S7_anchors"] = anchors(dat)
    res["secondary"] = sec
    res["methods"] = methods_table(dat, cells)
    return res


def clean_json(o):
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    return o
