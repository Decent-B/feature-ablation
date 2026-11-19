"""v2 analysis (frozen with the preregistration): damage in excess of random at fixed costs, C1-C3, S1-S5,
H0 and anchors. Every threshold comes from prereg/hypotheses.yaml.

Per intervention: damage and cost per bootstrap replicate (count-weighted, paired across every cell of a
(model, task): the same eval-example and generic-prompt weights for all depths, sites, strategies, bases, rules).
  single-token  damage = AUROC(clean score) - AUROC(ablated score)
  refusal       damage = clean generation refusal rate - ablated rate, on the clean-refused harmful eval prompts
  cost          mean over generic prompts of the 9-position KL
Curves: D(c) linear in log cost on the monotone envelope (stats.value_at_cost); averaged over discovery resamples
(rules) or draws (random); E = D_rule - D_random within the same cell and basis.
`python -m pilot.analysis_v2 --precompute RUN_DIR` fills the per-chunk bootstrap caches (background worker).
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .utils import load_yaml, read_json

RULES3 = ("meandiff", "l1probe", "attrib")
READOUT = ("meandiff", "l1probe")
BASES = ("sae", "pca", "dims")


# ============================================================================= per-chunk samples

def _chunks(run_dir: Path) -> list[Path]:
    root = Path(run_dir) / "ablate" / "chunks"
    return sorted(p for p in root.rglob("*.parquet")) if root.exists() else []


def _clean(chunk: Path) -> dict:
    p = chunk.parent.parent / "clean.npz"                 # .../<task>/L<L>/<site_strategy>/<group>.parquet
    z = np.load(p)
    return {k: z[k] for k in z.files}


def _rng(seed: int, *keys) -> np.random.Generator:
    """Deterministic generator from a base seed and keys (same construction as utils.rng_for, kept here so the
    frozen analysis does not depend on unfrozen helpers)."""
    import hashlib

    h = hashlib.sha256(("|".join(str(k) for k in (seed,) + keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _weights(seed: int, model: str, task: str, labels: np.ndarray, n_gen: int, B: int):
    rng = _rng(seed, model, task, "bootstrap-v2")
    return stats.bootstrap_counts(rng, len(labels), B, strata=labels), stats.bootstrap_counts(rng, n_gen, B)


def _samples(arrays: dict, clean: dict, We, Wg) -> tuple[np.ndarray, np.ndarray]:
    labels = clean["labels"]
    if bool(clean["refusal"]):
        hm = labels == 1
        Wr = We[:, hm]
        rc = stats.weighted_mean(clean["gen_label"][hm].astype(float), Wr)
        ra = stats.weighted_mean(arrays["gen_label"][:, hm].astype(float), Wr)
        dmg = rc[None, :] - np.atleast_2d(ra)
    else:
        au_c = stats.auroc_weighted(clean["score"], labels, We)
        dmg = np.stack([au_c - stats.auroc_weighted(s, labels, We) for s in arrays["score"]])
    cost = np.atleast_2d(stats.weighted_mean(arrays["kl9"], Wg))
    return dmg, cost


def chunk_samples(chunk: Path, B: int, seed: int, cache: bool = True) -> dict:
    rows = pd.read_parquet(chunk)
    cdir = Path(str(chunk.with_suffix("")).replace("/ablate/chunks/", "/analyze/cache/"))
    cpath = cdir.with_suffix(f".B{B}.npz")
    src = chunk.with_suffix(".npz")
    if cache and cpath.exists() and cpath.stat().st_mtime >= src.stat().st_mtime:
        z = np.load(cpath)
        return {"rows": rows, **{k: z[k] for k in z.files}}
    z = np.load(src)
    arrays = {k: z[k] for k in z.files}
    clean = _clean(chunk)
    We, Wg = _weights(seed, rows.model.iloc[0], rows.task.iloc[0], clean["labels"], arrays["kl9"].shape[1], B)
    dmg_b, cost_b = _samples(arrays, clean, We, Wg)
    one_e, one_g = np.ones((1, len(clean["labels"]))), np.ones((1, arrays["kl9"].shape[1]))
    dmg_p, cost_p = _samples(arrays, clean, one_e, one_g)
    out = {"dmg_b": dmg_b.astype(np.float32), "cost_b": cost_b.astype(np.float32), "dmg_p": dmg_p[:, 0],
           "cost_p": cost_p[:, 0]}
    if cache:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_name(cpath.name + f".{__import__('os').getpid()}.tmp.npz")
        np.savez(tmp, **out)
        tmp.replace(cpath)
    return {"rows": rows, **out}


def bootstrap_B(run_dir: Path) -> int:
    """Smoke runs use the config's small count; dev and confirm use the preregistered count."""
    man = read_json(Path(run_dir) / "manifest.json")
    if man["config"].get("phase") == "smoke":
        return int(man["config"]["analysis"]["bootstrap"])
    return int(load_hyp(run_dir)["metric"]["bootstrap"])


def precompute(run_dir: Path, B: int | None = None, seed: int | None = None) -> int:
    run_dir = Path(run_dir)
    man = read_json(run_dir / "manifest.json")
    B = B or bootstrap_B(run_dir)
    seed = int(man["config"]["seed"]) if seed is None else seed
    n = 0
    for c in _chunks(run_dir):
        chunk_samples(c, B, seed)
        n += 1
    return n


def results_table(run_dir: Path) -> pd.DataFrame:
    """One row per intervention with point damage, cost (v2 and v1 KL), coherence and refusal secondaries."""
    tabs = []
    for c in _chunks(run_dir):
        rows = pd.read_parquet(c)
        z = np.load(c.with_suffix(".npz"))
        a = {k: z[k] for k in z.files}
        cl = _clean(c)
        dmg, _ = _samples(a, cl, np.ones((1, len(cl["labels"]))), np.ones((1, a["kl9"].shape[1])))
        rows["damage"] = dmg[:, 0]
        if bool(cl["refusal"]):
            hl = cl["labels"] == 0
            rows["refusal_rate"] = a["gen_label"][:, cl["labels"] == 1].mean(1)
            rows["harmless_refusal_rate"] = a["gen_label"][:, hl].mean(1) if hl.any() else np.nan
            rows["first_token_auroc"] = [stats.auroc(s, cl["labels"]) for s in a["score"]]
            rows["coherence"] = np.nan
        else:
            rows["coherence"] = a["aux"].mean(1)
        rows["chunk"] = str(c.relative_to(Path(run_dir) / "ablate" / "chunks").with_suffix(""))
        tabs.append(rows)
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()


# ============================================================================= the analysis object

class Analysis:
    def __init__(self, run_dir: Path, B: int, seed: int, hyp: dict):
        self.run_dir, self.B, self.hyp = Path(run_dir), B, hyp
        m = hyp["metric"]
        self.c0, self.c_sec, self.delta, self.alpha = float(m["c0"]), [float(x) for x in m["c_secondary"]], \
            float(m["delta"]), float(m["alpha"])
        self.req_q = bool(m.get("require_bh_q", True))
        rows, db, cb, dp, cp = [], [], [], [], []
        for c in _chunks(run_dir):
            s = chunk_samples(c, B, seed)
            rows.append(s["rows"])
            db.append(s["dmg_b"]), cb.append(s["cost_b"]), dp.append(s["dmg_p"]), cp.append(s["cost_p"])
        if not rows:
            raise RuntimeError("no ablation results to analyse")
        self.df = pd.concat(rows, ignore_index=True)
        for col, default in (("source_task", ""), ("quintile", -1), ("strength", np.nan), ("attr_sum", np.nan)):
            if col not in self.df:
                self.df[col] = default
        self.dmg_b, self.cost_b = np.concatenate(db), np.concatenate(cb)
        self.dmg_p, self.cost_p = np.concatenate(dp), np.concatenate(cp)
        self._cells = {k: g for k, g in self.df.groupby(["model", "task", "layer", "site", "strategy"])}
        self._cache = {}

    # ---------------------------------------------------------------- helpers
    def cells(self):
        return list(self._cells)

    def cell_rows(self, cell) -> pd.DataFrame:
        return self._cells.get(tuple(cell), self.df.iloc[0:0])

    def anchor(self, cell):
        g = self.cell_rows(cell)
        a = g[(g.basis == "anchor") & (g.rule == "massmean")]
        if a.empty:
            return None
        i = a.index[0]
        rd = g[(g.basis == "anchor") & (g.rule == "random_dir")].index
        return {"D_p": float(self.dmg_p[i]), "D_b": self.dmg_b[i], "c_p": float(self.cost_p[i]), "c_b": self.cost_b[i],
                "rd_D_p": float(self.dmg_p[rd].mean()) if len(rd) else None,
                "rd_c_p": float(self.cost_p[rd].mean()) if len(rd) else None, "n_random_dir": int(len(rd))}

    def bases(self, cell) -> list[str]:
        g = self.cell_rows(cell)
        return [b for b in BASES if ((g.basis == b) & (g.rule == "random")).any()
                and all(((g.basis == b) & (g.rule == r)).any() for r in RULES3)]

    def D(self, cell, basis, rule, c_p, c_b):
        """(point, samples [B], never-reached fraction) of the curve-averaged damage at cost c."""
        cb = np.asarray(c_b, dtype=np.float64)
        key = (tuple(cell), basis, rule, float(c_p), hash(cb.tobytes()))
        if key in self._cache:
            return self._cache[key]
        g = self.cell_rows(cell)
        g = g[(g.basis == basis) & (g.rule == rule)]
        if g.empty:
            return None
        vp, vb, fl = [], [], []
        for _, cur in g.groupby(["resample", "draw"]):
            idx = cur.sort_values("k_eff").index.to_numpy()
            p, f = stats.value_at_cost(self.cost_p[idx][None, :], self.dmg_p[idx][None, :], c_p)
            b, _ = stats.value_at_cost(self.cost_b[idx].T, self.dmg_b[idx].T, c_b)
            vp.append(p[0]), vb.append(b), fl.append(f[0])
        out = (float(np.mean(vp)), np.mean(vb, axis=0), float(np.mean(np.array(fl) == stats.FLAG_NEVER)))
        self._cache[key] = out
        return out

    def E(self, cell, basis, rule, c):
        d, r = self.D(cell, basis, rule, c, c), self.D(cell, basis, "random", c, c)
        if d is None or r is None:
            return None
        return d[0] - r[0], d[1] - r[1], max(d[2], r[2])

    def ci(self, s):
        lo, hi = stats.percentile_ci(s, self.alpha)
        return [float(lo), float(hi)]

    def p_greater(self, s):           # one-sided p for H0: stat <= 0
        s = np.asarray(s)[np.isfinite(s)]
        return float((np.sum(s <= 0) + 1) / (s.size + 1))

    def primary_cell(self, model, task):
        prim = float(self.hyp["primary_setting"]["depth"])
        g = self.df[(self.df.model == model) & (self.df.task == task) & (np.isclose(self.df.depth, prim))
                    & (self.df.site == "S1") & (self.df.strategy == "mean")]
        return None if g.empty else (model, task, int(g.layer.iloc[0]), "S1", "mean")

    # ---------------------------------------------------------------- Delta (C1 statistic)
    def delta_b(self, cell, c=None):
        c = self.c0 if c is None else c
        out = {}
        for b in self.bases(cell):
            ea, em, el = self.E(cell, b, "attrib", c), self.E(cell, b, "meandiff", c), self.E(cell, b, "l1probe", c)
            pt = ea[0] - max(em[0], el[0])
            sm = ea[1] - np.maximum(em[1], el[1])
            out[b] = {"stat": pt, "samples": sm, "ci": self.ci(sm), "never_frac": max(ea[2], em[2], el[2])}
        return out

    def c1_pair(self, cell) -> dict:
        db = self.delta_b(cell)
        if not db:
            return {"stat": None, "reason": "no complete basis x rule grid"}
        stat = float(np.mean([v["stat"] for v in db.values()]))
        sm = np.mean([v["samples"] for v in db.values()], axis=0)
        ci = self.ci(sm)
        neg = [b for b, v in db.items() if v["ci"][1] < 0]
        return {"stat": stat, "ci": ci, "p": self.p_greater(sm), "bases": list(db), "basis_below_zero": neg,
                "delta_b": {b: {"stat": v["stat"], "ci": v["ci"], "never_reached_frac": v["never_frac"]} for b, v in db.items()},
                "_samples": sm}

    def c2_pair(self, cell) -> list[dict]:
        tests = []
        bs = self.bases(cell)
        for b1, b2 in itertools.combinations(bs, 2):
            ea1, ea2 = self.E(cell, b1, "attrib", self.c0), self.E(cell, b2, "attrib", self.c0)
            da_p, da_s = ea1[0] - ea2[0], ea1[1] - ea2[1]
            for r in READOUT:
                er1, er2 = self.E(cell, b1, r, self.c0), self.E(cell, b2, r, self.c0)
                dr_p, dr_s = er1[0] - er2[0], er1[1] - er2[1]
                cr, ca = self.ci(dr_s), self.ci(da_s)
                sep = (cr[0] > 0 and ca[1] < 0) or (cr[1] < 0 and ca[0] > 0)
                p = max(stats.boot_pvalue(dr_s), stats.boot_pvalue(da_s))
                tests.append({"bases": [b1, b2], "readout_rule": r, "d_readout": dr_p, "d_readout_ci": cr,
                              "d_attrib": da_p, "d_attrib_ci": ca, "reversal_ci_separated": bool(sep), "p": p})
        return tests


# ============================================================================= C3 / S4 helpers

def _rankavg(a: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis (ties averaged)."""
    a = np.atleast_2d(a)
    out = np.empty_like(a, dtype=float)
    for i, row in enumerate(a):
        o = np.argsort(row, kind="mergesort")
        r = np.empty(len(row))
        r[o] = np.arange(len(row), dtype=float)
        _, inv, cnt = np.unique(row, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        out[i] = sums[inv] / cnt[inv]
    return out


def spearman(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """x [n] fixed, Y [B, n] -> rho [B] (nan where Y is constant)."""
    rx = _rankavg(x)[0]
    rx = rx - rx.mean()
    rY = _rankavg(Y)
    rY = rY - rY.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum() * (rY ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rY * rx).sum(1) / np.where(den > 0, den, 1), np.nan)


def _fz(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


# ============================================================================= hypotheses

def confirmatory(an: Analysis, pairs: list[dict]) -> dict:
    hyp = an.hyp
    res = {"c0": an.c0, "delta": an.delta, "alpha": an.alpha, "require_bh_q": an.req_q, "bootstrap": an.B,
           "pairs": [], "excluded_pairs": []}
    conf = [p for p in pairs if p["role"] == "confirmatory"]
    # ---------------- C1
    c1 = []
    for p in conf:
        cell = an.primary_cell(p["model"], p["task"])
        r = an.c1_pair(cell) if cell else {"stat": None, "reason": "primary cell not run"}
        r.update(model=p["model"], task=p["task"], note=p.get("note"))
        if r["stat"] is None:
            res["excluded_pairs"].append({"model": p["model"], "task": p["task"], "reason": r["reason"]})
            continue
        res["pairs"].append({"model": p["model"], "task": p["task"], "note": p.get("note"), "bases": r["bases"]})
        c1.append(r)
    q = stats.bh_qvalues([r["p"] for r in c1]) if c1 else []
    for r, qq in zip(c1, q):
        r["q"] = float(qq)
        r["supported"] = bool(r["ci"][0] > 0 and r["stat"] >= an.delta and not r["basis_below_zero"]
                              and (qq <= an.alpha or not an.req_q))
        r.pop("_samples", None)
    n = len(c1)
    need = math.ceil(float(hyp["hypotheses"]["C1"]["overall"]["min_fraction_of_pairs"]) * n) if n else 0
    n_sup = sum(r["supported"] for r in c1)
    models = sorted({p["model"] for p in conf})
    per_model = {m: sum(r["supported"] for r in c1 if r["model"] == m) for m in models}
    per_model_ok = all(v >= 1 for v in per_model.values()) if hyp["hypotheses"]["C1"]["overall"].get(
        "at_least_one_pair_per_model", True) else True
    res["C1"] = {"per_pair": c1, "n_pairs": n, "n_supported": n_sup, "required": need, "per_model_supported": per_model,
                 "supported": bool(n > 0 and n_sup >= need and per_model_ok)}
    # ---------------- C2
    allt = []
    per = []
    for p in conf:
        cell = an.primary_cell(p["model"], p["task"])
        if not cell or not an.bases(cell):
            continue
        tests = an.c2_pair(cell)
        for t in tests:
            t.update(model=p["model"], task=p["task"])
        allt += tests
        per.append({"model": p["model"], "task": p["task"], "tests": tests, "n_bases": len(an.bases(cell))})
    q = stats.bh_qvalues([t["p"] for t in allt]) if allt else []
    for t, qq in zip(allt, q):
        t["q"] = float(qq)
        t["pass"] = bool(t["reversal_ci_separated"] and (qq <= an.alpha or not an.req_q))
    for pp in per:
        pp["reversal"] = any(t["pass"] for t in pp["tests"])
    frac = (sum(pp["reversal"] for pp in per) / len(per)) if per else 0.0
    res["C2"] = {"per_pair": per, "fraction_with_reversal": frac,
                 "required_fraction": float(hyp["hypotheses"]["C2"]["overall"]["min_fraction_of_pairs"]),
                 "supported": bool(per and frac >= float(hyp["hypotheses"]["C2"]["overall"]["min_fraction_of_pairs"]))}
    # ---------------- C3
    res["C3"] = c3(an, conf)
    s1, s2 = res["C1"]["supported"], res["C2"]["supported"]
    res["label"] = "SUPPORTED" if (s1 and s2) else ("PARTIALLY SUPPORTED" if (s1 or s2) else "NOT SUPPORTED")
    return res


def _groups(an: Analysis, cell, basis):
    g = an.cell_rows((cell[0], cell[1], cell[2], "S1", "mean"))
    g = g[(g.basis == basis) & (g.rule == "c3_group")]
    return g


def c3(an: Analysis, pairs: list[dict], what: str = "strength") -> dict:
    out = {"per_pair_basis": [], "pooled_rho": None, "ci": None, "supported": False}
    zs_p, zs_b = [], []
    for p in pairs:
        cell = an.primary_cell(p["model"], p["task"])
        if not cell:
            continue
        for b in an.bases(cell):
            g = _groups(an, cell, b)
            if len(g) < 3:
                continue
            idx = g.index.to_numpy()
            x = g[what].to_numpy(float) if what == "strength" else -g["attr_sum"].to_numpy(float)
            rp = spearman(x, an.dmg_p[idx][None, :])[0]
            rb = spearman(x, an.dmg_b[idx].T)
            rec = {"model": p["model"], "task": p["task"], "basis": b, "n_groups": int(len(idx)),
                   "rho": None if not np.isfinite(rp) else float(rp)}
            out["per_pair_basis"].append(rec)
            if np.isfinite(rp):
                zs_p.append(_fz(rp))
                zs_b.append(_fz(np.where(np.isfinite(rb), rb, rp)))
    if zs_p:
        rho = float(np.tanh(np.mean(zs_p)))
        rb = np.tanh(np.mean(zs_b, axis=0))
        ci = an.ci(rb)
        c3h = an.hyp["hypotheses"]["C3"]
        out.update(pooled_rho=rho, ci=ci, n_components=len(zs_p))
        if what == "strength":
            out["supported"] = bool(rho <= 0.3 and ci[1] < 0.5)
            out["rule"] = c3h["supported_if"]
    return out


def h0_cell(an: Analysis, cell) -> dict:
    a = an.anchor(cell)
    h = an.hyp["h0"]
    if a is None:
        return {"status": "no anchor"}
    task = cell[1]
    rec = {"anchor_D": a["D_p"], "cstar": a["c_p"], "random_dir_D": a["rd_D_p"], "random_dir_cost": a["rd_c_p"],
           "random_checks": [], "anchor_check": None}
    ok = True
    for b in an.bases(cell):
        dr = an.D(cell, b, "random", a["c_p"], a["c_b"])
        passed = a["D_p"] > 0 and dr[0] <= float(h["random_frac_of_anchor"]) * a["D_p"]
        rec["random_checks"].append({"basis": b, "D_random_at_cstar": dr[0],
                                     "ratio": dr[0] / a["D_p"] if a["D_p"] else None, "pass": bool(passed)})
        ok &= passed
    if task == "refusal":
        passed = a["D_p"] >= float(h["refusal_anchor_drop"])
        rec["anchor_check"] = {"generation_drop": a["D_p"], "threshold": float(h["refusal_anchor_drop"]), "pass": bool(passed)}
        ok &= passed
    else:
        rec["anchor_note"] = "single-token task: the anchor's AUROC drop is descriptive"
    fails = [f"random/anchor {c['ratio']:.2f} ({c['basis']})" if c["ratio"] is not None else f"anchor D <= 0 ({c['basis']})"
             for c in rec["random_checks"] if not c["pass"]]
    if rec["anchor_check"] and not rec["anchor_check"]["pass"]:
        fails.append(f"anchor drop {a['D_p']:.2f} < {h['refusal_anchor_drop']}")
    rec["status"] = "pass" if ok else "fail: " + "; ".join(fails)
    # G only where the anchor exceeds random by >= g_min_gap at c*
    G = {}
    for b in an.bases(cell):
        dr = an.D(cell, b, "random", a["c_p"], a["c_b"])
        gap = a["D_p"] - dr[0]
        for r in RULES3:
            d = an.D(cell, b, r, a["c_p"], a["c_b"])
            G[f"{b}/{r}"] = {"D_at_cstar": d[0], "never_reached_frac": d[2],
                             "G": (d[0] - dr[0]) / gap if gap >= float(an.hyp["g_min_gap"]) else "not computed"}
    rec["G"] = G
    return rec


def secondary(an: Analysis, pairs: list[dict]) -> dict:
    out = {"S1_depth": [], "S2_site": [], "S3_strategy": [], "S4": {"a_shuffled": [], "b_cross_task": [], "c_attribution": None},
           "S5_anchors_h0": []}
    for p in pairs:
        m, t = p["model"], p["task"]
        for cell in [c for c in an.cells() if c[0] == m and c[1] == t and c[3] == "S1" and c[4] == "mean"]:
            db = an.delta_b(cell)
            if db:
                sm = np.mean([v["samples"] for v in db.values()], axis=0)
                out["S1_depth"].append({"model": m, "task": t, "role": p["role"], "layer": cell[2],
                                        "depth": float(an.cell_rows(cell).depth.iloc[0]),
                                        "delta_mean": float(np.mean([v["stat"] for v in db.values()])), "ci": an.ci(sm),
                                        "delta_b": {b: {"stat": v["stat"], "ci": v["ci"]} for b, v in db.items()}})
        if t == "refusal":
            prim = an.primary_cell(m, t)
            if prim:
                s2 = (m, t, prim[2], "S2", "mean")
                r1, r2 = an.c1_pair(prim), an.c1_pair(s2) if an.bases(s2) else {"stat": None}
                if r1.get("stat") is not None and r2.get("stat") is not None:
                    out["S2_site"].append({"model": m, "role": p["role"], "delta_S1": r1["stat"], "ci_S1": r1["ci"],
                                           "delta_S2": r2["stat"], "ci_S2": r2["ci"],
                                           "diff_S2_minus_S1": r2["stat"] - r1["stat"],
                                           "diff_ci": an.ci(r2["_samples"] - r1["_samples"])})
                rs = (m, t, prim[2], "S1", "resample")
                cl = [(b, r) for b in an.bases(prim) if b in an.bases(rs) for r in RULES3]
                if len(cl) >= 3:
                    a_p = np.array([an.E(prim, b, r, an.c0)[0] for b, r in cl])
                    b_p = np.array([an.E(rs, b, r, an.c0)[0] for b, r in cl])
                    a_b = np.stack([an.E(prim, b, r, an.c0)[1] for b, r in cl], 1)
                    b_b = np.stack([an.E(rs, b, r, an.c0)[1] for b, r in cl], 1)
                    out["S3_strategy"].append({"model": m, "role": p["role"], "n_cells": len(cl),
                                               "kendall_tau": float(stats.kendall_tau(a_p, b_p)),
                                               "ci": an.ci(stats.kendall_tau(a_b, b_b))})
        prim = an.primary_cell(m, t)
        if prim:
            for b in an.bases(prim):
                e = an.E(prim, b, "attrib_shuffled", an.c0)
                if e is not None:
                    out["S4"]["a_shuffled"].append({"model": m, "task": t, "role": p["role"], "basis": b, "E": e[0],
                                                    "ci": an.ci(e[1]), "within_delta": bool(abs(e[0]) <= an.delta)})
                out["S4"]["b_cross_task"] += _cross_task(an, prim, b, p["role"])
        for cell in [c for c in an.cells() if c[0] == m and c[1] == t]:
            out["S5_anchors_h0"].append({"model": m, "task": t, "role": p["role"], "layer": cell[2], "site": cell[3],
                                         "strategy": cell[4], **h0_cell(an, cell)})
    out["S4"]["c_attribution"] = c3(an, [p for p in pairs if p["role"] == "confirmatory"], what="attr")
    out["S4"]["c_attribution_development"] = c3(an, [p for p in pairs if p["role"] == "development"], what="attr")
    return out


def _cross_task(an: Analysis, cell, basis, role) -> list[dict]:
    g = an.cell_rows(cell)
    x = g[(g.basis == basis) & (g.rule == "cross_task")]
    out = []
    rnd = g[(g.basis == basis) & (g.rule == "random")]
    own = g[(g.basis == basis) & (g.rule == "attrib") & (g.resample == 0)]
    if x.empty or rnd.empty:
        return out
    for i in x.index:
        k = int(x.loc[i, "k_eff"])
        o = own[own.k_eff == k]
        if o.empty:
            continue
        j = o.index[0]

        def excess(ix):
            dr = an.D(cell, basis, "random", float(an.cost_p[ix]), an.cost_b[ix])
            return an.dmg_p[ix] - dr[0], an.dmg_b[ix] - dr[1]
        ex_p, ex_b = excess(i)
        own_p, own_b = excess(j)
        diff = ex_b - 0.5 * own_b
        out.append({"model": cell[0], "task_B": cell[1], "source_task_A": str(x.loc[i, "source_task"]), "basis": basis,
                    "role": role, "k": k, "E_B_units_A": float(ex_p), "E_B_units_B": float(own_p),
                    "specific": bool(ex_p < 0.5 * own_p), "diff_ci": an.ci(diff)})
    return out


def exploratory_cells(an: Analysis) -> pd.DataFrame:
    rows = []
    for cell in an.cells():
        a = an.anchor(cell)
        for b in BASES:
            for r in RULES3 + ("random", "attrib_shuffled"):
                if an.D(cell, b, r, an.c0, an.c0) is None:
                    continue
                rec = {"model": cell[0], "task": cell[1], "layer": cell[2], "site": cell[3], "strategy": cell[4],
                       "basis": b, "rule": r, "cstar": a["c_p"] if a else None}
                for c in [an.c0] + an.c_sec:
                    d = an.D(cell, b, r, c, c)
                    rec[f"D@{c}"], rec[f"D@{c}_lo"], rec[f"D@{c}_hi"] = d[0], *an.ci(d[1])
                    rec[f"never@{c}"] = d[2]
                    e = an.E(cell, b, r, c)
                    if e is not None and r != "random":
                        rec[f"E@{c}"], rec[f"E@{c}_lo"], rec[f"E@{c}_hi"] = e[0], *an.ci(e[1])
                if a:
                    d = an.D(cell, b, r, a["c_p"], a["c_b"])
                    rec["D@cstar"], rec["never@cstar"] = d[0], d[2]
                rows.append(rec)
    return pd.DataFrame(rows)


def clean_json(o):
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    return o


def load_hyp(run_dir: Path) -> dict:
    p = Path(run_dir) / "prereg_snapshot" / "hypotheses.yaml"
    return load_yaml(p)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--precompute":
        n = precompute(Path(sys.argv[2]))
        print(f"[analysis_v2] bootstrap caches up to date for {n} chunks", flush=True)
        sys.exit(0)
    print("usage: python -m pilot.analysis_v2 --precompute RUN_DIR")
    sys.exit(2)
