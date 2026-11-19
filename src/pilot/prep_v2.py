"""v2 per-model prep: generic continuations, PCA per depth, and one forward + backward pass per discovery
example that yields every selection statistic at all depths at once.

Readout value of unit i for example x: its mean over x's kept prompt tokens (not first, not sink).
  meandiff  Cohen's d of the per-example means; rank by |d|
  l1probe   L1 logistic path on standardised per-example means (SAE prefiltered to 8,192 latents by |d|)
  attrib    A_i = E_x[ sum_t (v_i - r_i(x_t)) * w_i . dS/dx_t ], S = y * m(x) (single-token) or y * r(x) (refusal,
            first-token score), v_i the pooled prompt-role mean. Accumulated as v * G_x - RG_x with
            G_x = sum_t w.g_t and RG_x = sum_t r_t * (w.g_t), so v (known only after the pass) enters linearly and
            shuffled-label attribution needs no extra backward pass.
  random    uniform; SAE only over latents firing on >= 1% of discovery tokens
Means: one value per unit and role (prompt: discovery prompt tokens; generated: clean greedy replies to the
refusal discovery prompts). Reservoirs (refusal, primary depth): random prompt- and generated-role tokens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import bases as BZ
from . import metrics as MET
from . import models as M
from . import selection as SEL
from .engine_v2 import MCtx, generate
from .sites import GradCapture, MultiCapture, role_masks
from .utils import LOG, read_json, rng_for, write_json
from .v2common import tasks_in


def _seed(*keys) -> int:
    return int(rng_for(0, *keys).integers(2**31))


def _proj_grad(basis, g):
    if basis.key == "dims":
        return g
    if basis.key == "pca":
        return g @ basis.U.T
    if basis.key == "sae":
        return g @ basis.sae.directions().T
    raise ValueError(basis.key)


def _prep_pca(mc: MCtx, Ls: list[int], norms: dict, tasks: list[str]) -> dict:
    toks = list(mc.tokenized("generic", "fit").input_ids)
    for t in tasks:
        toks += list(mc.tokenized(t, "discovery").input_ids)
    target = int(mc.cfg["pca"]["n_tokens"])
    total = sum(len(t) - 1 for t in toks)
    p = min(1.0, 1.3 * target / max(total, 1))
    pools = {L: [] for L in Ls}
    gens = {L: torch.Generator().manual_seed(_seed(mc.model_id, "pca", L)) for L in Ls}
    cap = MultiCapture(mc.layers, Ls)
    try:
        with torch.inference_mode():
            for b in mc.batches(toks):
                mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                         use_cache=False, logits_to_keep=1)
                pm = role_masks(b)[0]
                for L in Ls:
                    h = cap.h[L].float()
                    keep = (pm & (torch.linalg.vector_norm(h, dim=-1) <= norms[str(L)]["sink_threshold"])).cpu()
                    sel = keep & (torch.rand(keep.shape, generator=gens[L]) < p)
                    pools[L].append(h[sel.to(h.device)].cpu())
    finally:
        cap.remove()
    info = {}
    for L in Ls:
        X = torch.cat(pools[L])
        if X.shape[0] > target:
            X = X[torch.as_tensor(np.sort(rng_for(mc.cfg["seed"], mc.model_id, "pca-trim", L).choice(X.shape[0], target, replace=False)))]
        mu, U, ev = BZ.fit_pca(X)
        torch.save({"mu": mu, "U": U, "evals": ev}, mc.dir / "prep" / f"pca_L{L}.pt")
        cum = torch.cumsum(ev, 0) / ev.sum()
        info[str(L)] = {"n_tokens": int(X.shape[0]), "n_components": int(U.shape[0]),
                        "var_top10": float(cum[min(10, len(cum)) - 1])}
    LOG.info("%s PCA per layer: %s", mc.slug, info)
    return info


class _Acc:
    def __init__(self, n_ex: int, n_units: int, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.rsum, self.G, self.RG = z(n_ex, n_units), z(n_ex, n_units), z(n_ex, n_units)
        self.pool, self.fire, self.gpool = z(n_units), z(n_units), z(n_units)
        self.cnt, self.gcnt = 0.0, 0.0


def _sample(X: torch.Tensor, p: float, gen: torch.Generator) -> torch.Tensor:
    return X[(torch.rand(X.shape[0], generator=gen) < p).to(X.device)].to(torch.bfloat16).cpu()


def _task_pass(mc: MCtx, task: str, Ls: list[int], B: dict, norms: dict, res_L: set) -> dict:
    """Forward + backward on every discovery example; accumulates per-basis statistics at every layer."""
    cfg = mc.cfg
    disc = mc.tokenized(task, "discovery")
    n = len(disc)
    dev = mc.run.device
    g = mc.gate["tasks"][task]
    if task == "refusal":
        sf = lambda lg: MET.refusal_scores(lg, g["start_ids"])[0]  # noqa: E731
    else:
        sf = lambda lg: MET.truth_scores(lg, g["answer_ids"])[0]  # noqa: E731
    acc = {L: {k: _Acc(n, b.n_units, dev) for k, b in B[L].items()} for L in Ls}
    cnt_ex = {L: torch.zeros(n, device=dev) for L in Ls}
    excl = {L: np.zeros(n, dtype=np.int64) for L in Ls}
    total = sum(len(t) - 1 for t in disc.input_ids)
    rcfg = cfg["reservoir"]
    p_res = min(1.0, 1.3 * int(rcfg["prompt"]) / max(total, 1))
    res = {L: [] for L in res_L}
    rgen = {L: torch.Generator().manual_seed(_seed(mc.model_id, task, "reservoir-prompt", L)) for L in res_L}
    for b in mc.batches(list(disc.input_ids), "grad"):
        gc = GradCapture(mc.layers, Ls)
        try:
            with torch.enable_grad():
                sf(M.last_logits(mc.model, b)).sum().backward()
        finally:
            gc.remove()
        pm = role_masks(b)[0]
        ex = torch.as_tensor(b.idx, device=dev)
        for L in Ls:
            x = gc.x[L].detach().float()
            gr = gc.g[L].float()
            keep = pm & (torch.linalg.vector_norm(x, dim=-1) <= norms[str(L)]["sink_threshold"])
            excl[L][b.idx] = (pm & ~keep).sum(-1).cpu().numpy() + 1
            bi, ti = keep.nonzero(as_tuple=True)
            xs, gs, rows = x[bi, ti], gr[bi, ti], ex[bi]
            cnt_ex[L].index_add_(0, rows, torch.ones(rows.shape[0], device=dev))
            if L in res_L:
                res[L].append(_sample(xs, p_res, rgen[L]))
            for key, basis in B[L].items():
                a = acc[L][key]
                for i in range(0, xs.shape[0], 2048):
                    r = basis.read(xs[i:i + 2048])
                    gw = _proj_grad(basis, gs[i:i + 2048])
                    rw = rows[i:i + 2048]
                    a.rsum.index_add_(0, rw, r)
                    a.G.index_add_(0, rw, gw)
                    a.RG.index_add_(0, rw, r * gw)
                    a.pool += r.sum(0)
                    if key == "sae":
                        a.fire += (r != 0).float().sum(0)
                a.cnt += float(xs.shape[0])
        del gc
    out = {"acc": acc, "cnt_ex": cnt_ex, "excluded": excl, "res_prompt": {}}
    for L in res_L:
        X = torch.cat(res[L])
        N = int(rcfg["prompt"])
        if X.shape[0] > N:
            X = X[torch.as_tensor(np.sort(rng_for(cfg["seed"], mc.model_id, task, "res-trim", L).choice(X.shape[0], N, replace=False)))]
        out["res_prompt"][L] = X
    return out


def _gen_role(mc: MCtx, task: str, Ls: list[int], B: dict, norms: dict, acc: dict, res_L: set) -> dict:
    """Clean greedy replies to the discovery prompts; generated-role pooled sums and reservoir."""
    cfg = mc.cfg
    disc = mc.tokenized(task, "discovery")
    nr = int(cfg["gen_role"]["discovery_reply_tokens"])
    replies, _ = generate(mc, list(disc.input_ids), nr)
    seqs = [list(p) + list(r) for p, r in zip(disc.input_ids, replies)]
    plen = np.array([len(p) for p in disc.input_ids])
    total = sum(len(r) for r in replies)
    N = int(cfg["reservoir"]["generated"])
    p = min(1.0, 1.3 * N / max(total, 1))
    res = {L: [] for L in res_L}
    rgen = {L: torch.Generator().manual_seed(_seed(mc.model_id, task, "reservoir-gen", L)) for L in res_L}
    cap = MultiCapture(mc.layers, Ls)
    try:
        with torch.inference_mode():
            for b in mc.batches(seqs):
                mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                         use_cache=False, logits_to_keep=1)
                _, _, gm, _ = role_masks(b, plen[b.idx])
                for L in Ls:
                    x = cap.h[L].float()
                    keep = gm & (torch.linalg.vector_norm(x, dim=-1) <= norms[str(L)]["sink_threshold"])
                    xs = x[keep]
                    if L in res_L:
                        res[L].append(_sample(xs, p, rgen[L]))
                    for key, basis in B[L].items():
                        a = acc[L][key]
                        for i in range(0, xs.shape[0], 2048):
                            a.gpool += basis.read(xs[i:i + 2048]).sum(0)
                        a.gcnt += float(xs.shape[0])
    finally:
        cap.remove()
    out = {}
    for L in res_L:
        X = torch.cat(res[L])
        if X.shape[0] > N:
            X = X[torch.as_tensor(np.sort(rng_for(cfg["seed"], mc.model_id, task, "res-gen-trim", L).choice(X.shape[0], N, replace=False)))]
        out[L] = X
    LOG.info("%s/%s: generated-role statistics from %d clean replies (%d tokens)", mc.slug, task, len(replies), total)
    return out


def _subsets(cfg: dict, model_id: str, task: str, y: np.ndarray, R: int) -> list[np.ndarray]:
    frac = float(cfg["discovery_resample_frac"])
    subs = []
    for r in range(R):
        rng = rng_for(cfg["seed"], model_id, task, "discovery-resample", r)
        idx = [rng.choice(np.flatnonzero(y == lab), size=max(1, int(round(frac * (y == lab).sum()))), replace=False)
               for lab in (0, 1)]
        subs.append(np.sort(np.concatenate(idx)))
    return subs


def _rank_layer(mc: MCtx, task: str, L: int, B: dict, acc: dict, cnt_ex: torch.Tensor, y_np: np.ndarray,
                primary: bool) -> tuple[dict, dict, list]:
    cfg = mc.cfg
    dev = mc.run.device
    t1, t2 = cfg["tiers"]["tier1"], cfg["tiers"]["tier2"]
    R = int((t1 if primary else t2)["discovery_resamples"])
    draws = int((t1 if primary else t2)["random_draws"])
    subs = _subsets(cfg, mc.model_id, task, y_np, R)
    y = torch.as_tensor(y_np, device=dev)
    ysign = torch.as_tensor(2.0 * y_np - 1.0, device=dev, dtype=torch.float32)
    orders, caps, stab = {}, {}, []
    info = {"absd": {}, "A_full": {}, "means": {}, "alive": None, "n_alive": None}
    for key, basis in B.items():
        a = acc[key]
        rbar = a.rsum / cnt_ex.clamp_min(1).unsqueeze(1)
        v_p = a.pool / max(a.cnt, 1.0)
        v_g = (a.gpool / a.gcnt) if a.gcnt > 0 else None
        info["means"][key] = {"prompt": v_p.cpu(), "generated": None if v_g is None else v_g.cpu()}
        c = v_p.unsqueeze(0) * a.G - a.RG                            # per-example contribution for S = score
        del a.G, a.RG
        nu = basis.n_units
        alive = None
        if key == "sae":
            alive = (a.fire / max(a.cnt, 1.0) >= float(cfg["random_rule"]["sae_min_fire_rate"])).cpu().numpy()
            info["alive"], info["n_alive"] = alive, int(alive.sum())
        d_full = SEL.cohens_d(rbar, y)
        info["absd"][key] = d_full.abs().cpu().numpy().astype(np.float32)
        info["A_full"][key] = (ysign.unsqueeze(1) * c).mean(0).cpu().numpy().astype(np.float32)
        by_rule = {}
        for r, sub in enumerate(subs):
            si = torch.as_tensor(sub, device=dev)
            Rs, ys = rbar[si], y[si]
            for rule in cfg["rules"]:
                if rule == "random":
                    continue
                k = f"{key}|{rule}|{r}|-1"
                if rule == "meandiff":
                    order, _, _ = SEL.rank_meandiff(Rs, ys)
                    cap = nu
                elif rule == "l1probe":
                    d_abs = SEL.cohens_d(Rs, ys).abs().cpu().numpy()
                    pre = int(cfg["l1probe"]["sae_prefilter"]) if key == "sae" else None
                    order, _, dg = SEL.rank_l1probe(Rs, ys, d_abs, cfg["l1probe"], pre, f"{mc.slug}/{task}/L{L}/{key}/r{r}")
                    cap = min(nu, pre) if pre else nu
                    info.setdefault("l1", {})[k] = {kk: dg[kk] for kk in ("unconverged_steps", "nonzero_at_weakest",
                                                                         "prefiltered", "n_candidates")}
                elif rule == "attrib":
                    A = (ysign[si].unsqueeze(1) * c[si]).mean(0).cpu().numpy()
                    order, _ = SEL.rank_attrib(A)
                    cap = nu
                else:
                    raise ValueError(rule)
                orders[k] = np.asarray(order[:cap], dtype=np.int32)
                caps[k] = int(cap)
                by_rule.setdefault(rule, []).append(order)
            if primary and r == 0 and cfg["controls"].get("attrib_shuffled", True):
                perm = rng_for(cfg["seed"], mc.model_id, task, "shuffle-labels").permutation(len(sub))
                ysh = ysign[si][torch.as_tensor(perm, device=dev)]
                A = (ysh.unsqueeze(1) * c[si]).mean(0).cpu().numpy()
                order, _ = SEL.rank_attrib(A)
                orders[f"{key}|attrib_shuffled|0|-1"] = np.asarray(order, dtype=np.int32)
                caps[f"{key}|attrib_shuffled|0|-1"] = nu
        if "random" in cfg["rules"]:
            for dr in range(draws):
                order, _ = SEL.rank_random(nu, rng_for(cfg["seed"], mc.model_id, task, L, key, "random", dr), alive)
                cap = int(alive.sum()) if alive is not None else nu
                orders[f"{key}|random|-1|{dr}"] = np.asarray(order[:cap], dtype=np.int32)
                caps[f"{key}|random|-1|{dr}"] = cap
        ladder = [int(k) for k in (t1 if primary else t2)["budgets"]]
        for rule, os_ in by_rule.items():
            for kk in ladder:
                jm, jmin, npairs = SEL.jaccard_topk(os_, kk)
                if npairs:
                    stab.append({"model": mc.model_id, "task": task, "layer": L, "basis": key, "rule": rule, "k": kk,
                                 "jaccard_mean": jm, "jaccard_min": jmin, "n_pairs": npairs})
        del c
    return orders, caps, {"info": info, "stability": stab}


def _c3_groups(cfg: dict, model_id: str, task: str, key: str, absd: np.ndarray, A: np.ndarray, alive) -> list[dict]:
    c3 = cfg["controls"]["c3"]
    elig = np.flatnonzero(alive) if alive is not None else np.arange(len(absd))
    order = elig[np.argsort(absd[elig], kind="stable")]
    qs = np.array_split(order, int(c3["n_quintiles"]))
    rng = rng_for(cfg["seed"], model_id, task, "c3", key)
    out = []
    for qi, q in enumerate(qs):
        for gi in range(int(c3["groups_per_quintile"])):
            size = min(int(c3["group_size"]), len(q))
            if size == 0:
                continue
            u = np.sort(rng.choice(q, size=size, replace=False))
            out.append({"quintile": qi, "group": gi, "units": u.astype(np.int64), "strength": float(absd[u].mean()),
                        "attr_sum": float(A[u].sum())})
    return out


def stage_prep(mc: MCtx, tasks: list[str]) -> dict:
    cfg = mc.cfg
    pdir = mc.dir / "prep"
    pdir.mkdir(parents=True, exist_ok=True)
    norms = read_json(pdir / "norms.json")
    Ls = sorted(set(mc.layer(d) for d in mc.depths()))
    prim = float(cfg["primary_depth"])
    primary_L = mc.layer(prim) if any(abs(prim - d) < 1e-9 for d in mc.depths()) else None
    summary = {"layers": Ls, "primary_layer": primary_L}
    # generic continuations (clean, greedy, exactly n tokens) for the teacher-forced cost
    n_cont = int(cfg["cost"]["continuation_tokens"])
    gen = mc.tokenized("generic", "cost")
    conts, _ = generate(mc, list(gen.input_ids), n_cont, min_new=n_cont)
    torch.save({"cont": conts, "n": n_cont}, pdir / "generic_cont.pt")
    summary["pca"] = _prep_pca(mc, Ls, norms, tasks)
    stab_all = []
    for task in tasks:
        tdir = pdir / task
        tdir.mkdir(parents=True, exist_ok=True)
        B = {L: mc.bases(L) for L in Ls}
        res_L = {primary_L} if (task == "refusal" and primary_L is not None) else set()
        tp = _task_pass(mc, task, Ls, B, norms, res_L)
        res_gen = _gen_role(mc, task, Ls, B, norms, tp["acc"], res_L) if task == "refusal" else {}
        y_np = mc.tokenized(task, "discovery").label.to_numpy().astype(int)
        for L in Ls:
            primary = L == primary_L
            orders, caps, rk = _rank_layer(mc, task, L, B[L], tp["acc"][L], tp["cnt_ex"][L], y_np, primary)
            info = rk["info"]
            stab_all += rk["stability"]
            dims_mean = info["means"]["dims"]
            d = mc.d_model
            rbar_dims = tp["acc"][L]["dims"].rsum / tp["cnt_ex"][L].clamp_min(1).unsqueeze(1)
            yt = torch.as_tensor(y_np, device=rbar_dims.device)
            u = rbar_dims[yt == 1].mean(0) - rbar_dims[yt == 0].mean(0)
            u = (u / u.norm()).cpu()
            n_rd = max(int(cfg["tiers"]["tier1"]["random_dir"]), int(cfg["tiers"]["tier2"]["random_dir"]))
            gq = torch.Generator().manual_seed(_seed(mc.model_id, task, "random-dir", L))
            Rd = torch.randn(n_rd, d, generator=gq)
            Rd = Rd / Rd.norm(dim=-1, keepdim=True)
            groups = {}
            if primary:
                for key in B[L]:
                    groups[key] = _c3_groups(cfg, mc.model_id, task, key, info["absd"][key], info["A_full"][key],
                                             info["alive"] if key == "sae" else None)
            ex = tp["excluded"][L]
            rec = {"means": info["means"], "dims_mean": dims_mean, "anchors": {"massmean": u, "random_dir": Rd},
                   "alive": info["alive"], "n_alive": info["n_alive"], "absd": info["absd"], "A_full": info["A_full"],
                   "caps": caps, "groups": groups, "bases": list(B[L]), "n_examples": int(len(y_np)),
                   "excluded_per_prompt": {"mean": float(ex.mean()), "max": int(ex.max()),
                                           "frac_prompts_with_sink": float((ex > 1).mean())},
                   "l1": info.get("l1", {}),
                   "reservoir": ({"prompt": tp["res_prompt"][L], "generated": res_gen.get(L)} if L in res_L else None)}
            torch.save(rec, tdir / f"L{L}.pt")
            np.savez(tdir / f"L{L}_orders.npz", **{k.replace("|", "__"): v for k, v in orders.items()})
            top = [{"model": mc.model_id, "task": task, "layer": L, "ranking": k, "rank": i, "unit": int(u_)}
                   for k, o in orders.items() for i, u_ in enumerate(o[:50])]
            pd.DataFrame(top).to_parquet(tdir / f"L{L}_top50.parquet", index=False)
            summary.setdefault("tasks", {}).setdefault(task, {})[str(L)] = {
                "bases": list(B[L]), "n_alive_sae": info["n_alive"], "excluded_per_prompt": rec["excluded_per_prompt"],
                "reservoir": {k: (None if v is None else int(v.shape[0])) for k, v in (rec["reservoir"] or {}).items()}}
        LOG.info("%s/%s: selection done at layers %s (bases per layer %s)", mc.slug, task, Ls,
                 {L: list(B[L]) for L in Ls})
        del tp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(stab_all).to_csv(pdir / "stability.csv", index=False)
    write_json(pdir / "prep_summary.json", summary)
    return {"layers": Ls, "tasks": tasks}


def load_orders(mc: MCtx, task: str, L: int) -> dict:
    z = np.load(mc.dir / "prep" / task / f"L{L}_orders.npz")
    return {k.replace("__", "|"): z[k] for k in z.files}


__all__ = ["stage_prep", "load_orders", "tasks_in"]
