"""v3 per-model prep: generic continuations, PCA per layer, and one forward + backward pass per discovery example
(query tokens, cached prefix) that yields every selection statistic at all needed layers at once.

Readout value of unit i for example x: its mean over x's kept query tokens (not a prefix token, not a sink).
  meandiff  Cohen's d of the per-example means; rank by |d|
  l1probe   L1 logistic path on standardised per-example means (SAE prefiltered to 8,192 latents by |d|)
  attrib    A_i = E_x[ y * sum_t (v_i - r_i(x_t)) * w_i . dS/dx_t ], S = logit(pos) - logit(neg), v_i the pooled
            query-token mean; accumulated as v * G_x - RG_x (as v2)
  random    uniform; SAE only over latents firing on >= 1% of discovery query tokens
Needed layers per (model, task): every depth for the reference task on Gemma (depth test), the reference depth
otherwise. Bases per layer: sae (included main/fallback file), pca, dims; at the reference layer of the reference
task also `sae_x` (Gemma 16k file, width test). Reservoir (resample): random query tokens, reference cell only.
Readout groups: reference cell, every basis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import bases as BZ
from . import models as M
from . import selection as SEL
from .engine_v3 import MCtx, forward_query, generate, query_lists, query_masks, score_fn
from .prep_v2 import _proj_grad, _sample, _seed
from .sites import GradCapture, MultiCapture, role_masks
from .utils import LOG, read_json, rng_for, write_json
from .v3common import needed_layers, ref, same


def _prep_pca(mc: MCtx, Ls: list[int], norms: dict, tasks: list[str]) -> dict:
    """PCA per layer on generic tokens (not first) and discovery query tokens."""
    target = int(mc.cfg["pca"]["n_tokens"])
    srcs = [(list(mc.tokenized("generic", "fit").input_ids), None)]
    for t in tasks:
        srcs.append((query_lists(mc.tokenized(t, "discovery")), t))
    total = sum(len(x) for s, _ in srcs for x in s)
    p = min(1.0, 1.3 * target / max(total, 1))
    pools = {L: [] for L in Ls}
    gens = {L: torch.Generator().manual_seed(_seed(mc.model_id, "pca-v3", L)) for L in Ls}
    cap = MultiCapture(mc.layers, Ls)
    try:
        with torch.inference_mode():
            for toks, t in srcs:
                pkv = mc.prefix(t) if t else None
                for b in mc.batches(toks):
                    forward_query(mc, pkv, b)
                    m = query_masks(b)[0] if t else role_masks(b)[0]
                    for L in Ls:
                        h = cap.h[L].float()
                        keep = (m & (torch.linalg.vector_norm(h, dim=-1) <= norms[str(L)]["sink_threshold"])).cpu()
                        sel = keep & (torch.rand(keep.shape, generator=gens[L]) < p)
                        pools[L].append(h[sel.to(h.device)].cpu())
    finally:
        cap.remove()
    info = {}
    for L in Ls:
        X = torch.cat(pools[L])
        if X.shape[0] > target:
            X = X[torch.as_tensor(np.sort(rng_for(mc.cfg["seed"], mc.model_id, "pca-trim-v3", L).choice(
                X.shape[0], target, replace=False)))]
        mu, U, ev = BZ.fit_pca(X)
        torch.save({"mu": mu, "U": U, "evals": ev}, mc.dir / "prep" / f"pca_L{L}.pt")
        cum = torch.cumsum(ev, 0) / ev.sum()
        info[str(L)] = {"n_tokens": int(X.shape[0]), "n_components": int(U.shape[0]), "var_top10": float(cum[min(10, len(cum)) - 1])}
    LOG.info("%s PCA per layer: %s", mc.slug, info)
    return info


class _Acc:
    def __init__(self, n_ex: int, n_units: int, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.rsum, self.G, self.RG = z(n_ex, n_units), z(n_ex, n_units), z(n_ex, n_units)
        self.pool, self.fire = z(n_units), z(n_units)
        self.cnt = 0.0


def _task_pass(mc: MCtx, task: str, Ls: list[int], B: dict, norms: dict, res_L: set) -> dict:
    """Forward + backward on every discovery example (query tokens, cached prefix)."""
    disc = mc.tokenized(task, "discovery")
    n = len(disc)
    dev = mc.run.device
    sf = score_fn(mc, task, disc)
    pkv = mc.prefix(task)
    acc = {L: {k: _Acc(n, b.n_units, dev) for k, b in B[L].items()} for L in Ls}
    cnt_ex = {L: torch.zeros(n, device=dev) for L in Ls}
    excl = {L: np.zeros(n, dtype=np.int64) for L in Ls}
    qs = query_lists(disc)
    total = sum(len(t) for t in qs)
    p_res = min(1.0, 1.3 * int(mc.cfg["reservoir"]["prompt"]) / max(total, 1))
    res = {L: [] for L in res_L}
    rgen = {L: torch.Generator().manual_seed(_seed(mc.model_id, task, "reservoir-v3", L)) for L in res_L}
    for b in mc.batches(qs, "grad"):
        gc = GradCapture(mc.layers, Ls)
        try:
            with torch.enable_grad():
                lg = forward_query(mc, pkv, b)[:, -1, :].float()
                sf(lg, b.idx)[0].sum().backward()
        finally:
            gc.remove()
        qm = query_masks(b)[0]
        ex = torch.as_tensor(b.idx, device=dev)
        for L in Ls:
            x = gc.x[L].detach().float()
            gr = gc.g[L].float()
            keep = qm & (torch.linalg.vector_norm(x, dim=-1) <= norms[str(L)]["sink_threshold"])
            excl[L][b.idx] = (qm & ~keep).sum(-1).cpu().numpy()
            bi, ti = keep.nonzero(as_tuple=True)
            xs, gs, rows = x[bi, ti], gr[bi, ti], ex[bi]
            cnt_ex[L].index_add_(0, rows, torch.ones(rows.shape[0], device=dev))
            if L in res_L:
                res[L].append(_sample(xs, p_res, rgen[L]))
            for key, basis in B[L].items():
                a = acc[L][key]
                for i in range(0, xs.shape[0], 2048):
                    r = basis.read(xs[i:i + 2048])
                    gw = _proj_grad(basis if key != "sae_x" else _as_sae(basis), gs[i:i + 2048])
                    rw = rows[i:i + 2048]
                    a.rsum.index_add_(0, rw, r)
                    a.G.index_add_(0, rw, gw)
                    a.RG.index_add_(0, rw, r * gw)
                    a.pool += r.sum(0)
                    if key in ("sae", "sae_x"):
                        a.fire += (r != 0).float().sum(0)
                a.cnt += float(xs.shape[0])
        del gc
    out = {"acc": acc, "cnt_ex": cnt_ex, "excluded": excl, "res_prompt": {}}
    for L in res_L:
        X = torch.cat(res[L])
        N = int(mc.cfg["reservoir"]["prompt"])
        if X.shape[0] > N:
            X = X[torch.as_tensor(np.sort(rng_for(mc.cfg["seed"], mc.model_id, task, "res-trim-v3", L).choice(
                X.shape[0], N, replace=False)))]
        out["res_prompt"][L] = X
    return out


def _as_sae(basis):
    """_proj_grad dispatches on basis.key; the 16k basis is an SAE basis under another key."""
    class _V:
        key = "sae"
        sae = basis.sae
    return _V


def _subsets(cfg: dict, model_id: str, task: str, y: np.ndarray, R: int) -> list[np.ndarray]:
    frac = float(cfg["discovery_resample_frac"])
    subs = []
    for r in range(R):
        rng = rng_for(cfg["seed"], model_id, task, "discovery-resample-v3", r)
        idx = [rng.choice(np.flatnonzero(y == lab), size=max(1, int(round(frac * (y == lab).sum()))), replace=False)
               for lab in (0, 1)]
        subs.append(np.sort(np.concatenate(idx)))
    return subs


def _rank_layer(mc: MCtx, task: str, L: int, B: dict, acc: dict, cnt_ex: torch.Tensor, y_np: np.ndarray):
    cfg = mc.cfg
    dev = mc.run.device
    cc = cfg["cell"]
    subs = _subsets(cfg, mc.model_id, task, y_np, int(cc["discovery_resamples"]))
    y = torch.as_tensor(y_np, device=dev)
    ysign = torch.as_tensor(2.0 * y_np - 1.0, device=dev, dtype=torch.float32)
    orders, caps = {}, {}
    info = {"absd": {}, "A_full": {}, "means": {}, "alive": {}, "n_alive": {}, "l1": {}}
    for key, basis in B.items():
        a = acc[key]
        rbar = a.rsum / cnt_ex.clamp_min(1).unsqueeze(1)
        v_p = a.pool / max(a.cnt, 1.0)
        info["means"][key] = {"prompt": v_p.cpu(), "generated": None}
        c = v_p.unsqueeze(0) * a.G - a.RG
        del a.G, a.RG
        nu = basis.n_units
        alive = None
        if key in ("sae", "sae_x"):
            alive = (a.fire / max(a.cnt, 1.0) >= float(cfg["random_rule"]["sae_min_fire_rate"])).cpu().numpy()
            info["alive"][key], info["n_alive"][key] = alive, int(alive.sum())
        info["absd"][key] = SEL.cohens_d(rbar, y).abs().cpu().numpy().astype(np.float32)
        info["A_full"][key] = (ysign.unsqueeze(1) * c).mean(0).cpu().numpy().astype(np.float32)
        for r, sub in enumerate(subs):
            si = torch.as_tensor(sub, device=dev)
            Rs, ys = rbar[si], y[si]
            for rule in ("meandiff", "l1probe", "attrib"):
                k = f"{key}|{rule}|{r}|-1"
                if rule == "meandiff":
                    order, _, _ = SEL.rank_meandiff(Rs, ys)
                    cap = nu
                elif rule == "l1probe":
                    d_abs = SEL.cohens_d(Rs, ys).abs().cpu().numpy()
                    pre = int(cfg["l1probe"]["sae_prefilter"]) if key in ("sae", "sae_x") else None
                    order, _, dg = SEL.rank_l1probe(Rs, ys, d_abs, cfg["l1probe"], pre, f"{mc.slug}/{task}/L{L}/{key}/r{r}")
                    cap = min(nu, pre) if pre else nu
                    info["l1"][k] = {kk: dg[kk] for kk in ("unconverged_steps", "nonzero_at_weakest", "prefiltered",
                                                          "n_candidates")}
                else:
                    A = (ysign[si].unsqueeze(1) * c[si]).mean(0).cpu().numpy()
                    order, _ = SEL.rank_attrib(A)
                    cap = nu
                orders[k] = np.asarray(order[:cap], dtype=np.int32)
                caps[k] = int(cap)
        for dr in range(int(cc["random_draws"])):
            order, _ = SEL.rank_random(nu, rng_for(cfg["seed"], mc.model_id, task, L, key, "random-v3", dr), alive)
            cap = int(alive.sum()) if alive is not None else nu
            orders[f"{key}|random|-1|{dr}"] = np.asarray(order[:cap], dtype=np.int32)
            caps[f"{key}|random|-1|{dr}"] = cap
        del c
    return orders, caps, info


def readout_groups(cfg: dict, model_id: str, task: str, key: str, absd: np.ndarray, A: np.ndarray, alive) -> list[dict]:
    """Per basis: units split into bands of readout strength |d|; random groups of `group_size` units per band."""
    rg = cfg["controls"]["readout_groups"]
    elig = np.flatnonzero(alive) if alive is not None else np.arange(len(absd))
    order = elig[np.argsort(absd[elig], kind="stable")]
    bands = np.array_split(order, int(rg["n_bands"]))
    rng = rng_for(cfg["seed"], model_id, task, "readout-groups-v3", key)
    out = []
    for qi, q in enumerate(bands):
        for gi in range(int(rg["groups_per_band"])):
            size = min(int(rg["group_size"]), len(q))
            if size == 0:
                continue
            u = np.sort(rng.choice(q, size=size, replace=False))
            out.append({"band": qi, "group": gi, "units": u.astype(np.int64), "strength": float(absd[u].mean()),
                        "attr_sum": float(A[u].sum())})
    return out


def stage_prep(mc: MCtx, tasks: list[str], ref_task: str) -> dict:
    cfg = mc.cfg
    pdir = mc.dir / "prep"
    pdir.mkdir(parents=True, exist_ok=True)
    norms = read_json(pdir / "norms.json")
    r = ref(cfg)
    ref_L = mc.layer(float(r["depth"]))
    per_task = {t: needed_layers(cfg, mc.model_id, t, mc.n_layers, ref_task) for t in tasks}
    all_L = sorted({L for v in per_task.values() for L in v})
    summary = {"layers": all_L, "per_task": per_task, "reference_layer": ref_L}
    n_cont = int(cfg["cost"]["continuation_tokens"])
    gen = mc.tokenized("generic", "cost")
    conts, _ = generate(mc, list(gen.input_ids), n_cont, min_new=n_cont)
    torch.save({"cont": conts, "n": n_cont}, pdir / "generic_cont.pt")
    summary["pca"] = _prep_pca(mc, all_L, norms, tasks)
    is_ref_model = mc.model_id == r["model"]
    for task in tasks:
        tdir = pdir / task
        tdir.mkdir(parents=True, exist_ok=True)
        Ls = per_task[task]
        ref_cell = is_ref_model and task == ref_task
        B = {}
        for L in Ls:
            keys = list(cfg["bases"]) + (["sae_x"] if ref_cell and L == ref_L else [])
            B[L] = mc.bases(L, keys)
        res_L = {ref_L} if (ref_cell and ref_L in Ls) else set()
        tp = _task_pass(mc, task, Ls, B, norms, res_L)
        y_np = mc.tokenized(task, "discovery").label.to_numpy().astype(int)
        for L in Ls:
            orders, caps, info = _rank_layer(mc, task, L, B[L], tp["acc"][L], tp["cnt_ex"][L], y_np)
            rbar = tp["acc"][L]["dims"].rsum / tp["cnt_ex"][L].clamp_min(1).unsqueeze(1)
            yt = torch.as_tensor(y_np, device=rbar.device)
            u = rbar[yt == 1].mean(0) - rbar[yt == 0].mean(0)
            u = (u / u.norm().clamp_min(1e-12)).cpu()
            gq = torch.Generator().manual_seed(_seed(mc.model_id, task, "random-dir-v3", L))
            Rd = torch.randn(int(cfg["cell"]["random_dir"]), mc.d_model, generator=gq)
            Rd = Rd / Rd.norm(dim=-1, keepdim=True)
            groups = {}
            if is_ref_model and same(mc.layer(float(r["depth"])), L):
                for key in B[L]:
                    if key == "sae_x":
                        continue
                    groups[key] = readout_groups(cfg, mc.model_id, task, key, info["absd"][key], info["A_full"][key],
                                                 info["alive"].get(key))
            ex = tp["excluded"][L]
            rec = {"means": info["means"], "dims_mean": info["means"]["dims"],
                   "anchors": {"massmean": u, "random_dir": Rd}, "alive": info["alive"], "n_alive": info["n_alive"],
                   "absd": info["absd"], "A_full": info["A_full"], "caps": caps, "groups": groups, "bases": list(B[L]),
                   "n_examples": int(len(y_np)), "l1": info["l1"],
                   "excluded_sinks_per_prompt": {"mean": float(ex.mean()), "max": int(ex.max()) if len(ex) else 0},
                   "reservoir": tp["res_prompt"].get(L)}
            torch.save(rec, tdir / f"L{L}.pt")
            np.savez(tdir / f"L{L}_orders.npz", **{k.replace("|", "__"): v for k, v in orders.items()})
            top = [{"model": mc.model_id, "task": task, "layer": L, "ranking": k, "rank": i, "unit": int(u_)}
                   for k, o in orders.items() for i, u_ in enumerate(o[:50])]
            pd.DataFrame(top).to_parquet(tdir / f"L{L}_top50.parquet", index=False)
            summary.setdefault("tasks", {}).setdefault(task, {})[str(L)] = {
                "bases": list(B[L]), "n_alive_sae": info["n_alive"], "reservoir": None if rec["reservoir"] is None
                else int(rec["reservoir"].shape[0])}
        LOG.info("%s/%s: selection done at layers %s (bases %s)", mc.slug, task, Ls, {L: list(B[L]) for L in Ls})
        del tp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(pdir / "prep_summary.json", summary)
    return {"layers": all_L, "tasks": tasks}


def load_orders(mc: MCtx, task: str, L: int) -> dict:
    z = np.load(mc.dir / "prep" / task / f"L{L}_orders.npz")
    return {k.replace("__", "|"): z[k] for k in z.files}
