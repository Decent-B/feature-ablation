"""`ablate` stage: every configuration x budget as a full (or layer-split) forward pass with
the ablation hook at layer L on eval and generic prompts; per-example scores saved in
resumable chunks; then the identity check, the fast-path equivalence test and the
generation checks.

Chunk = (task, strategy, basis) or (task, strategy, "anchor"); each chunk writes
ablate/per_example/<task>__<strategy>__<group>.npz plus a .parquet config table.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bases as BZ
from . import metrics as MET
from . import models as M
from .pipeline import TASKS, Ctx, HarnessInvalid, load_bases
from .utils import LOG, load_run_file, read_json, rng_for, write_json


class TaskRunner:
    """Holds prompt batches, clean references and resample assignments for one task."""

    def __init__(self, ctx: Ctx, task: str, fast_path: bool):
        self.ctx, self.task = ctx, task
        self.T, self.L = ctx.T, ctx.L
        self.ev = ctx.split(task, "eval")
        self.gen = ctx.split("generic", "cost")
        self.labels = self.ev.label.to_numpy()
        self.score_fn = ctx.score_fn(task)
        self.eval_b = ctx.batches(list(self.ev.input_ids))
        self.gen_b = ctx.batches(list(self.gen.input_ids))
        self.fast = False
        self.tokens = 0
        self.fwd_seconds = 0.0
        c = load_run_file(ctx.run_dir / "cache" / f"{task}.pt")
        self.cache = c
        self.disc_int = c["discovery_int"].to(ctx.device)
        self.disc_mlp = c["discovery_mlp"].to(ctx.device) if c["discovery_mlp"] is not None else None
        n_disc = self.disc_int.shape[0]
        # one fixed discovery source example per prompt (common random numbers across configs)
        self.src = {"eval": rng_for(ctx.cfg["seed"], task, "resample-src", "eval").integers(0, n_disc, len(self.ev)),
                    "gen": rng_for(ctx.cfg["seed"], task, "resample-src", "gen").integers(0, n_disc, len(self.gen))}
        if fast_path:
            self._prepare_fast()

    # ------------------------------------------------------------------ fast path
    @torch.inference_mode()
    def _prepare_fast(self):
        cap = M.Capture(self.ctx.layers, self.L, None, mlp=self.ctx.use_mlp)
        try:
            for b in self.eval_b + self.gen_b:
                M.last_logits(self.ctx.model, b)
                b.cache["h"] = cap.h.clone()
                if self.ctx.use_mlp:
                    b.cache["m"] = cap.m[:, -self.T:, :].clone()
        finally:
            cap.remove()
        self.fast = True

    def drop_fast(self, why: str):
        LOG.warning("fast path disabled for %s: %s; using full forward passes", self.task, why)
        for b in self.eval_b + self.gen_b:
            b.cache.clear()
        self.fast = False

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def run(self, batches, hook: M.AblationHook | None, src_key: str, collect):
        model, layers = self.ctx.model, self.ctx.layers
        split = M.LayerSplit(model, self.L) if self.fast else None
        if split is not None:
            split.__enter__()
        handles = []
        try:
            if hook is not None:
                if hook.values_src is not None:
                    hook.src_index = torch.as_tensor(self.src[src_key], dtype=torch.long)
                handles = M.attach_hook(layers, self.L, hook, self.fast)
            elif split is not None:
                pass
            for b in batches:
                if split is not None:
                    split.inject.value = b.cache["h"]
                if hook is not None:
                    hook.expected_len = b.S
                    hook.batch_idx = b.idx
                    if hook.needs_mlp and self.fast:
                        hook.mlp_cache = b.cache["m"]
                t0 = time.time()
                logits = M.last_logits(model, b)
                if logits.is_cuda:
                    torch.cuda.synchronize()
                self.fwd_seconds += time.time() - t0
                self.tokens += b.n_tokens
                collect(b, logits, hook)
        finally:
            for h in handles:
                h.remove()
            if split is not None:
                split.__exit__(None, None, None)

    def eval_scores(self, hook):
        n = len(self.ev)
        s, a = np.zeros(n, np.float32), np.zeros(n, np.float32)

        def col(b, logits, hk):
            x, y = self.score_fn(logits)
            s[b.idx] = x.cpu().numpy()
            a[b.idx] = y.cpu().numpy()
        self.run(self.eval_b, hook, "eval", col)
        return s, a

    def gen_logprobs(self):
        out = [None] * len(self.gen_b)

        def col(b, logits, hk):
            out[self._gi[id(b)]] = torch.log_softmax(logits, -1)
        self._gi = {id(b): i for i, b in enumerate(self.gen_b)}
        self.run(self.gen_b, None, "gen", col)
        return out

    def gen_cost(self, hook, clean_lp):
        n = len(self.gen)
        kl, rel = np.zeros(n, np.float32), np.zeros(n, np.float32)
        gi = {id(b): i for i, b in enumerate(self.gen_b)}

        def col(b, logits, hk):
            kl[b.idx] = MET.kl_from_logprobs(clean_lp[gi[id(b)]], logits).cpu().numpy()
            rel[b.idx] = hk.rel_change.cpu().numpy() if hk is not None and hk.rel_change is not None else 0.0
        self.run(self.gen_b, hook, "gen", col)
        return kl, rel

    def full_logits(self, batches):
        out = {}
        was = self.fast
        self.fast = False
        try:
            def col(b, logits, hk):
                out[id(b)] = logits.clone()
            self.run(batches, None, "eval", col)
        finally:
            self.fast = was
        return out

    def fast_logits(self, batches):
        out = {}

        def col(b, logits, hk):
            out[id(b)] = logits.clone()
        self.run(batches, None, "eval", col)
        return out


# ---------------------------------------------------------------------------- hooks per config

def make_hook(basis, idx: np.ndarray, strategy: str, means: torch.Tensor, runner: TaskRunner, T: int):
    dev = runner.ctx.device
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
    W = basis.directions(idx_t).float()
    reader = basis.reader(idx_t)
    if strategy == "mean":
        return M.AblationHook(T, W, reader, values_static=means[:, idx_t.cpu()].to(dev), needs_mlp=basis.needs_mlp)
    if strategy == "zero":
        return M.AblationHook(T, W, reader, values_static=torch.zeros(T, len(idx), device=dev), needs_mlp=basis.needs_mlp)
    if strategy == "resample":
        with torch.inference_mode():
            vals = []
            for i in range(0, runner.disc_int.shape[0], 128):
                x = runner.disc_int[i:i + 128].float()
                m = None if runner.disc_mlp is None else runner.disc_mlp[i:i + 128].float()
                vals.append(basis.read_sel(x, m, idx_t))
            src = torch.cat(vals)                                    # [N_disc, T, k]
        return M.AblationHook(T, W, reader, values_src=src, src_index=np.zeros(1, dtype=np.int64),
                              needs_mlp=basis.needs_mlp)
    raise ValueError(strategy)


# ---------------------------------------------------------------------------- checks

def identity_check(ctx: Ctx, runner: TaskRunner, B: dict, rankings: pd.DataFrame, tol: float) -> dict:
    """Ablate with v_i = r_i(x), where r comes from the separately cached clean activations,
    on the full model forward: logits must not move (bf16 tolerance)."""
    # Reuse the runner's eval batches: they have exactly the batch composition/padding the cache
    # stage used, so cached activations match the live forward bit-for-bit (re-batching a subset
    # changes padding/shapes and hence bf16 numerics, which fails the check spuriously).
    sub, n = [], 0
    for b in runner.eval_b:
        sub.append(b)
        n += len(b.idx)
        if n >= 32:
            break
    rows = np.concatenate([b.idx for b in sub])
    clean = runner.full_logits(sub)
    N = len(runner.ev)
    ev_int = runner.cache["eval_int"][torch.as_tensor(rows)].to(ctx.device)
    ev_mlp = runner.cache["eval_mlp"][torch.as_tensor(rows)].to(ctx.device) if runner.cache["eval_mlp"] is not None else None
    src_index = torch.zeros(N, dtype=torch.long)
    src_index[torch.as_tensor(rows)] = torch.arange(len(rows))
    kmax = max(int(k) for k in ctx.cfg["budgets"])
    res = {}
    for key, basis in B.items():
        sel = rankings[(rankings.task == runner.task) & (rankings.basis == key)]
        r = sel[(sel.rule == "meandiff") & (sel["resample"] == 0)]
        if r.empty:                                     # meandiff disabled: any ranking of this basis
            first = sel.iloc[0]
            r = sel[(sel.rule == first.rule) & (sel["resample"] == first["resample"]) & (sel["draw"] == first["draw"])]
        idx = r.sort_values("rank").unit.to_numpy().copy()[:min(kmax, basis.n_units)]
        if len(idx) == 0:
            raise HarnessInvalid(f"identity check for {runner.task}/{key}: no ranked units to ablate")
        idx_t = torch.as_tensor(idx, device=ctx.device)
        with torch.inference_mode():
            vals = basis.read_sel(ev_int.float(), None if ev_mlp is None else ev_mlp.float(), idx_t)
        hook = M.AblationHook(runner.T, basis.directions(idx_t).float(), basis.reader(idx_t), values_src=vals,
                              src_index=src_index, needs_mlp=basis.needs_mlp)
        was = runner.fast
        runner.fast = False
        diffs = []
        try:
            def col(b, logits, hk):
                diffs.append(float((logits - clean[id(b)]).abs().max()))
            handles = M.attach_hook(ctx.layers, runner.L, hook, False)
            try:
                with torch.inference_mode():
                    for b in sub:
                        hook.expected_len, hook.batch_idx = b.S, b.idx
                        col(b, M.last_logits(ctx.model, b), hook)
            finally:
                for h in handles:
                    h.remove()
        finally:
            runner.fast = was
        res[key] = {"k": int(len(idx)), "max_abs_logit_change": max(diffs), "pass": max(diffs) < tol,
                    "hook_fired": hook.fired}
        LOG.info("identity check %s/%s (k=%d): max |dlogit| = %.2e (tol %.0e) -> %s", runner.task, key, len(idx),
                 max(diffs), tol, "PASS" if max(diffs) < tol else "FAIL")
    return res


def fast_path_check(ctx: Ctx, runner: TaskRunner, tol: float) -> dict:
    sub_b = []
    n = 0
    for b in runner.eval_b + runner.gen_b:
        sub_b.append(b)
        n += len(b.idx)
        if n >= 64:
            break
    full = runner.full_logits(sub_b)
    fast = runner.fast_logits(sub_b)
    d = max(float((full[id(b)] - fast[id(b)]).abs().max()) for b in sub_b)
    ok = d < tol
    LOG.info("fast-path equivalence (%s, %d prompts): max |dlogit| vs full forward = %.2e (tol %.0e) -> %s",
             runner.task, n, d, tol, "enabled" if ok else "DISABLED")
    return {"n_prompts": n, "max_abs_logit_diff": d, "tol": tol, "enabled": ok}


# ---------------------------------------------------------------------------- config enumeration

def enumerate_configs(ctx: Ctx, task: str, rankings: pd.DataFrame, B: dict) -> list[dict]:
    cfg = ctx.cfg
    out = []
    for strategy in cfg["strategies"]:
        for key, basis in B.items():
            rk = rankings[(rankings.task == task) & (rankings.basis == key)]
            for (rule, res, draw), g in rk.groupby(["rule", "resample", "draw"]):
                order = g.sort_values("rank").unit.to_numpy().copy()
                for k in cfg["budgets"]:
                    k_eff = min(int(k), basis.n_units)
                    out.append({"task": task, "strategy": strategy, "group": key, "basis": key, "rule": rule,
                                "resample": int(res), "draw": int(draw), "k": int(k), "k_eff": k_eff,
                                "units": order[:k_eff]})
        if cfg["anchors"].get("massmean", True):
            out.append({"task": task, "strategy": strategy, "group": "anchor", "basis": "anchor", "rule": "massmean",
                        "resample": -1, "draw": -1, "k": 1, "k_eff": 1, "units": np.array([0])})
        for j in range(int(cfg["anchors"].get("random_dir", 0))):
            out.append({"task": task, "strategy": strategy, "group": "anchor", "basis": "anchor", "rule": "random_dir",
                        "resample": -1, "draw": j, "k": 1, "k_eff": 1, "units": np.array([j])})
    if cfg["anchors"].get("das"):
        LOG.warning("anchors.das=true but the DAS skyline is not implemented in this build; skipped")
    return out


# ---------------------------------------------------------------------------- generation checks

def _gen_refusal(ctx: Ctx, runner: TaskRunner, idx: np.ndarray, hook, max_new: int) -> np.ndarray:
    subs = ctx.cfg["tasks"]["refusal"]["substrings"]
    toks = [runner.ev.input_ids.iloc[i] for i in idx]
    batches = ctx.batches(toks)
    lab = np.zeros(len(idx), dtype=bool)
    handles = []
    if hook is not None:
        if hook.values_src is not None:
            hook.src_index = torch.as_tensor(runner.src["eval"][idx], dtype=torch.long)
        handles = M.attach_hook(ctx.layers, runner.L, hook, False)
    try:
        for b in batches:
            if hook is not None:
                hook.expected_len, hook.batch_idx = b.S, b.idx
            outs = M.generate_greedy(ctx.model, ctx.tok, b, max_new)
            for i, o in zip(b.idx, outs):
                lab[int(i)] = MET.is_refusal(M.decode(ctx.tok, o), subs)   # text never logged
    finally:
        for h in handles:
            h.remove()
    return lab


# ---------------------------------------------------------------------------- stage

def stage_ablate(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    out_dir = ctx.d("ablate")
    pe = out_dir / "per_example"
    pe.mkdir(exist_ok=True)
    rankings = pd.read_parquet(ctx.run_dir / "select" / "rankings.parquet")
    B = load_bases(ctx)
    gate = ctx.gate
    mode = gate["refusal"]["damage_mode"]
    elab = np.load(ctx.run_dir / "gate" / "eval_refusal_labels.npy")
    checks = {"identity": {}, "fast_path": {}}
    summary = {"tokens": 0, "fwd_seconds": 0.0, "configs": 0, "chunks_skipped": 0}
    t_stage = time.time()
    for task in TASKS:
        bz = load_run_file(ctx.run_dir / "bases" / f"{task}.pt")
        runner = TaskRunner(ctx, task, fast_path=bool(cfg["fast_path"]))
        if runner.fast:
            fp = fast_path_check(ctx, runner, float(cfg["fast_path_tol"]))
            checks["fast_path"][task] = fp
            if not fp["enabled"]:
                runner.drop_fast(f"max |dlogit| {fp['max_abs_logit_diff']:.2e} >= {cfg['fast_path_tol']}")
        else:
            checks["fast_path"][task] = {"enabled": False, "reason": "fast_path: false in config"}
        checks["identity"][task] = identity_check(ctx, runner, B, rankings, float(cfg["identity_tol"]))
        if not all(v["pass"] for v in checks["identity"][task].values()):
            write_json(out_dir / "checks.json", checks)
            raise HarnessInvalid(f"identity check failed for {task}: {checks['identity'][task]}")
        # clean references through the same forward path as the ablated runs
        clean_s, clean_a = runner.eval_scores(None)
        clean_lp = runner.gen_logprobs()
        refused = (runner.labels == 1) & elab if task == "refusal" else np.zeros(len(runner.labels), bool)
        np.savez(pe / f"{task}__clean.npz", score=clean_s, aux=clean_a, labels=runner.labels, refused=refused,
                 eval_pids=runner.ev.pid.to_numpy().astype(str), gen_pids=runner.gen.pid.to_numpy().astype(str))
        configs = enumerate_configs(ctx, task, rankings, B)
        anchors_U = {"massmean": bz["anchors"]["massmean"][None].to(ctx.device),
                     "random_dir": bz["anchors"]["random_dir"].to(ctx.device)}
        chunks = {}
        for c in configs:
            chunks.setdefault((c["strategy"], c["group"]), []).append(c)
        gen_idx = np.flatnonzero(refused)
        for (strategy, group), cl in chunks.items():
            stem = pe / f"{task}__{strategy}__{group}"
            if (stem.with_suffix(".npz")).exists() and (stem.with_suffix(".parquet")).exists():
                summary["chunks_skipped"] += 1
                continue
            tc = time.time()
            n_e, n_g = len(runner.ev), len(runner.gen)
            S = np.zeros((len(cl), n_e), np.float32)
            A = np.zeros((len(cl), n_e), np.float32)
            KL = np.zeros((len(cl), n_g), np.float32)
            REL = np.zeros((len(cl), n_g), np.float32)
            GEN = np.zeros((len(cl), len(gen_idx)), bool) if (task == "refusal" and mode == "generation") else None
            for j, c in enumerate(cl):
                if group == "anchor":
                    basis = BZ.DirectionBasis(anchors_U[c["rule"]])
                    means = bz["means"][f"anchor:{c['rule']}"]
                else:
                    basis = B[group]
                    means = bz["means"][group]
                hook = make_hook(basis, c["units"], strategy, means, runner, runner.T)
                S[j], A[j] = runner.eval_scores(hook)
                KL[j], REL[j] = runner.gen_cost(hook, clean_lp)
                if GEN is not None:
                    GEN[j] = _gen_refusal(ctx, runner, gen_idx, hook, int(cfg["gate"]["eval_label_tokens"]))
            for arr, name in ((S, "score"), (A, "aux"), (KL, "kl"), (REL, "rel")):
                if not np.isfinite(arr).all():
                    raise HarnessInvalid(f"non-finite {name} in chunk {stem.name}")
            tab = pd.DataFrame([{k: v for k, v in c.items() if k != "units"} for c in cl])
            tab["units_head"] = [",".join(map(str, c["units"][:10])) for c in cl]
            arrays = {"score": S, "aux": A, "kl": KL, "rel": REL}
            if GEN is not None:
                arrays["gen_refusal"] = GEN
            np.savez(stem.with_suffix(".npz"), **arrays)
            tab.to_parquet(stem.with_suffix(".parquet"), index=False)
            summary["configs"] += len(cl)
            LOG.info("%s/%s/%s: %d configs in %.1f s (%.0f prompt tokens/s)", task, strategy, group, len(cl),
                     time.time() - tc, runner.tokens / max(runner.fwd_seconds, 1e-9))
        summary["tokens"] += runner.tokens
        summary["fwd_seconds"] += runner.fwd_seconds
        summary.setdefault("fast_path", {})[task] = runner.fast
        if task == "refusal":
            _generation_checks(ctx, runner, B, bz, gen_idx, out_dir)
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(out_dir / "checks.json", checks)
    # results.parquet: one row per configuration
    from .analysis import load_task, results_table

    tabs = [results_table(ctx, load_task(ctx.run_dir, t, ctx.gate)) for t in TASKS]
    res = pd.concat(tabs, ignore_index=True)
    res.to_parquet(out_dir / "results.parquet", index=False)
    summary["throughput_tokens_per_s"] = summary["tokens"] / max(summary["fwd_seconds"], 1e-9)
    summary["stage_seconds"] = time.time() - t_stage
    write_json(out_dir / "throughput.json", summary)
    LOG.info("ablation throughput: %.0f prompt tokens/s over %d tokens (%s)", summary["throughput_tokens_per_s"],
             summary["tokens"], "layer-split fast path" if all(summary.get("fast_path", {}).values()) else "full forward")
    return {"configs": int(len(res)), "throughput_tokens_per_s": summary["throughput_tokens_per_s"]}


def _generation_checks(ctx: Ctx, runner: TaskRunner, B: dict, bz: dict, refused_idx: np.ndarray, out_dir: Path):
    """48 greedy tokens on harmful eval prompts the clean model refuses, for both anchors and each
    basis's best and worst cell (per strategy) at the budget nearest the anchor's cost."""
    from .analysis import load_task, point_cells

    cfg = ctx.cfg
    gc = cfg["generation_check"]
    idx = refused_idx[: int(gc["n_prompts"])]
    if len(idx) == 0:
        LOG.warning("generation check skipped: the clean model refused no harmful eval prompts")
        pd.DataFrame().to_csv(out_dir / "generation_checks.csv", index=False)
        return
    max_new = int(gc["max_new_tokens"])
    rows = []
    clean_rate = float(_gen_refusal(ctx, runner, idx, None, max_new).mean())
    rows.append({"label": "clean", "strategy": "", "basis": "", "rule": "", "k": 0, "n_prompts": len(idx),
                 "refusal_rate_gen": clean_rate, "drop": 0.0})
    td = load_task(ctx.run_dir, "refusal", ctx.gate)
    cells = point_cells(ctx, td)
    rankings = pd.read_parquet(ctx.run_dir / "select" / "rankings.parquet")
    for strategy in cfg["strategies"]:
        todo = [("anchor", "massmean", 0, "anchor:massmean", 1), ("anchor", "random_dir", 0, "anchor:random_dir", 1)]
        for basis_key in B:
            sub = cells[(cells.strategy == strategy) & (cells.basis == basis_key) & (cells.rule != "random")]
            if sub.empty:
                continue
            best = sub.loc[sub.D.idxmax()]
            worst = sub.loc[sub.D.idxmin()]
            for tag, row in (("best", best), ("worst", worst)):
                todo.append((basis_key, row.rule, int(row.k_at_cstar), f"{tag}", 0))
        for basis_key, rule, k, tag, _ in todo:
            if basis_key == "anchor":
                U = bz["anchors"][rule]
                U = U[None] if U.ndim == 1 else U[:1]
                basis = BZ.DirectionBasis(U.to(ctx.device))
                means = bz["means"][f"anchor:{rule}"][:, :1]
                units = np.array([0])
                label = f"anchor:{rule}"
            else:
                basis = B[basis_key]
                means = bz["means"][basis_key]
                r = rankings[(rankings.task == "refusal") & (rankings.basis == basis_key) & (rankings.rule == rule)
                             & (rankings["resample"] == 0)].sort_values("rank")
                units = r.unit.to_numpy().copy()[:min(k, basis.n_units)]
                label = f"{tag}:{basis_key}/{rule}"
            hook = make_hook(basis, units, strategy, means, runner, runner.T)
            rate = float(_gen_refusal(ctx, runner, idx, hook, max_new).mean())
            rows.append({"label": label, "strategy": strategy, "basis": basis_key, "rule": rule, "k": int(len(units)),
                         "n_prompts": len(idx), "refusal_rate_gen": rate, "drop": clean_rate - rate})
            LOG.info("generation check %s %s (k=%d): refusal rate %.3f (clean %.3f)", strategy, label, len(units),
                     rate, clean_rate)
    pd.DataFrame(rows).to_csv(out_dir / "generation_checks.csv", index=False)
