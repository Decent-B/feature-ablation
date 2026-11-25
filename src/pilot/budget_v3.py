"""v3 batch tuning, throughput and run-time projection (smoke), against the frozen budget.

Tuning (per model, smoke run): the largest forward batch (query tokens, cached prefix) and backward batch whose peak
memory stays under the cap, with measured tokens/s. Results go to runs/v3/tuning.json, which --dev / --confirm read.
Projection: seconds per intervention per (model, task) from the measured forward throughput, the full-run eval set
sizes and mean query lengths, then per block in priority order against `budget.total_hours`.
"""

from __future__ import annotations

import resource
import time

import numpy as np
import torch

from . import models as M
from .utils import LOG, fmt_secs, read_json, write_json
from .v3common import TUNING3_PATH, load_v3_config, ref


def _peak_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _reset():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tune_model(mc, task: str) -> dict:
    """Forward / backward batch sizes on `task` queries with the cached prefix."""
    from .engine_v3 import forward_query, query_lists
    from .sites import GradCapture

    cfg = mc.cfg
    bc = cfg["batching"]
    cap = float(bc["memory_cap_gb"])
    qs = query_lists(mc.tokenized(task, "eval"))
    L = int(np.median([len(q) for q in qs]))
    pkv = mc.prefix(task)
    out = {"device": str(mc.run.device), "memory_cap_gb": cap, "tuned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "task": task, "prefix_tokens": pkv.p, "forward": [], "backward": []}

    def batch(tokens):
        n = max(1, tokens // max(L, 1))
        reps = [qs[i % len(qs)] for i in range(n)]
        return M.make_batches(reps, mc.tok.pad_token_id, mc.run.device, 10**6, 10**9)[0]

    best = None
    for tok_n in bc["tune_candidates_tokens"]:
        b = batch(int(tok_n))
        _reset()
        try:
            with torch.inference_mode():
                forward_query(mc, pkv, b)
                _sync()
                t0 = time.time()
                forward_query(mc, pkv, b)
                _sync()
                dt = time.time() - t0
        except torch.OutOfMemoryError:
            out["forward"].append({"tokens": int(tok_n), "oom": True})
            break
        peak = _peak_gb()
        rec = {"tokens": int(b.n_tokens), "batch": int(len(b.idx)), "seconds": dt, "tokens_per_s": b.n_tokens / dt,
               "peak_gb": peak}
        out["forward"].append(rec)
        if peak > 0.9 * cap:
            break
        best = rec
    out["max_batch_tokens"] = int(max(best["tokens"], 512)) if best else int(bc["max_batch_tokens"])
    out["fwd_tokens_per_s"] = best["tokens_per_s"] if best else None
    Ls = sorted({mc.layer(d) for d in mc.depths()})
    bestg = None
    for tok_n in [x // 4 for x in bc["tune_candidates_tokens"]]:
        b = batch(int(tok_n))
        _reset()
        try:
            gc = GradCapture(mc.layers, Ls)
            try:
                t0 = time.time()
                with torch.enable_grad():
                    forward_query(mc, pkv, b)[:, -1, 0].float().sum().backward()
                _sync()
                dt = time.time() - t0
            finally:
                gc.remove()
        except torch.OutOfMemoryError:
            out["backward"].append({"tokens": int(tok_n), "oom": True})
            break
        peak = _peak_gb()
        rec = {"tokens": int(b.n_tokens), "seconds": dt, "tokens_per_s": b.n_tokens / dt, "peak_gb": peak}
        out["backward"].append(rec)
        if peak > 0.9 * cap:
            break
        bestg = rec
    out["grad_batch_tokens"] = int(max(bestg["tokens"], 256)) if bestg else int(bc["grad_batch_tokens"])
    out["grad_tokens_per_s"] = bestg["tokens_per_s"] if bestg else None
    LOG.info("%s tuned: forward %d query tokens/batch (%.0f tok/s), backward %d tokens/batch", mc.slug,
             out["max_batch_tokens"], out["fwd_tokens_per_s"] or 0, out["grad_batch_tokens"])
    return out


def save_tuning(key: str, rec: dict):
    allt = read_json(TUNING3_PATH) if TUNING3_PATH.exists() else {}
    allt[key] = rec
    TUNING3_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(TUNING3_PATH, allt)


def _sizes(cfg: dict) -> dict:
    d = cfg["data"]
    return {"truth": int(d["truth"]["n_eval"]), "sentiment": int(d["sentiment"]["n_eval"]),
            "answerability": int(d["answerability"]["n_eval"]), "toxicity": int(d["toxicity"]["n_eval"]),
            "ioi": int(cfg["ioi"]["n_eval"])}


def projected_seconds(cfg: dict, model: str, task: str, dev_tiny: bool = False) -> float | None:
    """Seconds per intervention at full-run sizes: eval queries + cost prompts at the measured forward rate (the fast
    path runs about half the layers on average), plus the per-intervention overhead."""
    if not TUNING3_PATH.exists():
        return None
    t = read_json(TUNING3_PATH).get(("dev_tiny:" if dev_tiny else "") + model)
    if not t or not t.get("fwd_tokens_per_s") or not t.get("mean_query_tokens"):
        return None
    full = load_v3_config("confirm", dev_tiny=dev_tiny)
    n_eval = _sizes(full).get(task, 500)
    q = float(t["mean_query_tokens"].get(task, np.mean(list(t["mean_query_tokens"].values()))))
    cost_tokens = int(full["cost"]["n_generic"]) * (float(t.get("mean_generic_tokens", 40)) + int(full["cost"]["continuation_tokens"]))
    tokens = n_eval * q * 1.1 + cost_tokens          # +10%: attention to the cached prefix
    return 0.55 * tokens / float(t["fwd_tokens_per_s"]) + float(full["budget"]["overhead_s_per_intervention"])


def project(dev_tiny: bool) -> dict:
    """Projected dev and confirm times per block (full-run sizes) from the tuning file."""
    from .ablate_v3 import SPEC_SECONDS

    full = load_v3_config("confirm", dev_tiny=dev_tiny)
    r = ref(full)
    c = full["cell"]
    per_cell = 3 * (3 * int(c["discovery_resamples"]) + int(c["random_draws"])) * len(c["budgets"]) + 1 + int(c["random_dir"])
    tasks = ["truth", "sentiment", "answerability", "toxicity", "ioi"]

    def sec(m, t):
        return projected_seconds(full, m, t, dev_tiny) or SPEC_SECONDS.get((m, t), 5.0)

    M_ = r["model"]
    rt = r["task"]
    other_models = [m for m in full["models"] if m != M_]
    depths = [d for d in full["models"][M_]["depths"] if abs(float(d) - float(r["depth"])) > 1e-9]
    x_curves = (3 * int(c["discovery_resamples"]) + int(c["random_draws"])) * len(c["budgets"])
    rg = full["controls"]["readout_groups"]
    blocks = [("B1_reference", per_cell * sec(M_, rt)),
              ("B2_depth", len(depths) * per_cell * sec(M_, rt)),
              ("B3_task", sum(per_cell * sec(M_, t) for t in tasks if t != rt)),
              ("B4_model", sum(per_cell * sec(m, rt) for m in other_models)),
              ("B5_sae_width", x_curves * sec(M_, rt)),
              ("B6_resample", per_cell * sec(M_, rt)),
              ("B7_controls", sum(3 * len(tasks) * len(full["controls"]["cross_task_k"]) * sec(M_, t) for t in tasks)
               + 3 * int(rg["n_bands"]) * int(rg["groups_per_band"]) * sec(M_, rt))]
    dev = 2 * per_cell * (sec(M_, "truth") + sec(M_, "ioi")) * 0.4 / 2
    budget = float(full["budget"]["total_hours"]) * 3600
    prep = 0.35 * 3600 if not dev_tiny else 60.0
    rows, cum = [], dev + prep
    for b, s in blocks:
        cum += s
        rows.append({"block": b, "seconds": s, "cumulative": cum, "fits": cum <= budget})
    return {"dev_seconds": dev, "prep_seconds": prep, "blocks": rows, "budget_seconds": budget,
            "total_seconds": cum, "not_fitting": [r_["block"] for r_ in rows if not r_["fits"]]}


def projection_md(p: dict) -> list[str]:
    L = ["| Step | Projected | Cumulative (dev + confirm) | Fits the budget |", "|---|---|---|---|",
         f"| dev run | {fmt_secs(p['dev_seconds'])} | {fmt_secs(p['dev_seconds'])} | |",
         f"| confirm prep | {fmt_secs(p['prep_seconds'])} | {fmt_secs(p['dev_seconds'] + p['prep_seconds'])} | |"]
    for r in p["blocks"]:
        L.append(f"| {r['block']} | {fmt_secs(r['seconds'])} | {fmt_secs(r['cumulative'])} | "
                 f"{'yes' if r['fits'] else '**NO**'} |")
    L += ["", f"Total {fmt_secs(p['total_seconds'])} against a budget of {fmt_secs(p['budget_seconds'])}."
          + (f" Blocks that would not fit: {', '.join(p['not_fitting'])}." if p["not_fitting"] else "")]
    return L
