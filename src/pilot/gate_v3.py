"""v3 per-model gate: tokenization (example prefix + query), answer tokens, task baselines and gates, attention-sink
thresholds, and SAE fidelity per layer (main file, fallback / extra file, both input-scaling options, layer-index
check). Writes models/<slug>/gate/gate.json, models/<slug>/data/*.parquet and models/<slug>/prep/norms.json.

Inclusion rules (preregistered, mechanical):
  task gate  AUROC >= 0.85 and coherence >= 0.9 on the eval split (clean model, cached prefix).
  SAE        FVE >= 0.70 and next-token loss increase <= 10% (relative) on task + generic prompts, per (model, layer);
             main file first (Gemma 65k / Qwen TopK 100), then the fallback (Gemma 16k / Qwen TopK 50); the Gemma
             16k file is also judged on its own for the width test (D6).
Input scaling: the smoke test measures every option in `input_norm_options` per file and layer and stores the
better one (higher FVE) in config/v3_sae_input_norm.yaml; dev / confirm read that file (a missing entry is a
harness error). Layer index: the main file's FVE on activations of layers N-1, N, N+1 must peak at N.
"""

from __future__ import annotations

import numpy as np
import torch

from . import metrics as MET
from . import models as M
from . import prompts as P
from . import sae_v3
from . import stats
from .data_v3 import SPLITS, load_demos, load_text, tokenize
from .engine_v3 import MCtx, forward_query, query_lists, scores_all
from .sites import MultiCapture, role_masks
from .utils import LOG, write_json
from .v3common import HarnessError, layer_for, tasks_in

SINGLE = ("truth", "sentiment", "answerability", "toxicity")


def fidelity_depths(cfg: dict, model_id: str) -> list[float]:
    """SAE fidelity covers every configured depth (smoke included); dev restricts to depths_only."""
    ds = [float(d) for d in cfg["models"][model_id]["depths"]]
    only = cfg.get("depths_only")
    return [d for d in ds if not only or any(abs(d - float(o)) < 1e-9 for o in only)]


# ============================================================================= activations

def _capture(mc: MCtx, token_lists, Ls: list[int], fn, prefix_task: str | None = None):
    """Forward (no grad) with layer outputs captured at Ls; fn(batch, {L: h}). With prefix_task the lists are queries
    and the cached prefix is used (h then covers query tokens only)."""
    cap = MultiCapture(mc.layers, Ls)
    pkv = mc.prefix(prefix_task) if prefix_task else None
    try:
        with torch.inference_mode():
            for b in mc.batches(token_lists):
                forward_query(mc, pkv, b)
                fn(b, cap.h)
    finally:
        cap.remove()


def _kept_mask(b, query: bool):
    if query:
        from .engine_v3 import query_masks

        return query_masks(b)[0]
    return role_masks(b)[0]                               # generic: every real token except the first


def _norm_stats(mc: MCtx, Ls: list[int]) -> dict:
    """Median residual norm per layer over generic tokens (not first) and discovery query tokens; the attention-sink
    threshold is sink_norm_multiple x median; E||x||^2 over kept tokens (for rms_d input scaling)."""
    norms = {L: [] for L in Ls}

    def mk(query):
        def fn(b, H):
            m = _kept_mask(b, query)
            for L in Ls:
                norms[L].append(torch.linalg.vector_norm(H[L].float(), dim=-1)[m].cpu())
        return fn
    _capture(mc, list(mc.tokenized("generic", "fit").input_ids), Ls, mk(False))
    for t in tasks_in(mc.cfg):
        _capture(mc, query_lists(mc.tokenized(t, "discovery").iloc[:64]), Ls, mk(True), prefix_task=t)
    mult = float(mc.cfg["sink_norm_multiple"])
    out = {}
    for L in Ls:
        n = torch.cat(norms[L])
        med = float(n.median())
        thr = mult * med
        keep = n <= thr
        out[str(L)] = {"median_norm": med, "sink_threshold": thr, "sink_fraction": float(1 - keep.float().mean()),
                       "mean_sq_norm": float((n[keep] ** 2).mean()), "n_tokens": int(n.numel())}
    return out


def _fidelity_prompts(mc: MCtx) -> list:
    """Full prompts (prefix + query) of every task plus generic prompts, for the SAE loss check."""
    n = int(mc.cfg["sae_inclusion"]["n_prompts"])
    srcs = [list(mc.tokenized("generic", "fit").input_ids)]
    for t in tasks_in(mc.cfg):
        srcs.append(list(mc.tokenized(t, "discovery").input_ids))
    per = max(1, n // len(srcs))
    return [x for s in srcs for x in s[:per]]


def _collect_x(mc: MCtx, Ls: list[int], norms: dict, toks) -> dict:
    """Kept-token activations (not first, not sink) at each layer and its neighbours, on CPU."""
    allL = sorted({x for L in Ls for x in (L - 1, L, L + 1) if 0 <= x < mc.n_layers})
    xs = {L: [] for L in allL}

    def fn(b, H):
        m = role_masks(b)[0]
        keep = m.clone()
        for L in Ls:                                      # sink rule of the hook layer (same tokens at every L)
            keep &= torch.linalg.vector_norm(H[L].float(), dim=-1) <= norms[str(L)]["sink_threshold"]
        for L in allL:
            xs[L].append(H[L].float()[keep].cpu())
    _capture(mc, toks, allL, fn)
    return {L: torch.cat(v) for L, v in xs.items()}


@torch.no_grad()
def _ce(mc: MCtx, toks, L: int | None = None, sae=None, thr: float | None = None) -> float:
    """Mean next-token CE over real tokens; with an SAE, layer-L output is replaced by its reconstruction at kept
    (not first, not sink) tokens."""
    state = {}

    def splice(module, args, output):
        h, rebuild = M._split_output(output)
        keep = state["mask"] & (torch.linalg.vector_norm(h.float(), dim=-1) <= thr)
        rec = sae.decode(sae.encode(h.float()))
        return rebuild(torch.where(keep.unsqueeze(-1), rec.to(h.dtype), h))

    tot, n = 0.0, 0
    with torch.inference_mode():
        for b in mc.batches(toks, "ce"):
            state["mask"] = role_masks(b)[0]
            hh = mc.layers[L].register_forward_hook(splice) if sae is not None else None
            try:
                o = mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                             use_cache=False)
            finally:
                if hh is not None:
                    hh.remove()
            tgt = b.input_ids[:, 1:]
            valid = b.attention_mask[:, 1:].bool() & b.attention_mask[:, :-1].bool()
            lp = torch.log_softmax(o.logits[:, :-1].float(), -1)
            tot += float((-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))[valid].sum())
            n += int(valid.sum())
    return tot / max(n, 1)


def _sae_record(mc: MCtx, sae_cfg: dict, info: dict | None, L: int, norms: dict, X: dict, toks, ce_clean: float,
                modes: list[str]) -> dict:
    """Fidelity of one SAE file at layer L for each input-scaling mode; the chosen mode is the stored one (or, when
    measuring, the higher-FVE one)."""
    if info is None:
        return {"info": None, "included": False, "reason": "no file for this layer"}
    sc = mc.cfg["sae_inclusion"]
    sae = sae_v3.load(sae_cfg, info, mc.run.device, mc.d_model)
    msq = norms[str(L)]["mean_sq_norm"]
    by_mode = {}
    for mode in modes:
        sae_v3.set_input_norm(sae, mode, msq)
        f = sae_v3.fve(sae, X[L], mc.run.device)
        rec = {"fve": f, "l0": sae_v3.l0(sae, X[L], mc.run.device), "norm_factor": sae.norm_factor}
        rec["fve_neighbours"] = {str(x): sae_v3.fve(sae, X[x], mc.run.device) for x in (L - 1, L + 1) if x in X}
        by_mode[mode] = rec
    chosen = max(by_mode, key=lambda m: by_mode[m]["fve"]) if len(modes) > 1 else modes[0]
    sae_v3.set_input_norm(sae, chosen, msq)
    ce = _ce(mc, toks, L, sae, norms[str(L)]["sink_threshold"])
    rel = (ce - ce_clean) / max(ce_clean, 1e-9)
    fve_c = by_mode[chosen]["fve"]
    ok = fve_c >= sc["min_fve"] and rel <= sc["max_rel_ce_increase"]
    nb = by_mode[chosen]["fve_neighbours"]
    index_ok = all(fve_c >= v for v in nb.values())
    del sae
    return {"info": info, "input_norm": chosen, "mean_sq_norm": msq, "by_mode": by_mode, "fve": fve_c,
            "l0": by_mode[chosen]["l0"], "ce_clean": ce_clean, "ce_sae": ce, "ce_rel_increase": rel,
            "included": bool(ok), "layer_index_ok": bool(index_ok),
            "reason": None if ok else f"FVE {fve_c:.3f} (min {sc['min_fve']}), loss +{100 * rel:.1f}% "
                                      f"(max {100 * sc['max_rel_ce_increase']:.0f}%)"}


# ============================================================================= stage

def stage_gate(mc: MCtx) -> dict:
    cfg = mc.cfg
    run = mc.run
    tok = mc.tok
    model = mc.model
    tc = M.text_config(model)
    Ls = {str(d): mc.layer(d) for d in fidelity_depths(cfg, mc.model_id)}
    minfo = {"weights": str(mc.weights), "n_layers": tc.num_hidden_layers, "d_model": tc.hidden_size,
             "params": M.count_params(model), "dtype": str(run.dtype), "device": str(run.device),
             "attn_implementation": getattr(model.config, "_attn_implementation", None),
             "final_logit_softcapping": getattr(tc, "final_logit_softcapping", None),
             "attn_logit_softcapping": getattr(tc, "attn_logit_softcapping", None),
             "layers_by_depth": Ls, "batching": mc.tuning}
    run.record(**{f"model:{mc.model_id}": minfo})
    out = {"model": mc.model_id, "model_info": minfo, "tasks": {}, "sae": {}}
    LOG.info("%s: %d layers, d_model %d, attention %s; depths -> layers %s", mc.model_id, tc.num_hidden_layers,
             tc.hidden_size, minfo["attn_implementation"], Ls)

    # ------------------------------------------------------------ tokenize every split for this model
    ddir = mc.dir / "data"
    ddir.mkdir(parents=True, exist_ok=True)
    demos = load_demos(run.run_dir / "data")
    tasks = tasks_in(cfg)
    for t in tasks + ["generic"]:
        for split in SPLITS[t]:
            p = run.run_dir / "data" / f"{t}_{split}.parquet"
            if p.exists():
                tokenize(tok, load_text(run.run_dir / "data", t, split),
                         None if t == "generic" else demos[t]["prefix"]).to_parquet(ddir / f"{t}_{split}.parquet",
                                                                                    index=False)
    bos = tok.bos_token_id
    first = mc.tokenized(tasks[0], "discovery").input_ids.iloc[0][0]
    out["prefix"] = {t: {"n_demo": int(mc.tokenized(t, "discovery").n_demo.iloc[0]), "starts_with_bos": bool(
        mc.tokenized(t, "discovery").input_ids.iloc[0][0] == bos) if bos is not None else None} for t in tasks}
    if "gemma" in mc.model_id.lower() and bos is not None and first != bos:
        raise HarnessError(f"{mc.model_id}: prompts must start with <bos> (got token {first})")

    # ------------------------------------------------------------ answer tokens and task gates
    es = cfg.get("eval_split", "eval")
    for t in tasks:
        g = cfg["gates"]["single_token"]
        rec = {}
        if t in SINGLE:
            words = cfg["tasks"][t]["answers"]
            if not all(isinstance(w, str) for w in words):
                raise HarnessError(f"{t}: answer words must be quoted strings in config/v3.yaml")
            variants = P.answer_variants(tok, words)
            for k, v in variants.items():
                if not v["distinct"]:
                    raise HarnessError(f"{t}: answer tokens identical ({k}: {v['tokens']})")
            gid = torch.tensor(variants["bare"]["ids"] + variants["space"]["ids"], device=run.device)
            disc = mc.tokenized(t, "discovery").iloc[:128]
            pkv = mc.prefix(t)
            lps = []
            with torch.inference_mode():
                for b in mc.batches(query_lists(disc)):
                    lps.append(torch.log_softmax(forward_query(mc, pkv, b)[:, -1, :].float(), -1)[:, gid].cpu())
            lp = torch.cat(lps).numpy()
            mass = {"bare": float(np.mean(np.exp(lp[:, 0]) + np.exp(lp[:, 1]))),
                    "space": float(np.mean(np.exp(lp[:, 2]) + np.exp(lp[:, 3])))}
            chosen = "bare" if mass["bare"] >= mass["space"] else "space"
            rec.update(answer_variant=chosen, answer_ids=variants[chosen]["ids"],
                       answer_tokens=variants[chosen]["tokens"], mass_by_variant=mass)
            out["tasks"][t] = rec                         # score_fn reads answer_ids from the gate record
            (mc.dir / "gate").mkdir(parents=True, exist_ok=True)
            write_json(mc.dir / "gate" / "gate.json", out)
        else:
            rec.update(answer_ids="per_example (leading-space first names)")
            out["tasks"][t] = rec
        ev = mc.tokenized(t, es)
        m, coh = scores_all(mc, t, es, use_prefix=True, df=ev)
        y = ev.label.to_numpy()
        au = stats.auroc(m, y)
        base = {"auroc": au, "accuracy_at_0": float(np.mean((m > 0) == y)), "coherence": float(np.mean(coh)),
                "n": int(len(y)), "n_pos": int(y.sum()), "split": es}
        reasons = []
        if not (au >= g["min_auroc"]):
            reasons.append(f"AUROC {au:.3f} < {g['min_auroc']}")
        if not (base["coherence"] >= g["min_coherence"]):
            reasons.append(f"coherence {base['coherence']:.3f} < {g['min_coherence']}")
        rec.update(baseline=base, gate={"pass": not reasons, "reasons": reasons})
        out["tasks"][t] = rec
        LOG.info("%s/%s: answers %s; clean AUROC %.3f, coherence %.3f -> gate %s", mc.slug, t,
                 rec.get("answer_tokens", "per-example names"), au, base["coherence"],
                 "PASS" if not reasons else f"FAIL {reasons}")
    write_json(mc.dir / "gate" / "gate.json", out)

    # ------------------------------------------------------------ sink thresholds, then SAE per layer
    all_L = sorted(set(Ls.values()))
    need_L = sorted({x for L in all_L for x in (L - 1, L, L + 1) if 0 <= x < mc.n_layers})
    norms = _norm_stats(mc, need_L)
    (mc.dir / "prep").mkdir(parents=True, exist_ok=True)
    write_json(mc.dir / "prep" / "norms.json", norms)
    LOG.info("%s sink thresholds (10x median norm): %s", mc.slug,
             {L: round(norms[str(L)]["sink_threshold"], 1) for L in all_L})
    sae_cfg = cfg["models"][mc.model_id]["sae"]
    resolved = sae_v3.resolve(sae_cfg, all_L)
    toks = _fidelity_prompts(mc)
    X = _collect_x(mc, all_L, norms, toks)
    ce_clean = _ce(mc, toks)
    measuring = cfg["phase"] == "smoke"
    stored = (cfg.get("sae_input_norm") or {}).get(mc.model_id, {})
    for L in all_L:
        rec = {"files": {}}
        for slot in ("main", "extra"):
            info = resolved[L][slot]
            if measuring:
                modes = list(cfg["input_norm_options"])
            else:
                key = f"{L}:{slot}"
                if info is not None and key not in stored:
                    raise HarnessError(f"{mc.model_id} L{L} {slot}: no measured input scaling in "
                                       f"{cfg['sae_input_norm_file']} (run --v3 --smoke first)")
                modes = [stored.get(key, "none")]
            rec["files"][slot] = _sae_record(mc, sae_cfg, info, L, norms, X, toks, ce_clean, modes)
        m_ok, x_ok = rec["files"]["main"]["included"], rec["files"]["extra"]["included"]
        rec["used"] = "main" if m_ok else ("extra" if x_ok else None)
        rec["extra_included"] = bool(x_ok and m_ok and sae_cfg["format"] == "gemmascope")   # width test needs both
        rec["included"] = rec["used"] is not None
        out["sae"][str(L)] = rec
        fm = rec["files"]["main"]
        LOG.info("%s SAE L%d main %s: FVE %.3f (%s), loss %+.1f%%, L0 %.1f, layer-index %s -> %s", mc.slug, L,
                 (fm.get("info") or {}).get("file"), fm.get("fve", float("nan")), fm.get("input_norm"),
                 100 * fm.get("ce_rel_increase", float("nan")), fm.get("l0", float("nan")),
                 "ok" if fm.get("layer_index_ok") else "CHECK", f"basis uses {rec['used']}")
    write_json(mc.dir / "gate" / "gate.json", out)
    run.record(**{f"sae:{mc.model_id}": {L: {s: {"file": (v["files"][s].get("info") or {}).get("file"),
                                                  "included": v["files"][s].get("included"),
                                                  "input_norm": v["files"][s].get("input_norm"),
                                                  "fve": v["files"][s].get("fve")} for s in ("main", "extra")}
                                          | {"used": v["used"]} for L, v in out["sae"].items()}})
    return {"tasks": {t: v["gate"]["pass"] for t, v in out["tasks"].items()},
            "sae_used": {L: v["used"] for L, v in out["sae"].items()}}


def measured_input_norm(gates: dict) -> dict:
    """{model: {"L:slot": mode}} from smoke gate records (written to config/v3_sae_input_norm.yaml)."""
    out = {}
    for m, g in gates.items():
        for L, v in g["sae"].items():
            for slot, r in v["files"].items():
                if r.get("info") is not None:
                    out.setdefault(m, {})[f"{L}:{slot}"] = r["input_norm"]
    return out


# ============================================================================= run-level inclusion

def inclusion(cfg: dict, gates: dict) -> dict:
    from .v3common import resolve_reference_task

    enforced = bool(cfg.get("gates_enforced", True))
    out = {"enforced": enforced, "baselines": {}, "sae": {}, "excluded": []}
    for m, g in gates.items():
        out["baselines"][m] = {t: {"baseline": v.get("baseline"), "gate": v["gate"]} for t, v in g["tasks"].items()}
        out["sae"][m] = {L: {"used": v["used"], "extra_included": v.get("extra_included"),
                             "main": {k: v["files"]["main"].get(k) for k in ("fve", "ce_rel_increase", "l0", "input_norm",
                                                                             "included", "layer_index_ok", "reason")},
                             "extra": {k: v["files"]["extra"].get(k) for k in ("fve", "ce_rel_increase", "l0",
                                                                               "input_norm", "included", "reason")}}
                         for L, v in g["sae"].items()}
        for t, v in g["tasks"].items():
            if not v["gate"]["pass"]:
                out["excluded"].append({"model": m, "task": t, "reasons": v["gate"]["reasons"], "enforced": enforced})
    rt = resolve_reference_task(cfg, gates)
    out["reference_task"] = rt["task"]
    out["reference_task_changed"] = rt["changed"]
    gm = cfg["reference"]["model"]
    n_ok = sum(1 for t, v in gates.get(gm, {"tasks": {}})["tasks"].items() if v["gate"]["pass"] or not enforced)
    out["n_tasks_reference_model"] = n_ok
    out["inconclusive"] = (rt["task"] is None) or (n_ok < int(cfg["gates"]["min_tasks_gemma"]) and
                                                   cfg["phase"] != "dev")
    return out
