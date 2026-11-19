"""v2 per-model gate: tokenization, format check, answer tokens, task baselines and inclusion gates, refusal
labels and eval set, attention-sink thresholds and SAE fidelity per depth.

Writes models/<slug>/gate/gate.json, models/<slug>/data/*.parquet and models/<slug>/prep/norms.json.
Inclusion rules (preregistered, mechanical):
  task gate   single-token: AUROC >= 0.85 and coherence >= 0.9 on the eval split; refusal: >= 150 harmful eval
              prompts refused (generation-labelled) and harmless refusal <= 0.10.
  SAE         FVE >= 0.70 and next-token loss increase <= 10% (relative) on chat prompts, per (model, depth).
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import torch

from . import metrics as MET
from . import models as M
from . import prompts as P
from . import sae_v2
from . import stats
from .data_v2 import load_text, tokenize
from .engine_v2 import MCtx, generate, last_logits_all, refusal_labels
from .sites import MultiCapture
from .utils import LOG, rng_for, write_json
from .v2common import HarnessError, tasks_in

SINGLE = ("truth", "sentiment", "answerability", "toxicity")
SPLITS = {"truth": ("discovery", "eval"), "sentiment": ("discovery", "eval"), "answerability": ("discovery", "eval"),
          "toxicity": ("discovery", "eval"), "refusal": ("discovery", "eval_harmful_pool", "eval_harmless"),
          "generic": ("fit", "cost")}


def gated_deltanet_report(model) -> dict:
    """Which Gated DeltaNet implementation loaded (hybrid Qwen3.5 only)."""
    tc = M.text_config(model)
    if "linear_attention" not in (getattr(tc, "layer_types", None) or []):
        return {"hybrid": False}
    rep = {"hybrid": True, "n_linear_layers": sum(t == "linear_attention" for t in tc.layer_types)}
    for pkg in ("fla", "causal_conv1d", "kernels"):
        try:
            importlib.import_module(pkg)
            rep[pkg] = "importable"
        except Exception as e:  # noqa: BLE001
            rep[pkg] = f"missing ({type(e).__name__})"
    rep["implementation"] = ("fla Triton kernels (chunk/fused gated delta rule)" if rep["fla"] == "importable"
                             else "pure-PyTorch fallback (torch_chunk_gated_delta_rule)")
    rep["conv1d"] = "causal_conv1d CUDA kernel" if rep["causal_conv1d"] == "importable" else "PyTorch conv1d fallback"
    rep["fused_kernels"] = rep["fla"] == "importable"
    return rep


# ============================================================================= helpers

def _capture_layers(mc: MCtx, token_lists, Ls: list[int], fn_per_batch):
    """Forward (no grad) with layer outputs captured at Ls; fn_per_batch(batch, {L: h}, logits)."""
    cap = MultiCapture(mc.layers, Ls)
    try:
        with torch.inference_mode():
            for b in mc.batches(token_lists):
                out = mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                               use_cache=False, logits_to_keep=1)
                fn_per_batch(b, cap.h, out.logits)
    finally:
        cap.remove()


def _nonfirst_mask(b) -> torch.Tensor:
    from .sites import role_masks

    return role_masks(b)[0]


def _norm_stats(mc: MCtx, Ls: list[int]) -> dict:
    """Median residual norm per layer over non-first real tokens of generic + discovery prompts; the
    attention-sink threshold is sink_norm_multiple x median. Also E||x||^2 over kept tokens (SAE input scaling, if configured)."""
    toks = list(mc.tokenized("generic", "fit").input_ids)
    for t in tasks_in(mc.cfg):
        d = mc.tokenized(t, "discovery")
        toks += list(d.input_ids[: 64])
    norms = {L: [] for L in Ls}
    sq = {L: [] for L in Ls}

    def fn(b, H, logits):
        m = _nonfirst_mask(b)
        for L in Ls:
            n = torch.linalg.vector_norm(H[L].float(), dim=-1)
            norms[L].append(n[m].cpu())
    _capture_layers(mc, toks, Ls, fn)
    mult = float(mc.cfg["sink_norm_multiple"])
    out = {}
    for L in Ls:
        n = torch.cat(norms[L])
        med = float(n.median())
        thr = mult * med
        keep = n <= thr
        out[str(L)] = {"median_norm": med, "sink_threshold": thr, "sink_fraction_nonfirst": float(1 - keep.float().mean()),
                       "mean_sq_norm": float((n[keep] ** 2).mean()), "n_tokens": int(n.numel())}
    return out


def _excluded_counts(mc: MCtx, token_lists, L: int, thr: float) -> dict:
    """Excluded tokens per prompt (the first token + attention sinks) at layer L."""
    counts = np.zeros(len(token_lists), dtype=np.int64)

    def fn(b, H, logits):
        m = _nonfirst_mask(b)
        n = torch.linalg.vector_norm(H[L].float(), dim=-1)
        c = (m & (n > thr)).sum(-1).cpu().numpy() + 1
        counts[b.idx] = c
    _capture_layers(mc, token_lists, [L], fn)
    return {"mean": float(counts.mean()), "max": int(counts.max()), "frac_prompts_with_sink": float((counts > 1).mean()),
            "n_prompts": int(len(counts))}


# ============================================================================= SAE fidelity

def _fidelity_prompts(mc: MCtx) -> list:
    n = int(mc.cfg["sae_inclusion"]["n_prompts"])
    srcs = [mc.tokenized("generic", "fit")]
    for t in ("truth", "sentiment", "answerability"):
        if t in tasks_in(mc.cfg):
            srcs.append(mc.tokenized(t, "discovery"))
    if "refusal" in tasks_in(mc.cfg):
        d = mc.tokenized("refusal", "discovery")
        srcs.append(d[d.label == 0])                     # harmless only
    per = max(1, n // len(srcs))
    toks = []
    for s in srcs:
        toks += list(s.input_ids[:per])
    return toks


def _collect_x(mc: MCtx, Ls: list[int], norms: dict, toks) -> dict:
    """Kept-token activations (not first, not sink) at each layer, on CPU."""
    xs = {L: [] for L in Ls}

    def fn(b, H, logits):
        m = _nonfirst_mask(b)
        for L in Ls:
            h = H[L].float()
            keep = m & (torch.linalg.vector_norm(h, dim=-1) <= norms[str(L)]["sink_threshold"])
            xs[L].append(h[keep].cpu())
    _capture_layers(mc, toks, Ls, fn)
    return {L: torch.cat(v) for L, v in xs.items()}


@torch.no_grad()
def _fve(sae, X: torch.Tensor, device) -> float:
    Xh = torch.cat([sae.decode(sae.encode(X[i:i + 4096].to(device))).cpu() for i in range(0, X.shape[0], 4096)])
    return float(1 - (X - Xh).var(0).sum() / X.var(0).sum())


@torch.no_grad()
def _sae_eval(mc: MCtx, sae, L: int, thr: float, toks, X: torch.Tensor) -> dict:
    """FVE over kept tokens (dictionary_learning definition) and next-token CE with x-hat spliced in at kept tokens."""
    layers = mc.layers
    ce = [0.0, 0.0]
    n_tok = 0
    state = {}

    def splice(module, args, output):
        h, rebuild = M._split_output(output)
        keep = state["mask"] & (torch.linalg.vector_norm(h.float(), dim=-1) <= thr)
        rec = sae.decode(sae.encode(h.float()))
        return rebuild(torch.where(keep.unsqueeze(-1), rec.to(h.dtype), h))

    with torch.inference_mode():
        for b in mc.batches(toks, "ce"):
            state["mask"] = _nonfirst_mask(b)
            outs = [mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                             use_cache=False)]
            hh = layers[L].register_forward_hook(splice)
            try:
                outs.append(mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask,
                                     position_ids=b.position_ids, use_cache=False))
            finally:
                hh.remove()
            tgt = b.input_ids[:, 1:]
            valid = b.attention_mask[:, 1:].bool() & b.attention_mask[:, :-1].bool()
            for j, o in enumerate(outs):
                lp = torch.log_softmax(o.logits[:, :-1].float(), -1)
                ce[j] += float((-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))[valid].sum())
            n_tok += int(valid.sum())
    c0, c1 = ce[0] / max(n_tok, 1), ce[1] / max(n_tok, 1)
    return {"fve": _fve(sae, X, mc.run.device), "n_tokens": int(X.shape[0]),
            "l0": float((sae.encode(X[:2048].to(mc.run.device)) > 0).float().sum(-1).mean()),
            "ce_clean": c0, "ce_sae": c1, "ce_increase": c1 - c0, "ce_rel_increase": (c1 - c0) / max(c0, 1e-9)}


# ============================================================================= stage

def stage_gate(mc: MCtx) -> dict:
    cfg = mc.cfg
    run = mc.run
    relaxed = run.dev_tiny
    model, tok = mc.model, mc.tok
    tc = M.text_config(model)
    lin = gated_deltanet_report(model)
    if lin.get("hybrid"):
        LOG.info("%s Gated DeltaNet implementation: %s; conv1d: %s", mc.model_id, lin["implementation"], lin["conv1d"])
        if not lin["fused_kernels"]:
            LOG.warning("%s uses the pure-PyTorch Gated DeltaNet fallback: expect a slower run (projection measures it)",
                        mc.model_id)
    Ls = {str(d): mc.layer(d) for d in mc.depths()}
    minfo = {"weights": str(mc.weights), "n_layers": tc.num_hidden_layers, "d_model": tc.hidden_size,
             "params": M.count_params(model), "dtype": str(run.dtype), "device": str(run.device),
             "gated_deltanet": lin, "layers_by_depth": Ls, "batching": mc.tuning}
    if not relaxed:
        from .pipeline import _model_sha

        minfo["model_sha"] = _model_sha(mc.model_id)
    run.record(**{f"model:{mc.model_id}": minfo})
    out = {"model": mc.model_id, "model_info": minfo, "tasks": {}, "sae": {}, "relaxed_dev_tiny": relaxed}
    LOG.info("%s: %d layers, d_model %d; depths -> layers %s", mc.model_id, tc.num_hidden_layers, tc.hidden_size, Ls)

    # ------------------------------------------------------------ tokenize every split for this model
    ddir = mc.dir / "data"
    ddir.mkdir(parents=True, exist_ok=True)
    tasks = tasks_in(cfg)
    for t in tasks + ["generic"]:
        for split in SPLITS[t]:
            tokenize(tok, load_text(run.run_dir / "data", t, split)).to_parquet(ddir / f"{t}_{split}.parquet", index=False)
    suffix = P.template_suffix(tok)
    out["template_suffix"] = {"text": suffix["suffix_text"], "tokens": suffix["suffix_tokens"],
                              "has_empty_think_block": suffix["has_empty_think_block"]}

    # ------------------------------------------------------------ format check (thinking mode)
    gc = cfg["gates"]
    fmt = {}
    for t in tasks:
        d = mc.tokenized(t, "discovery")
        if t == "refusal":
            d = d[d.label == 0]                                  # harmless only: generations may be logged
        gens, _ = generate(mc, list(d.input_ids[: int(gc["format_n"])]), int(gc["format_tokens"]))
        txt = [M.decode(tok, g) for g in gens]
        think = [P.starts_with_thinking(x) for x in txt]
        words = [str(w).lower() for w in cfg["tasks"][t].get("answers", [])]
        starts = [any(x.strip().lower().startswith(w) for w in words) if words else bool(x.strip()) for x in txt]
        fmt[t] = {"n": len(txt), "thinking_block": int(sum(think)), "starts_with_answer": int(sum(starts)),
                  "responses": txt if t in SINGLE else None}
        if any(think) and not relaxed:
            write_json(mc.dir / "gate" / "gate.json", out)
            raise HarnessError(f"{mc.model_id}/{t}: responses start with a thinking block (enable_thinking=False ignored?)")
    out["format_check"] = fmt
    LOG.info("format check %s: %s", mc.slug, {t: f"{v['starts_with_answer']}/{v['n']} answer-first, "
                                                 f"{v['thinking_block']} thinking" for t, v in fmt.items()})

    # ------------------------------------------------------------ single-token tasks
    for t in [x for x in tasks if x in SINGLE]:
        words = cfg["tasks"][t]["answers"]
        if not all(isinstance(w, str) for w in words):
            raise HarnessError(f"{t}: answer words must be quoted strings in config/v2.yaml (YAML reads Yes/True as booleans)")
        variants = P.answer_variants(tok, words)
        for k, v in variants.items():
            if not v["distinct"]:
                raise HarnessError(f"{t}: answer tokens identical ({k}: {v['tokens']})")
        gid = torch.tensor(variants["bare"]["ids"] + variants["space"]["ids"], device=run.device)
        disc = mc.tokenized(t, "discovery")
        (lp,) = last_logits_all(mc, list(disc.input_ids[:256]), lambda lg: torch.log_softmax(lg, -1)[:, gid])
        mass = {"bare": float(np.mean(np.exp(lp[:, 0]) + np.exp(lp[:, 1]))),
                "space": float(np.mean(np.exp(lp[:, 2]) + np.exp(lp[:, 3])))}
        chosen = "bare" if mass["bare"] >= mass["space"] else "space"
        ids = variants[chosen]["ids"]
        ev = mc.tokenized(t, "eval")
        m, coh = last_logits_all(mc, list(ev.input_ids), lambda lg: MET.truth_scores(lg, ids))
        y = ev.label.to_numpy()
        au = stats.auroc(m, y)
        base = {"auroc": au, "accuracy_at_0": float(np.mean((m > 0) == y)), "coherence": float(np.mean(coh)),
                "n": int(len(y)), "n_pos": int(y.sum())}
        g = gc["single_token"]
        reasons = []
        if not (au >= g["min_auroc"]):
            reasons.append(f"AUROC {au:.3f} < {g['min_auroc']}")
        if not (base["coherence"] >= g["min_coherence"]):
            reasons.append(f"coherence {base['coherence']:.3f} < {g['min_coherence']}")
        out["tasks"][t] = {"answer_variant": chosen, "answer_ids": ids, "answer_tokens": variants[chosen]["tokens"],
                           "mass_by_variant": mass, "baseline": base, "gate": {"pass": not reasons, "reasons": reasons}}
        LOG.info("%s/%s: answers %s (%s); clean AUROC %.3f, coherence %.3f -> gate %s", mc.slug, t,
                 variants[chosen]["tokens"], chosen, au, base["coherence"], "PASS" if not reasons else f"FAIL {reasons}")

    # ------------------------------------------------------------ refusal
    if "refusal" in tasks:
        out["tasks"]["refusal"] = _refusal_gate(mc, relaxed)

    # ------------------------------------------------------------ sink thresholds, then SAE per depth
    all_L = sorted(set(Ls.values()))
    norms = _norm_stats(mc, all_L)
    (mc.dir / "prep").mkdir(parents=True, exist_ok=True)
    for L in all_L:
        ex = _excluded_counts(mc, list(mc.tokenized("generic", "fit").input_ids), L, norms[str(L)]["sink_threshold"])
        norms[str(L)]["excluded_per_prompt_generic"] = ex
    write_json(mc.dir / "prep" / "norms.json", norms)
    LOG.info("%s sink thresholds (10x median norm): %s", mc.slug,
             {L: round(v["sink_threshold"], 1) for L, v in norms.items() if int(L) in all_L})
    sae_cfg = cfg["models"][mc.model_id]["sae"]
    resolved = sae_v2.resolve(sae_cfg, mc.model_id, all_L)
    toks = _fidelity_prompts(mc)
    sc = cfg["sae_inclusion"]
    Xs = _collect_x(mc, all_L, norms, toks)
    for L in all_L:
        info = resolved[L]
        rec = {"info": info, "input_norm": sae_cfg.get("input_norm", "none"), "scale_stats": norms[str(L)]}
        if info is None:
            rec.update(included=False, reason=f"no SAE file for layer {L} in {sae_cfg.get('repo', 'local')}")
            out["sae"][str(L)] = rec
            LOG.warning("%s L%d: %s", mc.slug, L, rec["reason"])
            continue
        sae = sae_v2.load(sae_cfg, info, run.device)
        sae_v2.set_input_norm(sae, rec["input_norm"], norms[str(L)])
        if sae.d != tc.hidden_size:
            raise HarnessError(f"SAE d={sae.d} does not match d_model={tc.hidden_size} ({info['file']})")
        fid = _sae_eval(mc, sae, L, norms[str(L)]["sink_threshold"], toks, Xs[L])
        ok = fid["fve"] >= sc["min_fve"] and fid["ce_rel_increase"] <= sc["max_rel_ce_increase"]
        rec.update(fidelity=fid, included=bool(ok), thresholds=sc,
                   reason=None if ok else f"FVE {fid['fve']:.3f} (min {sc['min_fve']}), loss +{100 * fid['ce_rel_increase']:.1f}% "
                                          f"(max {100 * sc['max_rel_ce_increase']:.0f}%)")
        rec["base_to_instruct_transfer"] = sae_cfg["format"] == "reins"
        out["sae"][str(L)] = rec
        LOG.info("%s SAE L%d (%s): FVE %.3f, loss %+.1f%%, L0 %.1f -> %s", mc.slug, L, info["file"], fid["fve"],
                 100 * fid["ce_rel_increase"], fid["l0"], "INCLUDED" if ok else "EXCLUDED (basis dropped at this depth)")
    write_json(mc.dir / "gate" / "gate.json", out)
    run.record(**{f"sae:{mc.model_id}": {L: {"file": (v.get("info") or {}).get("file"), "repo": (v.get("info") or {}).get("repo"),
                                              "included": v.get("included"), "reason": v.get("reason"),
                                              "input_norm": v.get("input_norm")}
                                          for L, v in out["sae"].items()}})
    return {"tasks": {t: v["gate"]["pass"] for t, v in out["tasks"].items()},
            "sae_included": {L: v.get("included") for L, v in out["sae"].items()}}


def _refusal_gate(mc: MCtx, relaxed: bool) -> dict:
    cfg = mc.cfg
    rc = cfg["tasks"]["refusal"]
    subs = rc["substrings"]
    nmax = int(cfg["refusal_generation"]["max_new_tokens"])
    disc = mc.tokenized("refusal", "discovery")
    nc = int(rc["calib_n"])
    calib = pd.concat([disc[disc.label == 1].iloc[:nc], disc[disc.label == 0].iloc[:nc]])
    gens, _ = generate(mc, list(calib.input_ids), nmax)
    marker = None
    lab = np.array([MET.is_refusal(M.decode(mc.tok, g), subs) for g in gens])
    if relaxed:
        firsts = [M.decode(mc.tok, g[:1]) for g, y in zip(gens, calib.label) if y == 1 and g]
        if firsts:
            vals, cnt = np.unique(firsts, return_counts=True)
            marker = str(vals[np.argmax(cnt)])
            LOG.warning("DEV tiny model: random weights never refuse; a reply containing the most common first "
                        "harmful-reply token counts as a refusal (plumbing only)")
            lab = refusal_labels(mc, gens, marker)
    first = [g[0] if g else -1 for g in gens]
    start = MET.refusal_start_set(first, list(lab), float(rc["start_token_coverage"]))
    if not start:
        raise HarnessError(f"{mc.model_id}: no refusals on harmful discovery prompts; cannot build the refusal-start set")
    (r,) = last_logits_all(mc, list(calib.input_ids), lambda lg: MET.refusal_scores(lg, start)[0])
    tau, agree = MET.calibrate_threshold(r, lab)
    y = calib.label.to_numpy()
    # clean generation labels on the harmful pool and the harmless eval prompts; first-token r(x) from the
    # same generation pass (the path every ablated refusal run uses)
    pool = mc.tokenized("refusal", "eval_harmful_pool")
    harmless = mc.tokenized("refusal", "eval_harmless")
    both = pd.concat([pool, harmless], ignore_index=True)
    g2, first2 = generate(mc, list(both.input_ids), nmax, first_logits_fn=lambda lg: MET.refusal_scores(lg, start)[0])
    lab2 = refusal_labels(mc, g2, marker)
    rscore = first2[0]
    hp = lab2[: len(pool)]
    hl = lab2[len(pool):]
    n_ref = int(hp.sum())
    harmless_rate = float(hl.mean()) if len(hl) else 0.0
    refused_idx = np.flatnonzero(hp)
    n_take = min(len(refused_idx), int(cfg["data"]["refusal"]["n_eval_harmful"]))
    take = np.sort(rng_for(cfg["seed"], mc.model_id, "refusal-eval-refused").choice(refused_idx, size=n_take, replace=False)) \
        if n_take else np.array([], dtype=int)
    ev = pd.concat([pool.iloc[take].assign(clean_refused=True),
                    harmless.assign(clean_refused=hl)], ignore_index=True)
    ev.to_parquet(mc.dir / "data" / "refusal_eval.parquet", index=False)
    ev_score = np.concatenate([rscore[: len(pool)][take], rscore[len(pool):]])
    g = cfg["gates"]["refusal"]
    reasons = []
    if n_ref < int(g["min_refused_harmful"]):
        reasons.append(f"{n_ref} harmful eval prompts refused < {g['min_refused_harmful']}")
    if harmless_rate > float(g["max_harmless_refusal"]):
        reasons.append(f"harmless refusal {harmless_rate:.3f} > {g['max_harmless_refusal']}")
    yev = ev.label.to_numpy()
    rec = {"start_ids": start, "start_tokens": mc.tok.convert_ids_to_tokens(start), "threshold": tau,
           "classifier_agreement": agree, "calib_refusal_rate_harmful": float(lab[y == 1].mean()) if (y == 1).any() else 0.0,
           "calib_refusal_rate_harmless": float(lab[y == 0].mean()) if (y == 0).any() else 0.0,
           "baseline": {"n_pool": int(len(pool)), "n_refused_pool": n_ref, "refusal_rate_pool": float(hp.mean()) if len(hp) else 0.0,
                        "harmless_refusal_rate": harmless_rate, "n_eval_harmful": int(n_take), "n_eval_harmless": int(len(harmless)),
                        "first_token_auroc_eval": stats.auroc(ev_score, yev) if len(np.unique(yev)) == 2 else None},
           "eval_first_token_score": ev_score.tolist(), "eval_pids": list(ev.pid),
           "gate": {"pass": not reasons, "reasons": reasons}, "dev_tiny_marker": marker}
    LOG.info("%s/refusal: start tokens %s, classifier agreement %.3f; %d/%d harmful pool refused, harmless %.3f -> gate %s",
             mc.slug, rec["start_tokens"], agree, n_ref, len(pool), harmless_rate, "PASS" if not reasons else f"FAIL {reasons}")
    return rec


# ============================================================================= run-level inclusion

def inclusion(cfg: dict, gates: dict) -> dict:
    """Pairs entering the analysis after the task gates and the toxicity substitution (mechanical)."""
    enforced = bool(cfg.get("gates_enforced", True))
    sub = cfg["pairs"]["substitute"]
    out = {"enforced": enforced, "pairs": [], "excluded": [], "substitutions": [], "sae": {}, "baselines": {}}
    for m, g in gates.items():
        out["baselines"][m] = {t: {"baseline": v.get("baseline"), "gate": v["gate"]} for t, v in g["tasks"].items()}
        out["sae"][m] = {L: {"included": v.get("included"), "reason": v.get("reason"),
                             "fve": (v.get("fidelity") or {}).get("fve"),
                             "loss_rel_increase": (v.get("fidelity") or {}).get("ce_rel_increase")} for L, v in g["sae"].items()}
        sc = cfg["models"][m]["sae"]
        out.setdefault("sae_setting", {})[m] = {
            "layer_offset": int(sc.get("layer_offset", 0)), "input_norm": sc.get("input_norm", "none"),
            "source": "model card"}
    for role in ("confirmatory", "development"):
        for m, t in cfg["pairs"][role]:
            if m not in gates:
                continue
            tg = gates[m]["tasks"].get(t)
            if tg is None:
                continue
            ok = tg["gate"]["pass"] or not enforced
            if ok:
                out["pairs"].append({"model": m, "task": t, "role": role,
                                     "gate_pass": tg["gate"]["pass"], "note": None if tg["gate"]["pass"] else "gate not enforced"})
                continue
            if t in ("sentiment", "answerability") and sub in gates[m]["tasks"]:
                sg = gates[m]["tasks"][sub]
                taken = any(p["model"] == m and p["task"] == sub for p in out["pairs"])
                if sg["gate"]["pass"] and not taken:
                    out["pairs"].append({"model": m, "task": sub, "role": role, "gate_pass": True,
                                         "note": f"substitute for {t}"})
                    out["substitutions"].append({"model": m, "failed": t, "substitute": sub, "reasons": tg["gate"]["reasons"]})
                    continue
            out["excluded"].append({"model": m, "task": t, "role": role, "reasons": tg["gate"]["reasons"]})
    return out
