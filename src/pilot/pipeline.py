"""Run context plus the gate, data, cache, bases and select stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bases as BZ
from . import data as DATA
from . import metrics as MET
from . import models as M
from . import prompts as P
from . import selection as SEL
from .sae import SAE
from .sae import resolve as resolve_sae
from .utils import LOG, MODEL_PATTERNS, hf_snapshot, load_run_file, read_json, rng_for, update_json, write_json

TASKS = ("truth", "refusal")


class HarnessInvalid(RuntimeError):
    """A harness check failed: stop and report INVALID HARNESS."""


class Ctx:
    def __init__(self, cfg: dict, run_dir: Path, model_id: str, smoke: bool, watchdog=None):
        self.cfg, self.run_dir, self.model_id, self.smoke, self.wd = cfg, Path(run_dir), model_id, smoke, watchdog
        self.device = M.pick_device(allow_cpu=smoke)
        self.dtype = M.default_dtype(self.device)
        self._model = self._tok = self._sae = None
        self.dev = bool(cfg.get("dev_tiny", False))
        self.bs = int(cfg["batch_size"])
        self.max_tok = int(cfg["max_batch_tokens"])

    # ------------------------------------------------------------------ paths / manifest
    def d(self, stage: str) -> Path:
        p = self.run_dir / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def manifest(self) -> dict:
        return read_json(self.manifest_path) if self.manifest_path.exists() else {}

    def record(self, **kv) -> None:
        update_json(self.manifest_path, kv)

    # ------------------------------------------------------------------ lazy resources
    def _ensure_downloaded(self):
        if not Path(self.model_id).exists():          # hub id (a local dir is the dev tiny model)
            hf_snapshot(self.model_id, allow_patterns=MODEL_PATTERNS)

    @property
    def tok(self):
        if self._tok is None:
            self._ensure_downloaded()
            self._tok = M.load_tokenizer(self.model_id)
        return self._tok

    @property
    def model(self):
        if self._model is None:
            self._ensure_downloaded()
            self._model = M.load_model(self.model_id, self.device, self.dtype)
        return self._model

    @property
    def layers(self):
        return M.decoder_layers(self.model)

    @property
    def L(self) -> int:
        return int(self.manifest()["layer"])

    @property
    def T(self) -> int:
        return int(self.manifest()["positions"]["T"])

    @property
    def gate(self) -> dict:
        return read_json(self.run_dir / "gate" / "gate.json")

    def sae(self) -> SAE:
        if self._sae is None:
            from huggingface_hub import hf_hub_download

            info = self.manifest()["sae"]
            path = info["file"] if info["format"] == "local" else hf_hub_download(info["repo"], info["file"])
            self._sae = SAE.load(Path(path), self.device)
        return self._sae

    def split(self, set_name: str, split: str) -> pd.DataFrame:
        return DATA.load(self.run_dir / "data", set_name, split)

    def batches(self, token_lists):
        return M.make_batches(token_lists, self.tok.pad_token_id, self.device, self.bs, self.max_tok)

    @property
    def bases_enabled(self) -> list[str]:
        return list(self.cfg["bases"])

    @property
    def use_mlp(self) -> bool:
        return "mlp" in self.cfg["bases"]

    def score_fn(self, task: str):
        g = self.gate
        if task == "truth":
            ids = g["answer_tokens"]["ids"]
            return lambda logits: MET.truth_scores(logits, ids)
        start = g["refusal"]["start_ids"]
        return lambda logits: MET.refusal_scores(logits, start)


# =============================================================================== data

def stage_data(ctx: Ctx) -> dict:
    tmpl_check = __import__("pilot.check", fromlist=["load_check_config"]).load_check_config()["template"]
    if tmpl_check != ctx.cfg["tasks"]["truth"]["template"]:
        raise HarnessInvalid("truth template in config/pilot.yaml differs from config/check.yaml")
    summary = DATA.build(ctx.cfg, ctx.tok, ctx.d("data"))
    write_json(ctx.d("data") / "data_summary.json", summary)
    ctx.record(positions=summary["positions"], data_files=summary["files"])
    return {"splits": {k: v["n"] for k, v in summary["splits"].items()}}


# =============================================================================== gate

def _generate(ctx: Ctx, token_lists, max_new: int) -> list[list[int]]:
    out = [None] * len(token_lists)
    for b in ctx.batches(token_lists):
        for i, o in zip(b.idx, M.generate_greedy(ctx.model, ctx.tok, b, max_new)):
            out[int(i)] = o
    return out


@torch.inference_mode()
def _last_logits_all(ctx: Ctx, token_lists, fn):
    res = None
    for b in ctx.batches(token_lists):
        outs = fn(M.last_logits(ctx.model, b))
        outs = outs if isinstance(outs, tuple) else (outs,)
        if res is None:
            res = [np.zeros((len(token_lists),) + tuple(o.shape[1:]), dtype=np.float64) for o in outs]
        for r, o in zip(res, outs):
            r[b.idx] = o.double().cpu().numpy()
    return res


def _sae_sanity(ctx: Ctx, sae: SAE, token_lists) -> dict:
    """FVE (dictionary_learning definition) and next-token CE increase with x-hat spliced in,
    on chat-formatted prompts, excluding attention-sink tokens."""
    L = ctx.L
    layers = ctx.layers
    mult = float(ctx.cfg["sink_norm_multiple"])
    xs, xh = [], []
    ce_clean = ce_sae = 0.0
    n_tok = 0
    n_sink = n_all = 0
    state = {}

    def splice(module, args, output):
        h, rebuild = M._split_output(output)
        mask = state["mask"]
        hf = h.float()
        norms = hf.norm(dim=-1)
        med = norms[mask].median()
        keep = mask & (norms <= mult * med)
        state["keep"] = keep
        rec = sae.decode(sae.encode(hf))
        state["x"] = hf[keep]
        state["xhat"] = rec[keep]
        if state["do_splice"]:
            h2 = torch.where(keep.unsqueeze(-1), rec.to(h.dtype), h)
            return rebuild(h2)
        return output

    handle = layers[L].register_forward_hook(splice)
    try:
        with torch.inference_mode():
            for b in ctx.batches(token_lists):
                mask = b.attention_mask.bool()
                state["mask"] = mask
                ces = []
                for do in (False, True):
                    state["do_splice"] = do
                    out = ctx.model(input_ids=b.input_ids, attention_mask=b.attention_mask,
                                    position_ids=b.position_ids, use_cache=False)
                    lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
                    tgt = b.input_ids[:, 1:]
                    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    valid = mask[:, 1:] & mask[:, :-1]
                    ces.append(float(nll[valid].sum()))
                    if not do:
                        xs.append(state["x"].cpu())
                        xh.append(state["xhat"].cpu())
                        n_sink += int((mask & ~state["keep"]).sum())
                        n_all += int(mask.sum())
                        n_tok += int(valid.sum())
                ce_clean += ces[0]
                ce_sae += ces[1]
    finally:
        handle.remove()
    X = torch.cat(xs)
    Xh = torch.cat(xh)
    fve = float(1 - (X - Xh).var(0).sum() / X.var(0).sum())
    # diagnostic only: would global-RMS-normalised inputs fit better? (never switched automatically)
    s = float(X.pow(2).sum(-1).mean().sqrt() / np.sqrt(X.shape[1]))
    Xn = X.to(ctx.device) / s
    Xhn = sae.decode(sae.encode(Xn)).cpu() * s
    fve_rms = float(1 - (X - Xhn).var(0).sum() / X.var(0).sum())
    l0 = float((sae.encode(X[:2048].to(ctx.device)) > 0).float().sum(-1).mean())
    return {"fve": fve, "ce_clean": ce_clean / n_tok, "ce_sae": ce_sae / n_tok,
            "ce_increase": (ce_sae - ce_clean) / n_tok, "l0": l0, "n_tokens": int(X.shape[0]),
            "sink_fraction": n_sink / max(1, n_all), "diagnostic_fve_if_global_rms_normalised": fve_rms}


def stage_gate(ctx: Ctx) -> dict:
    cfg, gcfg = ctx.cfg, ctx.cfg["gate"]
    model, tok = ctx.model, ctx.tok
    tc = M.text_config(model)
    params = M.count_params(model)
    lin = M.linear_attn_impl_report(model)
    if lin.get("hybrid") and not lin.get("fused_kernels"):
        LOG.warning("hybrid model is using the pure-PyTorch Gated DeltaNet fallback; expect a slower ablation stage "
                    "(the smoke projection measures it)")
    ctx.record(model={"id": ctx.model_id, "n_layers": tc.num_hidden_layers, "d_model": tc.hidden_size,
                      "params": params, "dtype": str(ctx.dtype), "device": str(ctx.device),
                      "linear_attention": lin, "model_sha": _model_sha(ctx.model_id)})
    out = {"model": ctx.model_id, "model_reason": ctx.manifest().get("model_reason")}

    # ---------------------------------------------------------------- format checks (thinking mode)
    tdisc = ctx.split("truth", "discovery")
    rdisc = ctx.split("refusal", "discovery")
    n_fmt = int(gcfg["format_n"])
    t_out = _generate(ctx, list(tdisc.input_ids[:n_fmt]), int(gcfg["format_tokens"]))
    t_txt = [M.decode(tok, o) for o in t_out]
    harmless = rdisc[rdisc.label == 0]
    r_out = _generate(ctx, list(harmless.input_ids[:n_fmt]), int(gcfg["format_tokens"]))
    r_txt = [M.decode(tok, o) for o in r_out]
    fmt = {"truth_ok": [P.starts_with_answer(t) for t in t_txt],
           "refusal_ok": [bool(t.strip()) and not P.starts_with_thinking(t) for t in r_txt],
           "truth_responses": t_txt}
    out["format_check"] = {"truth_passed": sum(fmt["truth_ok"]), "refusal_passed": sum(fmt["refusal_ok"]),
                           "n": n_fmt, "truth_responses": t_txt}
    LOG.info("format check: truth %d/%d start with True/False; refusal (harmless) %d/%d start without a "
             "thinking block", sum(fmt["truth_ok"]), len(t_txt), sum(fmt["refusal_ok"]), len(r_txt))
    if not (all(fmt["truth_ok"]) and all(fmt["refusal_ok"])):
        if ctx.dev:
            LOG.warning("DEV tiny model: format check failed as expected for random weights; relaxed")
        else:
            write_json(ctx.d("gate") / "gate.json", out)
            raise HarnessInvalid("format check failed: responses start with a thinking block or non-answer text")

    # ---------------------------------------------------------------- answer tokens
    variants = P.answer_variants(tok)
    for k, v in variants.items():
        if not v["distinct"]:
            raise HarnessInvalid(f"answer tokens for True/False are identical ({k}: {v['tokens']})")
    gid = torch.tensor(variants["bare"]["ids"] + variants["space"]["ids"], device=ctx.device)
    (lp,) = _last_logits_all(ctx, list(tdisc.input_ids), lambda lg: torch.log_softmax(lg, -1)[:, gid])
    mass = {"bare": float(np.mean(np.exp(lp[:, 0]) + np.exp(lp[:, 1]))),
            "space": float(np.mean(np.exp(lp[:, 2]) + np.exp(lp[:, 3])))}
    chosen = "bare" if mass["bare"] >= mass["space"] else "space"
    out["answer_tokens"] = {"variant": chosen, "ids": variants[chosen]["ids"], "tokens": variants[chosen]["tokens"],
                            "mass_by_variant": mass, "variants": variants}
    LOG.info("answer tokens: using %s %s (mean mass bare=%.4f, space=%.4f)", chosen, variants[chosen]["tokens"],
             mass["bare"], mass["space"])

    # ---------------------------------------------------------------- refusal calibration (stage 0)
    subs = cfg["tasks"]["refusal"]["substrings"]
    nc = int(gcfg["calib_n"])
    calib = pd.concat([rdisc[rdisc.label == 1].iloc[:nc], rdisc[rdisc.label == 0].iloc[:nc]])
    gens = _generate(ctx, list(calib.input_ids), int(gcfg["calib_tokens"]))
    lab = np.array([MET.is_refusal(M.decode(tok, g), subs) for g in gens])
    if ctx.dev and not lab.any():
        LOG.warning("DEV tiny model: no real refusals; treating harmful calibration prompts as refusals (relaxed)")
        lab = calib.label.to_numpy() == 1
    first = [g[0] if g else -1 for g in gens]
    start = MET.refusal_start_set(first, list(lab), float(cfg["tasks"]["refusal"]["start_token_coverage"]))
    if not start:
        write_json(ctx.d("gate") / "gate.json", out)
        raise HarnessInvalid("no refusals observed on harmful discovery prompts; cannot build the refusal-start set")
    (r,) = _last_logits_all(ctx, list(calib.input_ids), lambda lg: MET.refusal_scores(lg, start)[0])
    tau, agree = MET.calibrate_threshold(r, lab)
    y = calib.label.to_numpy()
    mode = "first_token" if agree >= float(gcfg["classifier_min_agreement"]) else "generation"
    if mode == "generation":
        LOG.warning("first-token refusal classifier agrees with generation labels on only %.1f%% of stage-0 prompts "
                    "(< %.0f%%): switching refusal damage to generation for ALL configurations", 100 * agree,
                    100 * float(gcfg["classifier_min_agreement"]))
    out["refusal"] = {
        "start_ids": start, "start_tokens": tok.convert_ids_to_tokens(start), "threshold": tau,
        "classifier_agreement": agree, "damage_mode": mode, "n_calib_harmful": int((y == 1).sum()),
        "n_calib_harmless": int((y == 0).sum()), "calib_refusal_rate_harmful": float(lab[y == 1].mean()),
        "calib_refusal_rate_harmless": float(lab[y == 0].mean()), "calib_prompt_ids": list(calib.pid),
        "first_token_counts": {tok.convert_ids_to_tokens(int(t)): int(c) for t, c in
                               zip(*np.unique([f for f, l in zip(first, lab) if l], return_counts=True))},
    }
    LOG.info("refusal-start tokens %s; threshold %.3f; classifier agreement %.3f; damage mode %s",
             out["refusal"]["start_tokens"], tau, agree, mode)

    # ---------------------------------------------------------------- clean refusal labels on eval prompts
    rev = ctx.split("refusal", "eval")
    gens = _generate(ctx, list(rev.input_ids), int(gcfg["eval_label_tokens"]))
    elab = np.array([MET.is_refusal(M.decode(tok, g), subs) for g in gens])
    np.save(ctx.d("gate") / "eval_refusal_labels.npy", elab)
    ye = rev.label.to_numpy()
    n_ref = int(elab[ye == 1].sum())
    if n_ref == 0:
        if ctx.dev:
            LOG.warning("DEV tiny model: no clean refusals on harmful eval prompts; treating them all as refused (relaxed)")
            elab = elab | (ye == 1)
            np.save(ctx.d("gate") / "eval_refusal_labels.npy", elab)
            n_ref = int(elab[ye == 1].sum())
        else:
            write_json(ctx.d("gate") / "gate.json", out)
            raise HarnessInvalid("the clean model refused none of the harmful eval prompts: refusal damage is undefined "
                                 "(check the refusal substrings in config/pilot.yaml against gate/gate.json)")
    harmless_rate = float(elab[ye == 0].mean())
    under = n_ref < int(gcfg["min_refused_harmful"]) or harmless_rate > float(gcfg["refuse_harmless"])
    out["refusal_eval"] = {"n_harmful": int((ye == 1).sum()), "n_refused_harmful": n_ref,
                           "refusal_rate_harmful": float(elab[ye == 1].mean()), "refusal_rate_harmless": harmless_rate,
                           "min_refused_harmful": gcfg["min_refused_harmful"], "max_refuse_harmless": gcfg["refuse_harmless"],
                           "underpowered": under}
    if under:
        LOG.warning("refusal underpowered: %d refused harmful eval prompts (need >= %s), harmless refusal rate "
                    "%.3f (max %s). Refusal results will be reported as underpowered.", n_ref,
                    gcfg["min_refused_harmful"], harmless_rate, gcfg["refuse_harmless"])
    else:
        LOG.info("refusal eval: %d refused harmful prompts, harmless refusal rate %.3f", n_ref, harmless_rate)

    # ---------------------------------------------------------------- SAE: resolve, load, sanity check
    mcfg = cfg["models"][ctx.model_id]["sae"]
    info = resolve_sae(ctx.model_id, mcfg, tc.num_hidden_layers, float(cfg["layer_depth"]))
    ctx.record(sae=info, layer=info["layer"])
    LOG.info("SAE resolved: %s %s (layer %d of %d; available layers %s)", info["repo"], info["file"], info["layer"],
             tc.num_hidden_layers, info["available_layers"])
    sae = ctx.sae()
    if sae.d != tc.hidden_size:
        raise HarnessInvalid(f"SAE d={sae.d} does not match model d_model={tc.hidden_size}")
    nsp = int(cfg["sae_check"]["n_prompts"])
    gfit = ctx.split("generic", "fit")
    per = max(1, nsp // 3)
    toks = list(gfit.input_ids[:per]) + list(tdisc.input_ids[:per]) + list(rdisc[rdisc.label == 0].input_ids[:per])
    chk = _sae_sanity(ctx, sae, toks)
    sc = cfg["sae_check"]
    chk["pass"] = bool(chk["fve"] >= sc["min_fve"] and chk["ce_increase"] <= sc["max_ce_increase"])
    chk["thresholds"] = sc
    chk["base_to_instruct_transfer_check"] = ctx.model_id == "Qwen/Qwen3.5-4B"
    out["sae_check"] = chk
    LOG.info("SAE sanity: FVE %.3f (min %.2f), CE increase %.3f nats (max %.2f), L0 %.1f, sinks %.2f%% -> %s",
             chk["fve"], sc["min_fve"], chk["ce_increase"], sc["max_ce_increase"], chk["l0"],
             100 * chk["sink_fraction"], "PASS" if chk["pass"] else "FAIL")
    write_json(ctx.d("gate") / "gate.json", out)
    ctx.record(sae_check=chk, refusal_classifier={k: out["refusal"][k] for k in
                                                    ("start_ids", "start_tokens", "threshold", "damage_mode",
                                                     "classifier_agreement")},
               answer_tokens=out["answer_tokens"])
    if not chk["pass"] and ctx.dev:
        LOG.warning("DEV tiny model: random SAE fails the sanity check as expected; relaxed")
    elif not chk["pass"]:
        raise HarnessInvalid("SAE sanity check failed at the resolved layer (hook/site mismatch or poor "
                             "transfer). Documented contingency: Qwen/Qwen3-8B + adamkarvonen/qwen3-8b-saes, via "
                             "editing config/check.yaml and `bash run_pilot.sh --check --force` (manual).")
    return {"sae_fve": chk["fve"], "refused_harmful": n_ref, "damage_mode": mode}


def _model_sha(model_id: str):
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:  # noqa: BLE001
        return None


# =============================================================================== cache

@torch.inference_mode()
def _capture_split(ctx: Ctx, df: pd.DataFrame, fn, mlp: bool, pool: list | None = None):
    """Layer-L activations at the T intervention positions + clean scores (+ all-position
    activations appended to `pool` when given)."""
    T = ctx.T
    N = len(df)
    acts = None
    mlps = None
    scores = None
    cap = M.Capture(ctx.layers, ctx.L, None, mlp=mlp)
    try:
        for b in ctx.batches(list(df.input_ids)):
            logits = M.last_logits(ctx.model, b)
            h = cap.h
            if acts is None:
                acts = torch.zeros((N, T, h.shape[-1]), dtype=h.dtype)
            acts[torch.as_tensor(b.idx)] = h[:, -T:, :].cpu()
            if mlp:
                mm = cap.m[:, -T:, :]
                if mlps is None:
                    mlps = torch.zeros((N, T, mm.shape[-1]), dtype=mm.dtype)
                mlps[torch.as_tensor(b.idx)] = mm.cpu()
            if fn is not None:
                outs = fn(logits)
                if scores is None:
                    scores = [np.zeros(N) for _ in outs]
                for s, o in zip(scores, outs):
                    s[b.idx] = o.double().cpu().numpy()
            if pool is not None:
                mask = b.attention_mask.bool()
                pool.append(h[mask].float().cpu())
    finally:
        cap.remove()
    return acts, mlps, scores


def stage_cache(ctx: Ctx) -> dict:
    out = {}
    pool = []
    mlp = ctx.use_mlp
    clean = {}
    for task in TASKS:
        fn = ctx.score_fn(task)
        res = {}
        for split in ("discovery", "eval"):
            df = ctx.split(task, split)
            acts, mlps, sc = _capture_split(ctx, df, fn, mlp, pool if split == "discovery" else None)
            res[f"{split}_int"] = acts
            res[f"{split}_mlp"] = mlps
            res[f"{split}_labels"] = df.label.to_numpy()
            res[f"{split}_score"] = sc[0]
            res[f"{split}_aux"] = sc[1]
        torch.save(res, ctx.d("cache") / f"{task}.pt")
        lab, s = res["eval_labels"], res["eval_score"]
        from . import stats

        if task == "truth":
            clean["truth"] = {"auroc": stats.auroc(s, lab), "accuracy_at_0": float(np.mean((s > 0) == lab)),
                              "coherence": float(np.mean(res["eval_aux"])), "n": len(s)}
        else:
            tau = ctx.gate["refusal"]["threshold"]
            clean["refusal"] = {"auroc_r": stats.auroc(s, lab), "classifier_rate_harmful": float(np.mean(s[lab == 1] > tau)),
                                "classifier_rate_harmless": float(np.mean(s[lab == 0] > tau)), "n": len(s)}
        out[task] = {k: (tuple(v.shape) if torch.is_tensor(v) else None) for k, v in res.items() if torch.is_tensor(v)}
        LOG.info("cached %s: discovery %s, eval %s", task, tuple(res["discovery_int"].shape), tuple(res["eval_int"].shape))
    gfit = ctx.split("generic", "fit")
    _capture_split(ctx, gfit, None, False, pool)
    P_all = torch.cat(pool)
    norms = P_all.norm(dim=-1)
    med = float(norms.median())
    thr = float(ctx.cfg["sink_norm_multiple"]) * med
    keep = norms <= thr
    frac = float(1 - keep.float().mean())
    P_keep = P_all[keep]
    n_target = int(ctx.cfg["pca"]["n_tokens"])
    rng = rng_for(ctx.cfg["seed"], "pca-pool")
    if P_keep.shape[0] > n_target:
        P_keep = P_keep[torch.as_tensor(np.sort(rng.choice(P_keep.shape[0], n_target, replace=False)))]
    torch.save({"pool": P_keep, "median_norm": med, "sink_threshold": thr}, ctx.d("cache") / "pool.pt")
    LOG.info("PCA pool: %d tokens kept of %d; attention-sink tokens (norm > %.0fx median %.1f) excluded: %.3f%%",
             P_keep.shape[0], P_all.shape[0], ctx.cfg["sink_norm_multiple"], med, 100 * frac)
    ctx.record(sink={"median_norm": med, "threshold": thr, "fraction_excluded_pool": frac})
    write_json(ctx.d("cache") / "clean_metrics.json", clean)
    LOG.info("clean metrics: %s", clean)
    return {"pool_tokens": int(P_keep.shape[0]), "sink_fraction": frac, **clean}


# =============================================================================== bases

def load_bases(ctx: Ctx, keys=None) -> dict:
    keys = keys or ctx.bases_enabled
    out = {}
    for k in keys:
        if k == "sae":
            out[k] = BZ.SAEBasis(ctx.sae())
        elif k == "pca":
            p = load_run_file(ctx.run_dir / "bases" / "pca.pt")
            out[k] = BZ.PCABasis(p["mu"], p["U"], p["evals"]).to(ctx.device)
        elif k == "dims":
            out[k] = BZ.DimsBasis(int(ctx.manifest()["model"]["d_model"]), ctx.device)
        elif k == "mlp":
            out[k] = BZ.MLPBasis(ctx.layers[ctx.L].mlp.down_proj.weight.detach()).to(ctx.device)
        else:
            raise ValueError(k)
    return out


def sink_keep(acts: torch.Tensor, thr: float) -> torch.Tensor:
    return acts.float().norm(dim=-1) <= thr


def stage_bases(ctx: Ctx) -> dict:
    pool = load_run_file(ctx.run_dir / "cache" / "pool.pt")
    mu, U, ev = BZ.fit_pca(pool["pool"])
    torch.save({"mu": mu, "U": U, "evals": ev}, ctx.d("bases") / "pca.pt")
    cum = torch.cumsum(ev, 0) / ev.sum()
    info = {"pca": {"n_components": int(U.shape[0]), "n_tokens": int(pool["pool"].shape[0]),
                    "var_explained_top": {str(k): float(cum[min(k, len(cum)) - 1]) for k in (1, 10, 100, 500)}}}
    LOG.info("PCA: %d components from %d tokens; top-10 explain %.1f%% of variance", U.shape[0],
             pool["pool"].shape[0], 100 * float(cum[min(10, len(cum)) - 1]))
    B = load_bases(ctx)
    thr = pool["sink_threshold"]
    seed = ctx.cfg["seed"]
    for task in TASKS:
        c = load_run_file(ctx.run_dir / "cache" / f"{task}.pt")
        X = c["discovery_int"].to(ctx.device)
        Mh = c["discovery_mlp"].to(ctx.device) if c["discovery_mlp"] is not None else None
        keep = sink_keep(X, thr)
        frac = float(1 - keep.float().mean())
        means = {}
        for k, basis in B.items():
            means[k] = BZ.masked_offset_mean(basis, X, Mh, keep).cpu()
        alive = None
        if "sae" in B:
            rate = BZ.firing_rate(B["sae"], X, keep).cpu()
            alive = (rate >= float(ctx.cfg["random_rule"]["sae_min_fire_rate"])).numpy()
            info.setdefault("sae_alive", {})[task] = int(alive.sum())
            LOG.info("%s: %d of %d SAE latents fire on >= %.0f%% of discovery tokens", task, alive.sum(), len(alive),
                     100 * ctx.cfg["random_rule"]["sae_min_fire_rate"])
        # anchors: mass-mean direction at the final token, and random unit directions
        y = torch.as_tensor(c["discovery_labels"], device=ctx.device)
        xf = X[:, -1].float()
        kf = keep[:, -1]
        u = xf[kf & (y == 1)].mean(0) - xf[kf & (y == 0)].mean(0)
        u = u / u.norm()
        gen = torch.Generator().manual_seed(int(rng_for(seed, task, "random-dir").integers(2**31)))
        R = torch.randn(int(ctx.cfg["anchors"]["random_dir"]), X.shape[-1], generator=gen).to(ctx.device)
        R = R / R.norm(dim=-1, keepdim=True)
        anchors = {"massmean": u.cpu(), "random_dir": R.cpu()}
        for name, U_ in (("massmean", u[None]), ("random_dir", R)):
            means[f"anchor:{name}"] = BZ.masked_offset_mean(BZ.DirectionBasis(U_), X, None, keep).cpu()
        torch.save({"means": means, "alive": alive, "anchors": anchors, "keep": keep.cpu(),
                    "sink_fraction": frac}, ctx.d("bases") / f"{task}.pt")
        info.setdefault("sink_fraction_discovery_int", {})[task] = frac
        LOG.info("%s: per-offset mean values for %s; sink tokens excluded from discovery stats: %.3f%%",
                 task, list(B), 100 * frac)
    write_json(ctx.d("bases") / "bases_summary.json", info)
    return info


# =============================================================================== select

def _resample_subsets(ctx: Ctx, y: np.ndarray, keep: np.ndarray, task: str) -> list[np.ndarray]:
    R = int(ctx.cfg["discovery_resamples"])
    frac = float(ctx.cfg["discovery_resample_frac"])
    subs = []
    for r in range(R):
        rng = rng_for(ctx.cfg["seed"], task, "discovery-resample", r)
        idx = []
        for lab in (0, 1):
            pool = np.flatnonzero((y == lab) & keep)
            idx.append(rng.choice(pool, size=max(1, int(round(frac * len(pool)))), replace=False))
        subs.append(np.sort(np.concatenate(idx)))
    return subs


def stage_select(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    B = load_bases(ctx)
    rows = []
    stab = []
    diags = {}
    T = ctx.T
    store = int(cfg["rank_store"])
    budgets = [int(k) for k in cfg["budgets"]]
    for task in TASKS:
        c = load_run_file(ctx.run_dir / "cache" / f"{task}.pt")
        bz = load_run_file(ctx.run_dir / "bases" / f"{task}.pt")
        y_np = c["discovery_labels"]
        keep_all = bz["keep"].to(ctx.device)
        keep_final = keep_all[:, -1].cpu().numpy()
        subs = _resample_subsets(ctx, y_np, keep_final, task)
        dfx = ctx.split(task, "discovery")
        # ---------------------------------------------------- gradients for attribution (one backward per example)
        need_attr = "attrib" in cfg["rules"]
        contrib = {k: None for k in B}
        if need_attr:
            score_fn = ctx.score_fn(task)
            ysign = torch.as_tensor(2.0 * y_np - 1.0, device=ctx.device, dtype=torch.float32)
            for b in ctx.batches(list(dfx.input_ids)):
                ys = ysign[torch.as_tensor(b.idx, device=ctx.device)]
                x, g, m = M.grad_at_layer(ctx.model, ctx.layers, ctx.L, b, T,
                                          lambda lg: ys * score_fn(lg)[0], mlp=ctx.use_mlp)
                kb = keep_all[torch.as_tensor(b.idx, device=ctx.device)]
                for k, basis in B.items():
                    v = bz["means"][k].to(ctx.device)
                    cb = SEL.attrib_contrib(basis, x, g, m, v, kb).cpu()
                    if contrib[k] is None:
                        contrib[k] = torch.zeros(len(dfx), cb.shape[1])
                    contrib[k][torch.as_tensor(b.idx)] = cb
            LOG.info("%s: attribution gradients computed for %d discovery examples", task, len(dfx))
        X = c["discovery_int"][:, -1].to(ctx.device)
        Mf = c["discovery_mlp"][:, -1].to(ctx.device) if c["discovery_mlp"] is not None else None
        y = torch.as_tensor(y_np, device=ctx.device)
        for k, basis in B.items():
            Rf = basis.read(X.float(), None if Mf is None else Mf.float())
            orders_by_rule = {}
            for res, sub in enumerate(subs):
                si = torch.as_tensor(sub, device=ctx.device)
                Rs, ys = Rf[si], y[si]
                for rule in cfg["rules"]:
                    if rule == "random":
                        continue
                    tag = f"{task}/{k}/{rule}/r{res}"
                    if rule == "meandiff":
                        order, score, _ = SEL.rank_meandiff(Rs, ys)
                    elif rule == "l1probe":
                        d_abs = SEL.cohens_d(Rs, ys).abs().cpu().numpy()
                        pre = int(cfg["l1probe"]["sae_prefilter"]) if k == "sae" else None
                        order, score, dg = SEL.rank_l1probe(Rs, ys, d_abs, cfg["l1probe"], pre, tag)
                        diags[tag] = dg
                    elif rule == "attrib":
                        A = contrib[k][torch.as_tensor(sub)].mean(0).numpy()
                        order, score = SEL.rank_attrib(A)
                    else:
                        raise ValueError(rule)
                    orders_by_rule.setdefault(rule, []).append(order)
                    top = order[:store]
                    rows.append(pd.DataFrame({"task": task, "basis": k, "rule": rule, "resample": res, "draw": -1,
                                              "unit": top.astype(np.int64), "rank": np.arange(len(top)),
                                              "score": np.asarray(score)[top].astype(np.float64)}))
            if "random" in cfg["rules"]:
                allowed = bz["alive"] if k == "sae" else None
                for draw in range(int(cfg["random_draws"])):
                    order, score = SEL.rank_random(basis.n_units, rng_for(cfg["seed"], task, k, "random", draw), allowed)
                    top = order[:store]
                    rows.append(pd.DataFrame({"task": task, "basis": k, "rule": "random", "resample": -1, "draw": draw,
                                              "unit": top.astype(np.int64), "rank": np.arange(len(top)),
                                              "score": np.zeros(len(top))}))
            for rule, orders in orders_by_rule.items():
                for kk in budgets:
                    jm, jmin, npairs = SEL.jaccard_topk(orders, kk)
                    if npairs:
                        stab.append({"task": task, "basis": k, "rule": rule, "k": kk, "jaccard_mean": jm,
                                     "jaccard_min": jmin, "n_pairs": npairs})
            LOG.info("%s/%s: rankings done for %s", task, k, [r for r in cfg["rules"]])
    pd.concat(rows, ignore_index=True).to_parquet(ctx.d("select") / "rankings.parquet", index=False)
    st = pd.DataFrame(stab, columns=["task", "basis", "rule", "k", "jaccard_mean", "jaccard_min", "n_pairs"])
    st.to_csv(ctx.d("select") / "stability.csv", index=False)
    if st.empty:
        LOG.warning("selection stability not computed: needs >= 2 discovery resamples (have %s)",
                    cfg["discovery_resamples"])
    write_json(ctx.d("select") / "l1_diagnostics.json", diags)
    n_unconv = sum(d["unconverged_steps"] for d in diags.values())
    return {"rankings": int(sum(len(r) for r in rows)), "l1_unconverged_steps": n_unconv,
            "sae_prefiltered": any(d.get("prefiltered") for d in diags.values())}
