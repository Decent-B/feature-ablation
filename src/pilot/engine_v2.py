"""v2 execution engine: run / model contexts, forward and generation primitives, and the TaskRunner that
executes one intervention (damage + cost) for a (model, task, depth).

Damage pass   single-token tasks: one forward, logits at the first response position (fast path: layers L+1..).
              refusal: greedy generation (full model) on the clean-refused harmful + harmless eval prompts,
              hook on prompt tokens during prefill (S1) and also on every generated token (S2).
Cost pass     generic prompt + its clean 8-token continuation in one teacher-forced forward; KL(clean || ablated)
              averaged over the first response position + 8 continuation positions (v2), and at the first
              position only (v1). S1 edits prompt tokens; S2 also the continuation tokens.
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
from . import sites as ST
from transformers import LogitsProcessorList

from .utils import LOG, load_run_file, read_json, rng_for, update_json
from .v2common import HarnessError, TUNING_PATH, depths_for, layer_for, slug


# ============================================================================= contexts

class RunCtx:
    def __init__(self, cfg: dict, run_dir: Path, watchdog=None):
        self.cfg, self.run_dir, self.wd = cfg, Path(run_dir), watchdog
        self.phase, self.dev_tiny = cfg["phase"], bool(cfg.get("dev_tiny"))
        self.device = M.pick_device(allow_cpu=self.phase == "smoke" or self.dev_tiny)
        self.dtype = M.default_dtype(self.device)
        self._m = {}
        self.timers = {}

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def manifest(self) -> dict:
        return read_json(self.manifest_path) if self.manifest_path.exists() else {}

    def record(self, **kv) -> None:
        update_json(self.manifest_path, kv)

    def d(self, *parts) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def mctx(self, model_id: str) -> "MCtx":
        if model_id not in self._m:
            if not self.cfg.get("keep_models_loaded", True):
                for k in list(self._m):
                    self._m.pop(k).unload()
            self._m[model_id] = MCtx(self, model_id)
        return self._m[model_id]

    def unload_all(self):
        for k in list(self._m):
            self._m.pop(k).unload()


class MCtx:
    def __init__(self, run: RunCtx, model_id: str):
        self.run, self.cfg, self.model_id = run, run.cfg, model_id
        self.slug = slug(model_id)
        self.dir = run.d("models", self.slug)
        self.weights = self.cfg["models"][model_id].get("weights") or model_id
        self._model = self._tok = None
        self._sae = {}
        self.tuning = self._load_tuning()

    # ---------------------------------------------------------------- resources
    def _load_tuning(self) -> dict:
        b = dict(self.cfg["batching"])
        key = ("dev_tiny:" if self.run.dev_tiny else "") + self.model_id
        if TUNING_PATH.exists():
            t = read_json(TUNING_PATH).get(key)
            if t and t.get("device") == str(self.run.device):
                for k in ("max_batch_tokens", "gen_batch_size", "grad_batch_tokens"):
                    if t.get(k):
                        b[k] = int(t[k])
                b["source"] = f"{TUNING_PATH} ({t.get('tuned_at')})"
        b.setdefault("source", "config/v2.yaml defaults")
        return b

    @property
    def tok(self):
        if self._tok is None:
            self._tok = M.load_tokenizer(self.weights)
        return self._tok

    @property
    def model(self):
        if self._model is None:
            if not Path(self.weights).exists():
                from .utils import MODEL_PATTERNS, hf_snapshot

                hf_snapshot(self.weights, allow_patterns=MODEL_PATTERNS)
            self._model = M.load_model(self.weights, self.run.device, self.run.dtype)
        return self._model

    def unload(self):
        self._model = None
        self._sae = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def layers(self):
        return M.decoder_layers(self.model)

    @property
    def n_layers(self) -> int:
        return int(M.text_config(self.model).num_hidden_layers)

    @property
    def d_model(self) -> int:
        return int(M.text_config(self.model).hidden_size)

    def depths(self) -> list[float]:
        return depths_for(self.cfg, self.model_id)

    def layer(self, depth: float) -> int:
        return layer_for(depth, self.n_layers)

    # ---------------------------------------------------------------- per-model artefacts
    @property
    def gate(self) -> dict:
        return read_json(self.dir / "gate" / "gate.json")

    def tokenized(self, task: str, split: str) -> pd.DataFrame:
        from .data_v2 import load_tok

        return load_tok(self.dir / "data" / f"{task}_{split}.parquet")

    def prep(self, task: str, L: int) -> dict:
        return load_run_file(self.dir / "prep" / task / f"L{L}.pt")

    def sink_thr(self, L: int) -> float:
        return float(read_json(self.dir / "prep" / "norms.json")[str(L)]["sink_threshold"])

    def sae(self, L: int):
        if L not in self._sae:
            from . import sae_v2

            g = self.gate["sae"][str(L)]
            if not g.get("info"):
                return None
            sae = sae_v2.load(self.cfg["models"][self.model_id]["sae"], g["info"], self.run.device)
            sae_v2.set_input_norm(sae, g.get("input_norm", "none"), g.get("scale_stats"))
            self._sae[L] = sae
        return self._sae[L]

    def sae_included(self, L: int) -> bool:
        return bool(self.gate["sae"].get(str(L), {}).get("included"))

    def bases(self, L: int, keys=None) -> dict:
        out = {}
        for k in keys or self.cfg["bases"]:
            if k == "sae":
                if self.sae_included(L):
                    out[k] = BZ.SAEBasis(self.sae(L))
            elif k == "pca":
                p = load_run_file(self.dir / "prep" / f"pca_L{L}.pt")
                out[k] = BZ.PCABasis(p["mu"], p["U"], p["evals"]).to(self.run.device)
            elif k == "dims":
                out[k] = BZ.DimsBasis(self.d_model, self.run.device)
        return out

    def batches(self, token_lists, kind: str = "fwd"):
        t = self.tuning
        if kind == "grad":
            return M.make_batches(token_lists, self.tok.pad_token_id, self.run.device, int(t["max_batch_size"]),
                                  int(t["grad_batch_tokens"]))
        if kind == "ce":
            return M.make_batches(token_lists, self.tok.pad_token_id, self.run.device, int(t["max_batch_size"]),
                                  int(t.get("ce_batch_tokens", 2048)))
        if kind == "gen":
            bs = int(t["gen_batch_size"])
            return M.make_batches(token_lists, self.tok.pad_token_id, self.run.device, bs, bs * 1024)
        return M.make_batches(token_lists, self.tok.pad_token_id, self.run.device, int(t["max_batch_size"]),
                              int(t["max_batch_tokens"]))


# ============================================================================= primitives

def _ex(b) -> torch.Tensor:
    return torch.as_tensor(b.idx, dtype=torch.long)


@torch.inference_mode()
def last_logits_all(mc: MCtx, token_lists, fn):
    """fn(logits [B,V]) -> tuple of [B,...] tensors, gathered into numpy arrays in input order (no hook)."""
    res = None
    for b in mc.batches(token_lists):
        outs = fn(M.last_logits(mc.model, b))
        outs = outs if isinstance(outs, tuple) else (outs,)
        if res is None:
            res = [np.zeros((len(token_lists),) + tuple(o.shape[1:]), dtype=np.float64) for o in outs]
        for r, o in zip(res, outs):
            r[b.idx] = o.double().cpu().numpy()
    return res


@torch.inference_mode()
def generate(mc: MCtx, token_lists, max_new: int, hook: ST.SiteHook | None = None, L: int | None = None,
             min_new: int = 0, ex_ids: np.ndarray | None = None, first_logits_fn=None):
    """Greedy generation (full model). Returns (generated ids cut at the first stop token, first-step outputs).
    The hook (if any) is attached at layer L with prompt masks for the prefill and the decode-step mode."""
    out = [None] * len(token_lists)
    first = None
    stops = M.stop_token_ids(mc.tok)
    stopset = set(stops) | {mc.tok.pad_token_id}
    for b in mc.batches(token_lists, "gen"):
        handle = None
        if hook is not None:
            pm, pos, _, _ = ST.role_masks(b)
            ex = torch.as_tensor(b.idx if ex_ids is None else ex_ids[b.idx], dtype=torch.long)
            hook.set_batch(ex, pm, pos, generating=True)
            handle = ST.attach(mc.layers, L, hook)
        cap = _FirstStep(first_logits_fn) if first_logits_fn is not None else None
        try:
            g = mc.model.generate(input_ids=b.input_ids, attention_mask=b.attention_mask, max_new_tokens=max_new,
                                  min_new_tokens=min_new or None, do_sample=False, temperature=None, top_p=None,
                                  top_k=None, pad_token_id=mc.tok.pad_token_id, eos_token_id=stops,
                                  logits_processor=None if cap is None else LogitsProcessorList([cap]),
                                  return_dict_in_generate=True)
        finally:
            if handle is not None:
                handle.remove()
        new = g.sequences[:, b.S:].tolist()
        for i, row in zip(b.idx, new):
            cut = row
            if min_new < max_new:
                for j, t in enumerate(row):
                    if t in stopset:
                        cut = row[:j]
                        break
            out[int(i)] = cut
        if cap is not None:
            vals = cap.value
            if first is None:
                first = [np.zeros((len(token_lists),) + tuple(v.shape[1:])) for v in vals]
            for f, v in zip(first, vals):
                f[b.idx] = v.double().cpu().numpy()
    return out, first


class _FirstStep:
    """Logits processor that records fn(first-step logits) and changes nothing (greedy decoding uses no other
    processors here, so these are the raw logits at the first response position). Avoids output_logits, which
    keeps every step's full-vocabulary logits."""

    def __init__(self, fn):
        self.fn, self.value = fn, None

    def __call__(self, input_ids, scores):
        if self.value is None:
            v = self.fn(scores.float())
            self.value = v if isinstance(v, tuple) else (v,)
        return scores


def refusal_labels(mc: MCtx, gens: list[list[int]], marker: str | None = None) -> np.ndarray:
    """v1 substring rules. `marker` is the dev-tiny stand-in for a refusal (random weights never refuse)."""
    subs = mc.cfg["tasks"]["refusal"]["substrings"]
    out = np.zeros(len(gens), dtype=bool)
    for i, g in enumerate(gens):
        txt = M.decode(mc.tok, g)                        # never logged
        out[i] = MET.is_refusal(txt, subs) or (marker is not None and marker in txt)
    return out


def score_fn(mc: MCtx, task: str):
    g = mc.gate["tasks"][task]
    if task == "refusal":
        start = g["start_ids"]
        return lambda lg: MET.refusal_scores(lg, start)
    ids = g["answer_ids"]
    return lambda lg: MET.truth_scores(lg, ids)          # logit(pos) - logit(neg), mass on both answers


# ============================================================================= task runner

class TaskRunner:
    """One (model, task, depth): batches, fast-path caches, clean references, resample tables."""

    def __init__(self, mc: MCtx, task: str, depth: float):
        self.mc, self.task, self.depth = mc, task, depth
        self.cfg = mc.cfg
        self.L = mc.layer(depth)
        self.sink_thr = mc.sink_thr(self.L)
        self.refusal = task == "refusal"
        self.ev = mc.tokenized("refusal", "eval") if self.refusal else mc.tokenized(task, "eval")
        self.labels = self.ev.label.to_numpy().astype(int)
        self.max_new = int(self.cfg["refusal_generation"]["max_new_tokens"])
        cont = load_run_file(mc.dir / "prep" / "generic_cont.pt")
        self.gen = mc.tokenized("generic", "cost")
        self.cost_tokens = [list(p) + list(c) for p, c in zip(self.gen.input_ids, cont["cont"])]
        self.cost_plen = np.array([len(p) for p in self.gen.input_ids])
        self.n_cont = int(cont["n"])
        self.sf = score_fn(mc, task)
        self.eval_b = mc.batches(list(self.ev.input_ids)) if not self.refusal else None
        self.cost_b = mc.batches(self.cost_tokens)
        self.fast = False
        self.tokens = 0
        self.seconds = 0.0
        self.gen_steps = 0
        self.gen_seconds = 0.0
        self._tables = {}
        if self.cfg.get("fast_path", True):
            self._prepare_fast()
        self.clean = self._clean_refs()

    # ---------------------------------------------------------------- fast path
    @torch.inference_mode()
    def _prepare_fast(self):
        cap = M.Capture(self.mc.layers, self.L, None)
        try:
            for b in (self.eval_b or []) + self.cost_b:
                self.mc.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                              use_cache=False, logits_to_keep=1)
                b.cache["h"] = cap.h.clone()
        finally:
            cap.remove()
        self.fast = True

    def drop_fast(self, why: str):
        LOG.warning("fast path disabled for %s/%s L%d: %s", self.mc.model_id, self.task, self.L, why)
        for b in (self.eval_b or []) + self.cost_b:
            b.cache.clear()
        self.fast = False

    @torch.inference_mode()
    def _forward(self, b, hook, keep: int, fast: bool | None = None):
        fast = self.fast if fast is None else fast
        model = self.mc.model
        handle = None
        split = M.LayerSplit(model, self.L) if fast else None
        if split is not None:
            split.__enter__()
            split.inject.value = b.cache["h"]
        try:
            if hook is not None:
                handle = ST.attach(self.mc.layers, self.L, hook)
            t0 = time.time()
            out = model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                        use_cache=False, logits_to_keep=keep).logits.float()
            if out.is_cuda:
                torch.cuda.synchronize()
            self.seconds += time.time() - t0
            self.tokens += b.n_tokens
            return out
        finally:
            if handle is not None:
                handle.remove()
            if split is not None:
                split.__exit__(None, None, None)

    def fast_path_check(self, tol: float) -> dict:
        bs = [(b, 1) for b in (self.eval_b or [])[:2]] + [(b, self.n_cont + 1) for b in self.cost_b[:1]]
        d = 0.0
        for b, k in bs:
            d = max(d, float((self._forward(b, None, k, fast=True) - self._forward(b, None, k, fast=False)).abs().max()))
        ok = d < tol
        LOG.info("fast-path equivalence %s/%s L%d: max |dlogit| = %.2e (tol %.0e) -> %s", self.mc.slug, self.task,
                 self.L, d, tol, "enabled" if ok else "DISABLED")
        if not ok:
            self.drop_fast(f"max |dlogit| {d:.2e} >= {tol}")
            self.clean = self._clean_refs()
        return {"max_abs_logit_diff": d, "tol": tol, "enabled": ok}

    # ---------------------------------------------------------------- clean references
    @torch.inference_mode()
    def _clean_refs(self) -> dict:
        out = {"cost_lp": [self._lp(self._forward(b, None, self.n_cont + 1)) for b in self.cost_b]}
        if not self.refusal:
            s, a = self._score_eval(None)
            out.update(score=s, aux=a)
        else:
            gl, sc = self._refusal_gen(None)              # clean generation through the same path as ablated runs
            out.update(score=sc, gen_label=gl, aux=np.zeros(len(self.ev), np.float32))
            agree = float(np.mean(gl == self.ev.clean_refused.to_numpy().astype(bool)))
            out["gate_label_agreement"] = agree
            if agree < 0.98:
                LOG.warning("%s/refusal L%d: clean generation labels agree with the gate's on only %.1f%% of eval prompts",
                            self.mc.slug, self.L, 100 * agree)
        return out

    def save_clean(self, path: Path):
        c = self.clean
        np.savez(path, score=c["score"], aux=c["aux"], labels=self.labels,
                 gen_label=c.get("gen_label", np.zeros(len(self.ev), bool)),
                 eval_pids=self.ev.pid.to_numpy().astype(str), gen_pids=self.gen.pid.to_numpy().astype(str),
                 refusal=np.array(self.refusal))

    @staticmethod
    def _lp(logits):
        return torch.log_softmax(logits, dim=-1)

    # ---------------------------------------------------------------- passes
    def _score_eval(self, hook_maker):
        n = len(self.ev)
        s, a = np.zeros(n, np.float32), np.zeros(n, np.float32)
        for b in self.eval_b:
            hook = None
            if hook_maker is not None:
                hook = hook_maker("eval")
                pm, pos, _, _ = ST.role_masks(b)
                hook.set_batch(_ex(b), pm, pos)
            x, y = self.sf(self._forward(b, hook, 1)[:, -1, :])
            s[b.idx], a[b.idx] = x.cpu().numpy(), y.cpu().numpy()
        return s, a

    def _cost(self, hook_maker):
        n = len(self.gen)
        kl9, kl1, rel = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
        for b, lp_c in zip(self.cost_b, self.clean["cost_lp"]):
            hook = None
            if hook_maker is not None:
                hook = hook_maker("cost")
                pm, pos, gm, gs = ST.role_masks(b, self.cost_plen[b.idx])
                hook.set_batch(_ex(b), pm, pos, gm, gs)
            lp_a = self._lp(self._forward(b, hook, self.n_cont + 1))
            kl = (lp_c.exp() * (lp_c - lp_a)).sum(-1).clamp_min(0.0)         # [B, 9]
            kl9[b.idx] = kl.mean(-1).cpu().numpy()
            kl1[b.idx] = kl[:, 0].cpu().numpy()
            rel[b.idx] = hook.rel_change() if hook is not None else 0.0
        return kl9, kl1, rel

    def _refusal_gen(self, hook_maker):
        tl = list(self.ev.input_ids)
        hook = hook_maker("eval") if hook_maker is not None else None
        t0 = time.time()
        gens, first = generate(self.mc, tl, self.max_new, hook=hook, L=self.L,
                               first_logits_fn=lambda lg: self.sf(lg)[0])
        self.gen_seconds += time.time() - t0
        self.gen_steps += self.max_new * len(self.mc.batches(tl, "gen"))
        marker = self.mc.gate["tasks"]["refusal"].get("dev_tiny_marker")
        return refusal_labels(self.mc, gens, marker), first[0].astype(np.float32)

    # ---------------------------------------------------------------- resample tables (common random numbers)
    def table(self, which: str, role: str, n_src: int) -> torch.Tensor:
        key = (which, role, n_src)
        if key not in self._tables:
            n_ex = len(self.ev) if which == "eval" else len(self.gen)
            if role == ST.PROMPT:
                width = int(max(map(len, self.ev.input_ids if which == "eval" else self.cost_tokens)))
            else:
                width = max(self.max_new, self.n_cont) + 1
            rng = rng_for(self.cfg["seed"], self.mc.model_id, self.task, self.L, "resample-table", which, role)
            self._tables[key] = torch.as_tensor(rng.integers(0, n_src, size=(n_ex, width)), dtype=torch.long)
        return self._tables[key]

    # ---------------------------------------------------------------- one intervention
    def run(self, basis, units: np.ndarray, strategy: str, site: str, means: dict, reservoir: dict | None = None,
            identity: bool = False) -> dict:
        dev = self.mc.run.device
        idx = torch.as_tensor(np.asarray(units), dtype=torch.long, device=dev)
        W = basis.directions(idx).float()
        reader = basis.reader(idx)
        src = {}
        if strategy == "resample" and not identity:
            with torch.inference_mode():
                for role in (ST.PROMPT, ST.GEN):
                    X = (reservoir or {}).get(role)
                    if X is not None:
                        src[role] = torch.cat([reader(X[i:i + 4096].to(dev).float()) for i in range(0, X.shape[0], 4096)])

        def maker(which):
            if identity:
                vals = ST.IdentityValues()
            elif strategy == "mean":
                vg = means.get(ST.GEN)
                vals = ST.MeanValues(means[ST.PROMPT][idx.cpu()], None if vg is None else vg[idx.cpu()])
            elif strategy == "resample":
                tp = self.table(which, ST.PROMPT, src[ST.PROMPT].shape[0])
                tg = self.table(which, ST.GEN, src[ST.GEN].shape[0]) if ST.GEN in src else None
                vals = ST.ResampleValues(src[ST.PROMPT], tp, src.get(ST.GEN), tg)
            else:
                raise ValueError(strategy)
            return ST.SiteHook(W, reader, self.sink_thr, site, vals)

        t0 = time.time()
        out = {}
        if self.refusal:
            out["gen_label"], out["score"] = self._refusal_gen(maker)
        else:
            out["score"], out["aux"] = self._score_eval(maker)
        out["kl9"], out["kl1"], out["rel"] = self._cost(maker)
        out["seconds"] = time.time() - t0
        for k, v in out.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not np.isfinite(v).all():
                raise HarnessError(f"non-finite {k} for {self.mc.slug}/{self.task}/L{self.L}")
        return out
