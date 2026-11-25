"""v3 execution engine: run / model contexts, the cached example prefix, query batches, score functions and the
TaskRunner that executes one intervention (damage + cost) for a (model, task, depth).

Example prefix  Every task prompt is prefix + query, with an identical prefix per task. Its keys and values are
                computed once per (model, task) (`PrefixKV`) and every scoring / gradient pass runs the query tokens
                only, attending to the cached prefix. The cache layers return concat(prefix, new) without storing it,
                so nothing accumulates across layers or batches. The smoke test checks cached == full-prompt scores.
Token roles     query tokens (everything after the prefix) are the prompt role and the only intervened tokens;
                prefix (demo) tokens are never intervened on and never used in statistics; attention sinks (norm >
                10x median at layer L) are excluded as in v2. Generic cost prompts have no prefix: every real token
                except the first is the prompt role, the teacher-forced continuation is never intervened (site S1).
Damage pass     one forward over the eval queries, logits at the last token (fast path: layers L+1.. only).
Cost pass       generic prompt + its clean 8-token continuation, KL(clean || ablated) over the next position and the
                8 continuation positions (as v2).
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
from .engine_v2 import generate  # noqa: F401  (greedy generation is model-agnostic; re-exported)
from .utils import LOG, load_run_file, read_json, rng_for, update_json
from .v3common import TUNING3_PATH, HarnessError, depths_for, eval_split, layer_for, slug


# ============================================================================= contexts

class RunCtx:
    def __init__(self, cfg: dict, run_dir: Path, watchdog=None):
        self.cfg, self.run_dir, self.wd = cfg, Path(run_dir), watchdog
        self.phase, self.dev_tiny = cfg["phase"], bool(cfg.get("dev_tiny"))
        self.device = M.pick_device(allow_cpu=self.phase == "smoke" or self.dev_tiny)
        self.dtype = M.default_dtype(self.device)
        if self.device.type == "cpu" and not self.dev_tiny:
            cfg["keep_models_loaded"] = False             # two fp32 models do not fit next to each other on a laptop
            cfg["batching"].update(max_batch_tokens=4096, grad_batch_tokens=1024, ce_batch_tokens=512,
                                   tune_candidates_tokens=[1024, 2048, 4096])   # fp32 Gemma (10.5 GB) + 256k-vocab logits
        self._m = {}
        from .resources_v3 import Resources

        self.res = Resources(self)                       # competing memory use: wait, fit, back off (speed only)

    @property
    def batch_factor(self) -> float:
        return self.res.factor

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

    @property
    def ref_task(self) -> str:
        p = self.run_dir / "inclusion.json"
        if p.exists():
            t = read_json(p).get("reference_task")
            if t:
                return t
        return self.cfg["reference"]["task"]


class MCtx:
    def __init__(self, run: RunCtx, model_id: str):
        self.run, self.cfg, self.model_id = run, run.cfg, model_id
        self.slug = slug(model_id)
        self.dir = run.d("models", self.slug)
        mc = self.cfg["models"][model_id]
        self.weights = mc.get("weights") or model_id
        self.attn = mc.get("attn_implementation", "sdpa")
        self._model = self._tok = None
        self._sae = {}
        self._prefix = {}
        self.tuning = self._load_tuning()

    # ---------------------------------------------------------------- resources
    def _load_tuning(self) -> dict:
        b = dict(self.cfg["batching"])
        key = ("dev_tiny:" if self.run.dev_tiny else "") + self.model_id
        if TUNING3_PATH.exists():
            t = read_json(TUNING3_PATH).get(key)
            if t and t.get("device") == str(self.run.device):
                for k in ("max_batch_tokens", "grad_batch_tokens"):
                    if t.get(k):
                        b[k] = int(t[k])
                for k in ("weights_gb", "batch_mem_gb"):         # used to fit batches to free memory
                    if t.get(k) is not None:
                        b[k] = float(t[k])
                b["source"] = f"{TUNING3_PATH} ({t.get('tuned_at')})"
        b.setdefault("source", "config/v3.yaml defaults")
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
            self._model = M.load_model(self.weights, self.run.device, self.run.dtype, self.attn)
            impl = getattr(self._model.config, "_attn_implementation", None)
            if impl != self.attn:
                raise HarnessError(f"{self.model_id}: attention implementation {impl!r}, expected {self.attn!r}")
        return self._model

    def unload(self):
        self._model = None
        self._sae = {}
        self._prefix = {}
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
        from .data_v3 import load_tok

        return load_tok(self.dir / "data" / f"{task}_{split}.parquet")

    def prep(self, task: str, L: int) -> dict:
        return load_run_file(self.dir / "prep" / task / f"L{L}.pt")

    def sink_thr(self, L: int) -> float:
        return float(read_json(self.dir / "prep" / "norms.json")[str(L)]["sink_threshold"])

    def sae(self, L: int, which: str = "main"):
        """The SAE used at layer L: which = "main" (the `sae` basis: the included file, which may be the fallback)
        or "extra" (the Gemma 16k file for the width test)."""
        key = (L, which)
        if key not in self._sae:
            from . import sae_v3

            g = self.gate["sae"][str(L)]
            slot = g.get("used") if which == "main" else "extra"
            rec = (g.get("files") or {}).get(slot) if slot else None
            if not rec or not rec.get("info"):
                return None
            s = sae_v3.load(self.cfg["models"][self.model_id]["sae"], rec["info"], self.run.device, self.d_model)
            sae_v3.set_input_norm(s, rec.get("input_norm", "none"), rec.get("mean_sq_norm"))
            self._sae[key] = s
        return self._sae[key]

    def sae_included(self, L: int, which: str = "main") -> bool:
        g = self.gate["sae"].get(str(L), {})
        if which == "main":
            return bool(g.get("used"))
        return bool(g.get("extra_included"))

    def bases(self, L: int, keys=None) -> dict:
        out = {}
        for k in keys or self.cfg["bases"]:
            if k == "sae":
                if self.sae_included(L):
                    out[k] = BZ.SAEBasis(self.sae(L))
            elif k == "sae_x":
                if self.sae_included(L, "extra"):
                    out[k] = BZ.SAEBasis(self.sae(L, "extra"))
                    out[k].key = "sae_x"
            elif k == "pca":
                p = load_run_file(self.dir / "prep" / f"pca_L{L}.pt")
                out[k] = BZ.PCABasis(p["mu"], p["U"], p["evals"]).to(self.run.device)
            elif k == "dims":
                out[k] = BZ.DimsBasis(self.d_model, self.run.device)
        return out

    def batches(self, token_lists, kind: str = "fwd"):
        """Length-sorted, left-padded batches. Every token budget is scaled by the run's batch factor (1 unless other
        jobs' memory use forced a smaller batch; see resources_v3). Batch size never changes what is computed."""
        t = self.tuning
        f = float(getattr(self.run, "batch_factor", 1.0))
        n = {"grad": "grad_batch_tokens", "ce": "ce_batch_tokens"}.get(kind, "max_batch_tokens")
        if kind == "gen":
            bs = max(1, int(int(t["gen_batch_size"]) * f))
            return M.make_batches(token_lists, self.tok.pad_token_id, self.run.device, bs, bs * 1024)
        return M.make_batches(token_lists, self.tok.pad_token_id, self.run.device, int(t["max_batch_size"]),
                              max(128, int(int(t.get(n, 2048)) * f)))

    # ---------------------------------------------------------------- example prefix
    def prefix(self, task: str) -> "PrefixKV":
        if task not in self._prefix:
            d = self.tokenized(task, "discovery")
            p = int(d.n_demo.iloc[0])
            ids = list(d.input_ids.iloc[0][:p])
            for row in d.itertuples():
                if int(row.n_demo) != p or list(row.input_ids[:p]) != ids:
                    raise HarnessError(f"{self.model_id}/{task}: example prefix differs between prompts")
            self._prefix[task] = PrefixKV(self, ids)
        return self._prefix[task]


# ============================================================================= the cached prefix

class PrefixKV:
    """Keys/values of the example prefix at every layer, computed once (batch 1, no autograd state). `attach(B)`
    points each cache layer at expanded views and makes `update` return concat(prefix, new) without storing it."""

    def __init__(self, mc: MCtx, ids: list[int]):
        self.mc, self.ids, self.p = mc, list(ids), len(ids)
        with torch.no_grad():                      # not inference_mode: the views are reused under autograd (prep)
            out = mc.model(input_ids=torch.tensor([self.ids], device=mc.run.device), use_cache=True, logits_to_keep=1)
        self.cache = out.past_key_values
        self.kv = [(layer.keys.detach(), layer.values.detach()) for layer in self.cache.layers]
        self._B = None

    def attach(self, B: int):
        if self._B == B:
            return self.cache
        for layer, (k, v) in zip(self.cache.layers, self.kv):
            ke, ve = k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)
            layer.keys, layer.values = ke, ve

            def upd(key_states, value_states, *a, _k=ke, _v=ve, **kw):
                return torch.cat([_k, key_states], dim=-2), torch.cat([_v, value_states], dim=-2)
            layer.update = upd
        self._B = B
        return self.cache

    def inputs(self, b) -> dict:
        """Model kwargs for a left-padded query batch attending to the prefix."""
        B = int(b.input_ids.shape[0])
        ones = torch.ones(B, self.p, dtype=b.attention_mask.dtype, device=b.attention_mask.device)
        return {"input_ids": b.input_ids, "attention_mask": torch.cat([ones, b.attention_mask], dim=1),
                "position_ids": b.position_ids + self.p, "past_key_values": self.attach(B), "use_cache": True}


def query_lists(df: pd.DataFrame) -> list[list[int]]:
    return [list(x[int(n):]) for x, n in zip(df.input_ids, df.n_demo)]


def query_masks(b) -> tuple[torch.Tensor, torch.Tensor]:
    """Query batch (left-padded): every real token is the prompt role; pos = index within the query."""
    B, S = b.input_ids.shape
    dev = b.input_ids.device
    lens = torch.as_tensor(b.lengths, device=dev)
    pos = torch.arange(S, device=dev).unsqueeze(0) - (S - lens).unsqueeze(1)
    return pos >= 0, pos.clamp(min=0)


# ============================================================================= scores

def score_fn(mc: MCtx, task: str, df: pd.DataFrame):
    """fn(logits [B, V], rows [B] indices into df) -> (logit(pos) - logit(neg), p(pos) + p(neg))."""
    if task == "ioi":
        a = torch.as_tensor(df.a_id.to_numpy(), dtype=torch.long)
        bb = torch.as_tensor(df.b_id.to_numpy(), dtype=torch.long)

        def f(lg, rows):
            r = torch.as_tensor(rows, dtype=torch.long)
            ia, ib = a[r].to(lg.device), bb[r].to(lg.device)
            lp = torch.log_softmax(lg.float(), -1)
            la, lb = lp.gather(1, ia[:, None])[:, 0], lp.gather(1, ib[:, None])[:, 0]
            return la - lb, la.exp() + lb.exp()
        return f
    ids = mc.gate["tasks"][task]["answer_ids"]
    return lambda lg, rows: MET.truth_scores(lg, ids)


# ============================================================================= forward helpers

def forward_query(mc: MCtx, pkv: PrefixKV | None, b, keep: int = 1, hook=None, L: int | None = None):
    """Logits for a query batch with the cached prefix (pkv) or a full-prompt batch (pkv None)."""
    handle = ST.attach(mc.layers, L, hook) if hook is not None else None
    try:
        kw = pkv.inputs(b) if pkv is not None else {"input_ids": b.input_ids, "attention_mask": b.attention_mask,
                                                     "position_ids": b.position_ids, "use_cache": False}
        return mc.model(**kw, logits_to_keep=keep).logits
    finally:
        if handle is not None:
            handle.remove()


@torch.inference_mode()
def scores_all(mc: MCtx, task: str, split: str, use_prefix: bool = True, df: pd.DataFrame | None = None):
    """Clean (score, coherence) for every prompt of a split, input order."""
    df = mc.tokenized(task, split) if df is None else df
    sf = score_fn(mc, task, df)
    s, c = np.zeros(len(df)), np.zeros(len(df))
    if use_prefix:
        pkv = mc.prefix(task)
        for b in mc.batches(query_lists(df)):
            x, y = sf(forward_query(mc, pkv, b)[:, -1, :].float(), b.idx)
            s[b.idx], c[b.idx] = x.double().cpu().numpy(), y.double().cpu().numpy()
    else:
        for b in mc.batches(list(df.input_ids)):
            x, y = sf(forward_query(mc, None, b)[:, -1, :].float(), b.idx)
            s[b.idx], c[b.idx] = x.double().cpu().numpy(), y.double().cpu().numpy()
    return s, c


# ============================================================================= task runner

class TaskRunner:
    """One (model, task, depth): eval query batches with the cached prefix, fast-path caches, clean references,
    resample tables."""

    def __init__(self, mc: MCtx, task: str, depth: float, split: str | None = None):
        self.mc, self.task, self.depth = mc, task, depth
        self.cfg = mc.cfg
        self.L = mc.layer(depth)
        self.sink_thr = mc.sink_thr(self.L)
        self.split = split or eval_split(self.cfg)
        self.ev = mc.tokenized(task, self.split)
        self.labels = self.ev.label.to_numpy().astype(int)
        self.pkv = mc.prefix(task) if self.cfg.get("prefix_cache", True) else None
        cont = load_run_file(mc.dir / "prep" / "generic_cont.pt")
        self.gen = mc.tokenized("generic", "cost")
        self.cost_tokens = [list(p) + list(c) for p, c in zip(self.gen.input_ids, cont["cont"])]
        self.cost_plen = np.array([len(p) for p in self.gen.input_ids])
        self.n_cont = int(cont["n"])
        self.sf = score_fn(mc, task, self.ev)
        self.eval_tokens = query_lists(self.ev) if self.pkv is not None else list(self.ev.input_ids)
        self.n_demo = int(self.ev.n_demo.iloc[0])
        self.eval_b = mc.batches(self.eval_tokens)
        self.cost_b = mc.batches(self.cost_tokens)
        self.fast = False
        self.seconds = 0.0
        self._tables = {}
        if self.cfg.get("fast_path", True):
            self._prepare_fast()
        self.clean = self._clean_refs()

    # ---------------------------------------------------------------- masks
    def _eval_masks(self, b):
        if self.pkv is not None:
            return query_masks(b)
        pm, pos, _, _ = ST.role_masks(b)                 # full prompt: query = positions >= n_demo
        return pm & (pos >= self.n_demo), (pos - self.n_demo).clamp(min=0)

    # ---------------------------------------------------------------- fast path
    @torch.inference_mode()
    def _prepare_fast(self):
        cap = M.Capture(self.mc.layers, self.L, None)
        try:
            for b in self.eval_b:
                forward_query(self.mc, self.pkv, b)
                b.cache["h"] = cap.h.clone()
            for b in self.cost_b:
                forward_query(self.mc, None, b, keep=1)
                b.cache["h"] = cap.h.clone()
        finally:
            cap.remove()
        self.fast = True

    def drop_fast(self, why: str):
        LOG.warning("fast path disabled for %s/%s L%d: %s", self.mc.model_id, self.task, self.L, why)
        for b in self.eval_b + self.cost_b:
            b.cache.clear()
        self.fast = False

    @torch.inference_mode()
    def _forward(self, b, hook, keep: int, prefix: bool, fast: bool | None = None):
        fast = self.fast if fast is None else fast
        split = M.LayerSplit(self.mc.model, self.L) if fast else None
        if split is not None:
            split.__enter__()
            split.inject.value = b.cache["h"]
        try:
            t0 = time.time()
            out = forward_query(self.mc, self.pkv if prefix else None, b, keep, hook, self.L).float()
            if out.is_cuda:
                torch.cuda.synchronize()
            self.seconds += time.time() - t0
            return out
        finally:
            if split is not None:
                split.__exit__(None, None, None)

    def fast_path_check(self, tol: float) -> dict:
        d = 0.0
        for b in self.eval_b[:2]:
            d = max(d, float((self._forward(b, None, 1, True, fast=True)
                              - self._forward(b, None, 1, True, fast=False)).abs().max()))
        for b in self.cost_b[:1]:
            d = max(d, float((self._forward(b, None, self.n_cont + 1, False, fast=True)
                              - self._forward(b, None, self.n_cont + 1, False, fast=False)).abs().max()))
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
        s, a = self._score_eval(None)
        return {"cost_lp": [torch.log_softmax(self._forward(b, None, self.n_cont + 1, False), -1) for b in self.cost_b],
                "score": s, "aux": a}

    def save_clean(self, path: Path):
        c = self.clean
        np.savez(path, score=c["score"], aux=c["aux"], labels=self.labels,
                 gen_label=np.zeros(len(self.ev), bool), eval_pids=self.ev.pid.to_numpy().astype(str),
                 gen_pids=self.gen.pid.to_numpy().astype(str), refusal=np.array(False))

    # ---------------------------------------------------------------- passes
    def _score_eval(self, hook_maker):
        n = len(self.ev)
        s, a = np.zeros(n, np.float32), np.zeros(n, np.float32)
        for b in self.eval_b:
            hook = None
            if hook_maker is not None:
                hook = hook_maker("eval")
                pm, pos = self._eval_masks(b)
                hook.set_batch(torch.as_tensor(b.idx, dtype=torch.long), pm, pos)
            x, y = self.sf(self._forward(b, hook, 1, self.pkv is not None)[:, -1, :], b.idx)
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
                hook.set_batch(torch.as_tensor(b.idx, dtype=torch.long), pm, pos, gm, gs)
            lp_a = torch.log_softmax(self._forward(b, hook, self.n_cont + 1, False), -1)
            kl = (lp_c.exp() * (lp_c - lp_a)).sum(-1).clamp_min(0.0)
            kl9[b.idx] = kl.mean(-1).cpu().numpy()
            kl1[b.idx] = kl[:, 0].cpu().numpy()
            rel[b.idx] = hook.rel_change() if hook is not None else 0.0
        return kl9, kl1, rel

    # ---------------------------------------------------------------- resample tables (common random numbers)
    def table(self, which: str, n_src: int) -> torch.Tensor:
        key = (which, n_src)
        if key not in self._tables:
            n_ex = len(self.ev) if which == "eval" else len(self.gen)
            width = int(max(map(len, self.eval_tokens if which == "eval" else self.cost_tokens))) + 1
            rng = rng_for(self.cfg["seed"], self.mc.model_id, self.task, self.L, "resample-table-v3", which)
            self._tables[key] = torch.as_tensor(rng.integers(0, n_src, size=(n_ex, width)), dtype=torch.long)
        return self._tables[key]

    # ---------------------------------------------------------------- one intervention
    def run(self, basis, units: np.ndarray, strategy: str, means: dict, reservoir: torch.Tensor | None = None,
            identity: bool = False, values=None) -> dict:
        dev = self.mc.run.device
        idx = torch.as_tensor(np.asarray(units), dtype=torch.long, device=dev)
        W = basis.directions(idx).float()
        reader = basis.reader(idx)
        src = None
        if strategy == "resample" and not identity:
            if reservoir is None:
                raise HarnessError("resample strategy requested where no reservoir was built")
            with torch.inference_mode():
                src = torch.cat([reader(reservoir[i:i + 4096].to(dev).float()) for i in range(0, reservoir.shape[0], 4096)])

        def maker(which):
            if values is not None:
                vals = values
            elif identity:
                vals = ST.IdentityValues()
            elif strategy == "mean":
                vals = ST.MeanValues(means[ST.PROMPT][idx.cpu()])
            elif strategy == "resample":
                vals = ST.ResampleValues(src, self.table(which, src.shape[0]))
            else:
                raise ValueError(strategy)
            return ST.SiteHook(W, reader, self.sink_thr, "S1", vals)

        t0 = time.time()
        out = {}
        out["score"], out["aux"] = self._score_eval(maker)
        out["kl9"], out["kl1"], out["rel"] = self._cost(maker)
        out["seconds"] = time.time() - t0
        for k, v in out.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not np.isfinite(v).all():
                raise HarnessError(f"non-finite {k} for {self.mc.slug}/{self.task}/L{self.L}")
        return out
