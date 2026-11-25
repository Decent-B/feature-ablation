"""Model loading, batching, the layer-L hook point, the fast path and generation.

Conventions
- Prompts are left-padded and position ids are derived from the attention mask (as
  `generate` does), so a prompt's activations do not depend on its batch.
- With left padding the T post-instruction template tokens (the intervention positions)
  are always the last T columns of every row; offset t in [0, T) is column S - T + t, and
  the final prompt token (first response position) is column S - 1.
- The hook point is the output of decoder layer L (`resid_post`), the site where both SAE
  releases were trained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .utils import LOG


# --------------------------------------------------------------------------- device / loading

def pick_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        LOG.warning("torch.cuda.is_available() is False: running on CPU. The CPU path exists only "
                    "for the smoke check; GPU-only paths are untested here.")
        return torch.device("cpu")
    raise SystemExit("ERROR: torch.cuda.is_available() is False. The pilot needs the GB10 GPU "
                     "(a CPU fallback exists only for `--smoke`). Check that a cu130 torch wheel is "
                     "installed and the NVIDIA driver is visible.")


def default_dtype(device: torch.device) -> torch.dtype:
    # bf16 on the GPU per spec; fp32 on CPU because bf16 matmuls are emulated (very slow)
    # on CPUs without AVX512-BF16/AMX.
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id: str, device: torch.device, dtype: torch.dtype, attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    LOG.info("loading %s (%s, %s, attn=%s)", model_id, device, dtype, attn)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def decoder_layers(model) -> nn.ModuleList:
    n = text_config(model).num_hidden_layers
    cands = []
    base = getattr(model, "model", None)
    for obj in (base, getattr(base, "language_model", None), getattr(model, "language_model", None)):
        if obj is not None and isinstance(getattr(obj, "layers", None), nn.ModuleList):
            cands.append(obj.layers)
    for layers in cands:
        if len(layers) == n:
            return layers
    raise RuntimeError("could not find the decoder layer stack on this model")


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = 0
    seen = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) or name.endswith("lm_head"):
            w = getattr(mod, "weight", None)
            if w is not None and id(w) not in seen:
                seen.add(id(w))
                emb += w.numel()
    return {"total": int(total), "embedding_and_head": int(emb), "non_embedding": int(total - emb)}


def linear_attn_impl_report(model) -> dict:
    """Which Gated DeltaNet implementation loaded (hybrid Qwen3.5 only)."""
    tc = text_config(model)
    if "linear_attention" not in (getattr(tc, "layer_types", None) or []):
        return {"hybrid": False}
    rep = {"hybrid": True}
    try:
        import importlib

        mod = importlib.import_module(type(decoder_layers(model)[0]).__module__)
        for name in ("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule", "causal_conv1d_fn",
                     "is_fast_path_available"):
            if hasattr(mod, name):
                obj = getattr(mod, name)
                rep[name] = getattr(obj, "__module__", str(obj)) + "." + getattr(obj, "__name__", "")
        for pkg in ("fla", "causal_conv1d"):
            try:
                importlib.import_module(pkg)
                rep[f"pkg_{pkg}"] = "importable"
            except Exception as e:  # noqa: BLE001
                rep[f"pkg_{pkg}"] = f"missing ({type(e).__name__})"
        rep["fused_kernels"] = rep.get("pkg_fla") == "importable"
    except Exception as e:  # noqa: BLE001
        rep["error"] = repr(e)
    return rep


# --------------------------------------------------------------------------- batching

@dataclass
class Batch:
    idx: np.ndarray                 # indices into the prompt list
    input_ids: torch.Tensor         # [B, S]
    attention_mask: torch.Tensor    # [B, S]
    position_ids: torch.Tensor      # [B, S]
    lengths: np.ndarray             # real lengths
    cache: dict = field(default_factory=dict)   # per-batch tensors (fast-path hidden states, ...)

    @property
    def S(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def n_tokens(self) -> int:
        return int(self.lengths.sum())


def make_batches(token_lists: list[list[int]], pad_id: int, device: torch.device,
                 batch_size: int, max_batch_tokens: int) -> list[Batch]:
    """Sort by length (less padding) and left-pad."""
    lens = np.array([len(t) for t in token_lists])
    order = np.argsort(lens, kind="stable")
    groups, cur = [], []
    for i in order:
        L = int(lens[i])
        if cur and (len(cur) + 1 > batch_size or (len(cur) + 1) * L > max_batch_tokens):
            groups.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        S = int(max(lens[g]))
        ids = torch.full((len(g), S), pad_id, dtype=torch.long)
        mask = torch.zeros((len(g), S), dtype=torch.long)
        for r, i in enumerate(g):
            t = token_lists[i]
            ids[r, S - len(t):] = torch.tensor(t, dtype=torch.long)
            mask[r, S - len(t):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out.append(Batch(np.array(g), ids.to(device), mask.to(device), pos.to(device), lens[g]))
    return out


def _split_output(output):
    if isinstance(output, tuple):
        return output[0], (lambda h: (h,) + tuple(output[1:]))
    return output, (lambda h: h)


# --------------------------------------------------------------------------- forward helpers

def last_logits(model, batch: Batch) -> torch.Tensor:
    """fp32 logits at the first response position (last prompt token), [B, V]."""
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                position_ids=batch.position_ids, use_cache=False, logits_to_keep=1)
    return out.logits[:, -1, :].float()


class Capture:
    """Forward hook capturing layer-L output (and optionally the MLP down_proj input)."""

    def __init__(self, layers: nn.ModuleList, L: int, last_n: int | None, mlp: bool = False):
        self.L, self.last_n, self.mlp = L, last_n, mlp
        self.h = None
        self.m = None
        self._handles = [layers[L].register_forward_hook(self._hook)]
        if mlp:
            self._handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(self._pre))

    def _hook(self, module, args, output):
        h, _ = _split_output(output)
        self.h = h if self.last_n is None else h[:, -self.last_n:, :]
        self.h = self.h.detach()

    def _pre(self, module, args):
        a = args[0]
        self.m = (a if self.last_n is None else a[:, -self.last_n:, :]).detach()

    def remove(self):
        for h in self._handles:
            h.remove()


# --------------------------------------------------------------------------- ablation hook

class AblationHook:
    """x' = x + sum_{i in S} (v_i - r_i(x)) w_i at the last T positions of layer-L output.

    reader(x [B,T,d] fp32, mlp [B,T,f] fp32|None) -> r [B,T,k]; W [k, d] fp32.
    Values: `values_static` [T, k], or `values_src` [N_src, T, k] indexed per example by
    `src_index[batch.idx]`. Deltas are computed in fp32 and cast back to the model dtype.
    The hook acts only on the pass whose sequence length equals `expected_len` (the
    prompt/prefill pass), never on single-token decode steps.
    """

    def __init__(self, T: int, W: torch.Tensor, reader: Callable, values_static=None,
                 values_src=None, src_index=None, needs_mlp: bool = False):
        self.T = T
        self.W = W
        self.reader = reader
        self.values_static = values_static
        self.values_src = values_src
        self.src_index = None if src_index is None else torch.as_tensor(src_index, dtype=torch.long)
        self.needs_mlp = needs_mlp
        self.expected_len: int | None = None
        self.batch_idx: np.ndarray | None = None
        self.mlp_cache: torch.Tensor | None = None   # set by runner (fast path) or pre-hook
        self.rel_change: torch.Tensor | None = None  # [B] mean over T of ||dx||/||x||
        self.fired = 0

    def values_for_batch(self, B: int, device) -> torch.Tensor:
        if self.values_src is not None:
            src = self.src_index[torch.as_tensor(self.batch_idx, dtype=torch.long)]
            return self.values_src[src.to(self.values_src.device)].to(device)
        return self.values_static.to(device).unsqueeze(0).expand(B, -1, -1)

    def __call__(self, module, args, output):
        h, rebuild = _split_output(output)
        if self.expected_len is not None and h.shape[1] != self.expected_len:
            return output
        T = self.T
        x = h[:, -T:, :].float()
        m = self.mlp_cache.float() if self.needs_mlp else None
        r = self.reader(x, m)
        v = self.values_for_batch(h.shape[0], h.device)
        delta = (v - r) @ self.W
        self.rel_change = (delta.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).mean(-1)
        h2 = h.clone()
        h2[:, -T:, :] = (x + delta).to(h.dtype)
        self.fired += 1
        return rebuild(h2)


class _MlpPre:
    def __init__(self, hook: AblationHook, T: int):
        self.hook, self.T = hook, T

    def __call__(self, module, args):
        self.hook.mlp_cache = args[0][:, -self.T:, :].detach()


def attach_hook(layers: nn.ModuleList, L: int, hook: AblationHook, fast_path: bool) -> list:
    handles = [layers[L].register_forward_hook(hook)]
    if hook.needs_mlp and not fast_path:
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(_MlpPre(hook, hook.T)))
    return handles


# --------------------------------------------------------------------------- fast path

class _Skip(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class _Inject(nn.Module):
    def __init__(self):
        super().__init__()
        self.value: torch.Tensor | None = None

    def forward(self, hidden_states, *args, **kwargs):
        return self.value


class LayerSplit:
    """Run only layers L+1..end: layers 0..L-1 become pass-throughs and layer L emits a
    cached layer-L output. The model's own forward still builds masks and rotary
    embeddings, so nothing architecture-specific is reimplemented. Hooks registered on
    the module at index L while this context is active see the cached hidden state."""

    def __init__(self, model, L: int):
        self.layers = decoder_layers(model)
        self.L = L
        self.inject = _Inject()
        self._orig = None

    def __enter__(self):
        self._orig = [self.layers[i] for i in range(self.L + 1)]
        for i in range(self.L):
            self.layers[i] = _Skip()
        self.layers[self.L] = self.inject
        return self

    def __exit__(self, *exc):
        for i, m in enumerate(self._orig):
            self.layers[i] = m
        self._orig = None


# --------------------------------------------------------------------------- gradients

def grad_at_layer(model, layers, L: int, batch: Batch, T: int,
                  score_fn: Callable[[torch.Tensor], torch.Tensor], mlp: bool = False):
    """Returns (x [B,T,d] fp32, g = dS/dx [B,T,d] fp32, mlp_hidden [B,T,f] | None).
    One backward per batch of independent examples gives per-example gradients."""
    store = {}

    def hook(module, args, output):
        h, rebuild = _split_output(output)
        h = h.detach().requires_grad_(True)
        store["h"] = h
        return rebuild(h)

    def pre(module, args):
        store["m"] = args[0][:, -T:, :].detach().float()

    handles = [layers[L].register_forward_hook(hook)]
    if mlp:
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(pre))
    try:
        with torch.enable_grad():
            logits = last_logits(model, batch)
            S = score_fn(logits)
            S.sum().backward()
    finally:
        for h in handles:
            h.remove()
    h = store["h"]
    return h[:, -T:, :].detach().float(), h.grad[:, -T:, :].float(), store.get("m")


# --------------------------------------------------------------------------- generation

def stop_token_ids(tok) -> list[int]:
    ids = set()
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(ids)


@torch.inference_mode()
def generate_greedy(model, tok, batch: Batch, max_new_tokens: int) -> list[list[int]]:
    out = model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                         max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
                         eos_token_id=stop_token_ids(tok))
    new = out[:, batch.S:].tolist()
    stops = set(stop_token_ids(tok)) | {tok.pad_token_id}
    res = []
    for row in new:
        cut = [t for t in row]
        for j, t in enumerate(row):
            if t in stops:
                cut = row[:j]
                break
        res.append(cut)
    return res


def decode(tok, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x < 0, numerically stable."""
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))
