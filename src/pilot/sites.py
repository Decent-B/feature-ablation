"""v2 intervention sites: one hook that edits layer-L states on prompt and generated tokens.

x' = x + sum_{i in S} (v_i - r_i(x)) w_i at every intervened token (deltas in fp32, cast back).

Token roles
  prompt     every real prompt token except the first (chat-template tokens and the empty think block included)
  generated  every token the model produces (decode steps, or teacher-forced continuation tokens)
  excluded   the first token, attention sinks (||x|| at layer L > sink_thr = 10x median), and padding
Sites
  S1  prompt tokens only (decode steps pass through untouched)
  S2  prompt tokens + every generated token; at a decode step the new token's layer-L state is edited before
      layers L+1.. run, so the edited state is what they write to the KV / recurrent cache.
Values (one per unit and role, no per-position values)
  MeanValues      pooled per-role means
  ResampleValues  r_i copied jointly from one source token of the same role (seeded table per example/position)
  IdentityValues  v = r(x) (exact no-op; used by the identity and zero-cost checks)
  ExplicitValues  per (example, position) values for the prompt role (cached-activation identity check)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .models import _split_output

PROMPT, GEN = "prompt", "generated"


# ----------------------------------------------------------------------------- values

class MeanValues:
    kind = "mean"

    def __init__(self, v_prompt: torch.Tensor, v_gen: torch.Tensor | None = None):
        self.v = {PROMPT: v_prompt.float(), GEN: None if v_gen is None else v_gen.float()}

    def get(self, role, ex, pos, r):
        v = self.v[role]
        if v is None:
            raise RuntimeError(f"no {role}-role mean values for this intervention")
        return v.to(r.device).unsqueeze(0).expand(r.shape[0], -1)


class ResampleValues:
    kind = "resample"

    def __init__(self, src_prompt: torch.Tensor, assign_prompt: torch.Tensor,
                 src_gen: torch.Tensor | None = None, assign_gen: torch.Tensor | None = None):
        """src_*: [N_src, k] readouts of reservoir tokens; assign_*: [n_examples, max_pos] indices into src."""
        self.src = {PROMPT: src_prompt.float(), GEN: None if src_gen is None else src_gen.float()}
        self.assign = {PROMPT: assign_prompt, GEN: assign_gen}

    def get(self, role, ex, pos, r):
        src, a = self.src[role], self.assign[role]
        if src is None or a is None:
            raise RuntimeError(f"no {role}-role resample sources for this intervention")
        a = a.to(r.device)
        pos = pos.clamp(max=a.shape[1] - 1)
        return src.to(r.device)[a[ex, pos]]


class IdentityValues:
    kind = "identity"

    def get(self, role, ex, pos, r):
        return r


class ExplicitValues:
    kind = "explicit"

    def __init__(self, table: torch.Tensor):
        self.table = table            # [n_examples, max_pos, k] (prompt role)

    def get(self, role, ex, pos, r):
        return self.table.to(r.device)[ex, pos].float()


# ----------------------------------------------------------------------------- hook

class SiteHook:
    def __init__(self, W: torch.Tensor, reader: Callable, sink_thr: float, site: str, values):
        assert site in ("S1", "S2"), site
        self.W = W.float()
        self.reader = reader
        self.sink_thr = float(sink_thr)
        self.site = site
        self.values = values
        self.fired = 0
        self.n_edit = {PROMPT: 0, GEN: 0}
        self.n_sink = {PROMPT: 0, GEN: 0}
        self.chunk = 8192
        self._b = None

    # per batch ----------------------------------------------------------------
    def set_batch(self, ex: torch.Tensor, prompt_mask: torch.Tensor, pos: torch.Tensor,
                  gen_mask: torch.Tensor | None = None, gen_step: torch.Tensor | None = None,
                  generating: bool = False):
        """ex [B] example indices; prompt_mask/pos/gen_mask/gen_step [B, S] for the full-sequence pass."""
        self._b = {"ex": ex, "pm": prompt_mask, "pos": pos, "gm": gen_mask, "gs": gen_step,
                   "S": int(prompt_mask.shape[1]), "gen": generating, "step": 0,
                   "rel_sum": torch.zeros(len(ex), device=prompt_mask.device),
                   "rel_cnt": torch.zeros(len(ex), device=prompt_mask.device)}

    def rel_change(self) -> np.ndarray:
        b = self._b
        return (b["rel_sum"] / b["rel_cnt"].clamp_min(1)).cpu().numpy()

    # core ---------------------------------------------------------------------
    def _apply(self, h, h2, mask, role, ex_rows, pos):
        norms = torch.linalg.vector_norm(h.float(), dim=-1)
        sink = mask & (norms > self.sink_thr)
        mask = mask & ~sink
        self.n_sink[role] += int(sink.sum())
        BI, TI = mask.nonzero(as_tuple=True)
        b = self._b
        W = self.W.to(h.device)
        for s in range(0, BI.numel(), self.chunk):         # bounded memory for wide SAE readouts
            bi, ti = BI[s:s + self.chunk], TI[s:s + self.chunk]
            xs = h[bi, ti].float()
            r = self.reader(xs)
            v = self.values.get(role, ex_rows[bi], pos[bi, ti], r)
            delta = (v - r) @ W
            h2[bi, ti] = (xs + delta).to(h2.dtype)
            rel = delta.norm(dim=-1) / xs.norm(dim=-1).clamp_min(1e-6)
            b["rel_sum"].index_add_(0, bi, rel)
            b["rel_cnt"].index_add_(0, bi, torch.ones_like(rel))
        self.n_edit[role] += int(BI.numel())

    def __call__(self, module, args, output):
        h, rebuild = _split_output(output)
        b = self._b
        if b is None:
            raise RuntimeError("SiteHook fired without set_batch()")
        B, S, _ = h.shape
        h2 = h.clone()
        if S == b["S"]:                                     # scoring / teacher-forced / prefill pass
            self._apply(h, h2, b["pm"].to(h.device), PROMPT, b["ex"].to(h.device), b["pos"].to(h.device))
            if self.site == "S2" and b["gm"] is not None:
                self._apply(h, h2, b["gm"].to(h.device), GEN, b["ex"].to(h.device), b["gs"].to(h.device))
        elif S == 1 and b["gen"]:                           # one decode step: the newly generated token
            step = b["step"]
            b["step"] += 1
            if self.site != "S2":
                return output
            mask = torch.ones(B, 1, dtype=torch.bool, device=h.device)
            pos = torch.full((B, 1), step, dtype=torch.long, device=h.device)
            self._apply(h, h2, mask, GEN, b["ex"].to(h.device), pos)
        else:
            raise RuntimeError(f"SiteHook: unexpected sequence length {S} (batch S={b['S']}, generating={b['gen']})")
        self.fired += 1
        return rebuild(h2)


def attach(layers: nn.ModuleList, L: int, hook: SiteHook):
    return layers[L].register_forward_hook(hook)


# ----------------------------------------------------------------------------- role masks

def role_masks(batch, prompt_lens: np.ndarray | None = None):
    """For a left-padded batch: prompt mask (real, not the first token, before the continuation), token
    position within the sequence, generated mask and generated step (teacher-forced continuations)."""
    B, S = batch.input_ids.shape
    dev = batch.input_ids.device
    lens = torch.as_tensor(batch.lengths, device=dev)
    col = torch.arange(S, device=dev).unsqueeze(0)
    pos = col - (S - lens).unsqueeze(1)                     # token index within the real sequence
    real = pos >= 0
    plen = lens if prompt_lens is None else torch.as_tensor(prompt_lens, device=dev)
    prompt = real & (pos >= 1) & (pos < plen.unsqueeze(1))
    gen = real & (pos >= plen.unsqueeze(1)) if prompt_lens is not None else None
    gstep = (pos - plen.unsqueeze(1)).clamp(min=0) if prompt_lens is not None else None
    return prompt, pos.clamp(min=0), gen, gstep


# ----------------------------------------------------------------------------- capture at several layers

class MultiCapture:
    """Forward hooks storing detached layer outputs at several layers (inference)."""

    def __init__(self, layers: nn.ModuleList, Ls: list[int]):
        self.h = {}
        self._handles = [layers[L].register_forward_hook(self._mk(L)) for L in Ls]

    def _mk(self, L):
        def f(module, args, output):
            h, _ = _split_output(output)
            self.h[L] = h.detach()
        return f

    def remove(self):
        for h in self._handles:
            h.remove()


class GradCapture:
    """dS/dx at several layers from one backward pass. The lowest layer's output becomes a leaf that
    requires grad; higher layers keep their graph (so the gradient at the lowest layer is the total
    derivative) and record their gradient with a tensor hook."""

    def __init__(self, layers: nn.ModuleList, Ls: list[int]):
        self.Ls = sorted(Ls)
        self.x, self.g = {}, {}
        self._handles = [layers[L].register_forward_hook(self._mk(L)) for L in self.Ls]

    def _mk(self, L):
        def f(module, args, output):
            h, rebuild = _split_output(output)
            if L == self.Ls[0]:
                h = h.detach().requires_grad_(True)
                self.x[L] = h
                h.register_hook(lambda g, L=L: self.g.__setitem__(L, g))
                return rebuild(h)
            self.x[L] = h
            h.register_hook(lambda g, L=L: self.g.__setitem__(L, g))
            return output
        return f

    def remove(self):
        for h in self._handles:
            h.remove()
