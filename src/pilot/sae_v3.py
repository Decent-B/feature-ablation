"""v3 SAEs: Gemma Scope (google/gemma-scope-2b-pt-res) and Qwen-Scope (Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-*).

Both releases are applied exactly as their reference code does (checked 2026-11-25):

Gemma Scope, JumpReLU (SAELens `JumpReLUSAE` + `gemma_2_sae_huggingface_loader`: apply_b_dec_to_input=False,
normalize_activations=None, hook blocks.{N}.hook_resid_post)
    params.npz: W_enc [d, n], W_dec [n, d] (unit-norm rows), b_enc [n], b_dec [d], threshold [n] (one per latent)
    pre = x @ W_enc + b_enc            (no b_dec subtraction)
    f   = ReLU(pre) * (pre > threshold)
    x^  = f @ W_dec + b_dec
Qwen-Scope, TopK (Qwen's app.py `compute_sae_features` / `topk_relu`; SAELens qwen-scope loader:
apply_b_dec_to_input=False, normalize_activations none, hook blocks.{N}.hook_resid_post)
    layer{N}.sae.pt: W_enc [n, d], W_dec [d, n] (columns are decoder directions), b_enc [n], b_dec [d];
    k from the repo's config.json
    pre = x @ W_enc.T + b_enc          (no b_dec subtraction)
    f   = scatter(topk(ReLU(pre), k))  (a latent's value depends on all latents, so a subset is read by
                                        computing every pre-activation first)
    x^  = f @ W_dec.T + b_dec
Both: the file for layer N is the output of decoder block N (0-indexed), i.e. `model.model.layers[N]` output.
Internally both are stored as W_enc [n, d], W_dec [n, d] (rows = latents), fp32.

Input scaling: both model cards feed raw activations. The v2 lesson (REINS) is to measure instead of trusting
the card, so `norm_factor` can be set by `set_input_norm` ("none" | "rms_d"); the smoke test picks per layer.

File choice (mechanical, recorded in the manifest):
  gemmascope  main: width `width` (65k) at the average L0 closest to `l0_target` (100), ties to the lower L0;
              extra: width `extra_width` (16k) at the L0 closest to the main file's L0 (ties lower). The extra file is
              the fallback and the second width for the width test (D6).
  qwenscope   main: `repo` (TopK 100); fallback: `fallback_repo` (TopK 50); file layer{N}.sae.pt.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from .utils import LOG, read_json

_GS_RE = re.compile(r"^layer_(?P<layer>\d+)/width_(?P<width>\w+?)/average_l0_(?P<l0>\d+)/params\.npz$")
_QS_RE = re.compile(r"^layer(?P<layer>\d+)\.sae\.pt$")


class SAEError(RuntimeError):
    pass


# ============================================================================= the SAE object

class SAE3:
    """encode / decode / directions in raw activation space; fp32 on one device. Same interface as v1 `sae.SAE`
    (attributes n, d, W_dec, norm_factor; encode(x, idx); decode(f); directions(idx))."""

    def __init__(self, kind: str, W_enc, b_enc, W_dec, b_dec, threshold=None, k: int | None = None,
                 meta: dict | None = None):
        if kind not in ("jumprelu", "topk"):
            raise SAEError(f"unknown SAE kind {kind}")
        self.kind = kind
        self.W_enc = W_enc.float().contiguous()       # [n, d]
        self.b_enc = b_enc.float().contiguous()       # [n]
        self.W_dec = W_dec.float().contiguous()       # [n, d]
        self.b_dec = b_dec.float().contiguous()       # [d]
        self.threshold = None if threshold is None else threshold.float().contiguous()   # [n] (jumprelu)
        self.k = None if k is None else int(k)
        self.norm_factor = 1.0
        self.n, self.d = self.W_enc.shape
        self.meta = meta or {}
        if self.W_dec.shape != (self.n, self.d) or self.b_enc.shape != (self.n,) or self.b_dec.shape != (self.d,):
            raise SAEError(f"inconsistent SAE shapes: W_enc {tuple(self.W_enc.shape)}, W_dec {tuple(self.W_dec.shape)}, "
                           f"b_enc {tuple(self.b_enc.shape)}, b_dec {tuple(self.b_dec.shape)}")
        if kind == "jumprelu" and (self.threshold is None or self.threshold.shape != (self.n,)):
            raise SAEError("JumpReLU SAE needs a per-latent threshold vector [n]")
        if kind == "topk" and not (self.k and 0 < self.k <= self.n):
            raise SAEError(f"TopK SAE needs 0 < k <= n (k={self.k}, n={self.n})")

    def to(self, device) -> "SAE3":
        for name in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        return self

    # ------------------------------------------------------------------ core (x in raw activation space)
    def pre(self, x, idx=None):
        xs = x.float() / self.norm_factor
        if idx is None:
            return xs @ self.W_enc.T + self.b_enc
        return xs @ self.W_enc[idx].T + self.b_enc[idx]

    def encode(self, x, idx=None, chunk: int = 2048):
        if self.kind == "jumprelu":
            p = self.pre(x, idx)
            thr = self.threshold if idx is None else self.threshold[idx]
            return torch.relu(p) * (p > thr)
        # TopK: membership depends on every latent, so always compute all pre-activations
        shape = x.shape[:-1]
        X = x.reshape(-1, x.shape[-1])
        outs = []
        for i in range(0, max(X.shape[0], 1), chunk):
            p = torch.relu(self.pre(X[i:i + chunk]))
            vals, ind = p.topk(self.k, dim=-1)
            f = torch.zeros_like(p).scatter_(-1, ind, vals)
            outs.append(f if idx is None else f[:, idx])
        f = torch.cat(outs) if outs else torch.zeros(0, self.n if idx is None else len(idx), device=x.device)
        return f.reshape(*shape, f.shape[-1])

    def decode(self, f):
        return (f.float() @ self.W_dec + self.b_dec) * self.norm_factor

    def directions(self, idx=None):
        W = self.W_dec if idx is None else self.W_dec[idx]
        return W * self.norm_factor


# ============================================================================= files

def _files(sae_cfg: dict, repo_key: str = "repo") -> list[str]:
    if sae_cfg.get("local_dir"):
        root = Path(sae_cfg["local_dir"]) / sae_cfg.get(f"local_{repo_key}", "")
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(sae_cfg[repo_key])


def fetch(sae_cfg: dict, rel: str, repo_key: str = "repo") -> Path:
    if sae_cfg.get("local_dir"):
        return Path(sae_cfg["local_dir"]) / sae_cfg.get(f"local_{repo_key}", "") / rel
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(sae_cfg[repo_key], rel, revision=sae_cfg.get("revision")))


def _gs_options(files: list[str]) -> dict:
    """{layer: {width: [l0, ...]}} from a Gemma Scope listing."""
    out = {}
    for f in files:
        m = _GS_RE.match(f)
        if m:
            out.setdefault(int(m["layer"]), {}).setdefault(m["width"], []).append(int(m["l0"]))
    return {L: {w: sorted(set(v)) for w, v in ws.items()} for L, ws in out.items()}


def closest_l0(options: list[int], target: float) -> int:
    """The L0 closest to target; ties go to the lower L0."""
    return min(options, key=lambda v: (abs(v - target), v))


def resolve(sae_cfg: dict, layers: list[int], files: list[str] | None = None,
            fallback_files: list[str] | None = None) -> dict[int, dict]:
    """For each 0-indexed hook layer L: {"main": info | None, "extra": info | None}. `extra` is the Gemma 16k file
    (fallback + width test) or the Qwen TopK-50 file (fallback)."""
    fmt = sae_cfg["format"]
    out = {}
    if fmt == "gemmascope":
        files = files if files is not None else _files(sae_cfg)
        opts = _gs_options(files)
        for L in layers:
            rec = {"main": None, "extra": None}
            ws = opts.get(L, {})
            w, xw = str(sae_cfg["width"]), str(sae_cfg["extra_width"])
            if w in ws:
                l0 = closest_l0(ws[w], float(sae_cfg["l0_target"]))
                rec["main"] = {"format": fmt, "repo": sae_cfg.get("repo", "local"), "repo_key": "repo", "layer": L,
                               "width": w, "l0": l0, "file": f"layer_{L}/width_{w}/average_l0_{l0}/params.npz",
                               "l0_options": ws[w], "rule": f"L0 closest to {sae_cfg['l0_target']} (ties lower)"}
            if xw in ws:
                ref = rec["main"]["l0"] if rec["main"] else float(sae_cfg["l0_target"])
                l0 = closest_l0(ws[xw], ref)
                rec["extra"] = {"format": fmt, "repo": sae_cfg.get("repo", "local"), "repo_key": "repo", "layer": L,
                                "width": xw, "l0": l0, "file": f"layer_{L}/width_{xw}/average_l0_{l0}/params.npz",
                                "l0_options": ws[xw], "rule": f"L0 closest to the {w} file's L0 {ref} (ties lower)"}
            out[L] = rec
    elif fmt == "qwenscope":
        for key, fl, slot in (("repo", files, "main"), ("fallback_repo", fallback_files, "extra")):
            if not sae_cfg.get(key):
                for L in layers:
                    out.setdefault(L, {"main": None, "extra": None})
                continue
            fl = fl if fl is not None else _files(sae_cfg, key)
            have = {int(m["layer"]): f for f in fl if (m := _QS_RE.match(f))}
            for L in layers:
                rec = out.setdefault(L, {"main": None, "extra": None})
                if L in have:
                    rec[slot] = {"format": fmt, "repo": sae_cfg.get(key, "local"), "repo_key": key, "layer": L,
                                 "file": have[L], "config_file": "config.json"}
    else:
        raise SAEError(f"unknown v3 SAE format {fmt}")
    return out


# ============================================================================= loading

def load(sae_cfg: dict, info: dict, device, expected_d: int | None = None) -> SAE3:
    fmt = info["format"]
    path = fetch(sae_cfg, info["file"], info.get("repo_key", "repo"))
    if fmt == "gemmascope":
        z = np.load(path, allow_pickle=False)
        g = {k.lower(): z[k] for k in z.files}           # SAELens maps w_enc -> W_enc; accept both spellings
        need = ["w_enc", "w_dec", "b_enc", "b_dec", "threshold"]
        miss = [k for k in need if k not in g]
        if miss:
            raise SAEError(f"{info['file']}: missing {miss} (has {sorted(g)})")
        t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))  # noqa: E731
        W_enc, W_dec = t(g["w_enc"]), t(g["w_dec"])
        if W_enc.shape[::-1] != W_dec.shape:
            raise SAEError(f"{info['file']}: W_enc {tuple(W_enc.shape)} vs W_dec {tuple(W_dec.shape)}")
        sae = SAE3("jumprelu", W_enc.T, t(g["b_enc"]), W_dec, t(g["b_dec"]), threshold=t(g["threshold"]),
                   meta={**info, "trained_on": sae_cfg.get("trained_on")})
    elif fmt == "qwenscope":
        cfgp = fetch(sae_cfg, info.get("config_file", "config.json"), info.get("repo_key", "repo"))
        c = read_json(cfgp)
        if c.get("model_type") not in (None, "topk_sae"):
            raise SAEError(f"{info['repo']}: model_type {c.get('model_type')} is not topk_sae")
        if sae_cfg.get("trained_on") and c.get("base_model") and c["base_model"] != sae_cfg["trained_on"]:
            raise SAEError(f"{info['repo']} was trained on {c['base_model']}, not {sae_cfg['trained_on']}")
        if c.get("hook_point") not in (None, "resid_post"):
            raise SAEError(f"{info['repo']}: hook_point {c.get('hook_point')} is not resid_post")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        miss = [k for k in ("W_enc", "W_dec", "b_enc", "b_dec") if k not in sd]
        if miss:
            raise SAEError(f"{info['file']}: missing {miss} (has {sorted(sd)})")
        W_enc, W_dec = sd["W_enc"].float(), sd["W_dec"].float()
        n, d = int(c.get("d_sae", W_enc.shape[0])), int(c.get("d_model", W_enc.shape[1]))
        if tuple(W_enc.shape) != (n, d) or tuple(W_dec.shape) != (d, n):
            raise SAEError(f"{info['file']}: expected W_enc [{n}, {d}] and W_dec [{d}, {n}], got "
                           f"{tuple(W_enc.shape)} / {tuple(W_dec.shape)}")
        sae = SAE3("topk", W_enc, sd["b_enc"].float(), W_dec.T, sd["b_dec"].float(), k=int(c["k"]),
                   meta={**info, "trained_on": c.get("base_model"), "k": int(c["k"])})
    else:
        raise SAEError(f"unknown v3 SAE format {fmt}")
    if expected_d is not None and sae.d != int(expected_d):
        raise SAEError(f"SAE d={sae.d} does not match d_model={expected_d} ({info['file']})")
    LOG.info("SAE %s %s: %d latents x d=%d (%s%s)", info.get("repo"), info["file"], sae.n, sae.d, sae.kind,
             f", k={sae.k}" if sae.k else "")
    return sae.to(device)


def set_input_norm(sae: SAE3, mode: str, mean_sq_norm: float | None) -> SAE3:
    """none: raw activations (both model cards); rms_d: rescale so that E||x||^2 = d before encoding."""
    if mode in (None, "none"):
        sae.norm_factor = 1.0
    elif mode == "rms_d":
        sae.norm_factor = (float(mean_sq_norm) / sae.d) ** 0.5
    else:
        raise SAEError(f"unknown input_norm {mode}")
    return sae


def fve(sae: SAE3, X: torch.Tensor, device, chunk: int = 2048) -> float:
    """Fraction of variance explained over the rows of X (dictionary_learning definition)."""
    Xh = torch.cat([sae.decode(sae.encode(X[i:i + chunk].to(device))).cpu() for i in range(0, X.shape[0], chunk)])
    return float(1 - (X - Xh).var(0).sum() / X.var(0).sum())


def l0(sae: SAE3, X: torch.Tensor, device, n: int = 2048) -> float:
    return float((sae.encode(X[:n].to(device)) > 0).float().sum(-1).mean())
