"""One SAE interface for both releases (both are dictionary_learning BatchTopK checkpoints).

- adamkarvonen/qwen3-*-saes: <root>/resid_post_layer_<L>/trainer_<t>/{ae.pt, config.json}
- Carlos4869/REINS-SAE:       <folder>/layer_<LL>.pt
Both are run as JumpReLU with the stored scalar threshold (never batch top-k), so a
token's activations do not depend on its batch. If a checkpoint carries an explicit
`norm_factor`, it is applied here; otherwise the caller sets one (v2: `sae_v2.set_input_norm`).
The REINS files carry none and still expect global-RMS-scaled inputs (about x5 at mid depth on
Qwen3.5-4B), so they must not be fed raw residuals. Decoder directions are returned in raw
activation scale.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from .utils import LOG, read_json

_DL_RE = re.compile(r"^(?P<root>.*?)resid_post_layer_(?P<layer>\d+)/trainer_(?P<trainer>\d+)/ae\.pt$")
_REINS_RE = re.compile(r"^(?P<folder>.+)/layer_(?P<layer>\d+)\.pt$")


def _nearest(layers: list[int], target: float) -> int:
    return min(sorted(layers), key=lambda L: (abs(L - target), L))


def resolve(model_id: str, sae_cfg: dict, n_layers: int, depth: float) -> dict:
    """List the repo and pick the layer nearest depth*n_layers plus the matching file."""
    from huggingface_hub import HfApi, hf_hub_download

    if sae_cfg["format"] == "local":       # dev tiny model only
        return {"repo": "local", "format": "local", "layer": int(sae_cfg["layer"]), "file": sae_cfg["path"],
                "width": int(sae_cfg["width"]), "available_layers": [int(sae_cfg["layer"])]}
    api = HfApi()
    repo = sae_cfg["repo"]
    files = api.list_repo_files(repo)
    target = depth * n_layers
    fmt = sae_cfg["format"]
    if fmt == "dictionary_learning":
        cands = {}
        for f in files:
            m = _DL_RE.match(f)
            if m:
                cands.setdefault(int(m["layer"]), []).append((int(m["trainer"]), f))
        if not cands:
            raise RuntimeError(f"no resid_post_layer_*/trainer_*/ae.pt files in {repo}")
        layer = _nearest(list(cands), target)
        chosen = None
        options = []
        for trainer, f in sorted(cands[layer]):
            cfg_file = f.replace("ae.pt", "config.json")
            c = read_json(hf_hub_download(repo, cfg_file))["trainer"]
            options.append({"trainer": trainer, "dict_size": c["dict_size"], "k": c["k"],
                            "lm_name": c.get("lm_name"), "layer": c.get("layer")})
            if int(c["dict_size"]) == int(sae_cfg["width"]) and int(c["k"]) == int(sae_cfg.get("l0", c["k"])):
                chosen = (trainer, f, cfg_file, c)
        if chosen is None:
            raise RuntimeError(f"no trainer with width {sae_cfg['width']} and L0 {sae_cfg.get('l0')} at "
                               f"layer {layer} in {repo}; available: {options}")
        trainer, f, cfg_file, c = chosen
        if c.get("lm_name") and c["lm_name"] != model_id:
            raise RuntimeError(f"SAE {f} was trained on {c['lm_name']}, not {model_id}")
        return {"repo": repo, "format": fmt, "layer": layer, "file": f, "config_file": cfg_file,
                "trainer": trainer, "width": int(c["dict_size"]), "k": int(c["k"]),
                "available_layers": sorted(cands), "options_at_layer": options,
                "submodule": c.get("submodule_name"), "repo_sha": api.model_info(repo).sha}
    if fmt == "reins":
        folder = sae_cfg["folder"].strip("/")
        cands = {}
        for f in files:
            m = _REINS_RE.match(f)
            if m and m["folder"] == folder:
                cands[int(m["layer"])] = f
        if not cands:
            raise RuntimeError(f"no {folder}/layer_XX.pt files in {repo}")
        layer = _nearest(list(cands), target)
        return {"repo": repo, "format": fmt, "layer": layer, "file": cands[layer], "width": int(sae_cfg["width"]),
                "available_layers": sorted(cands), "repo_sha": api.model_info(repo).sha}
    raise ValueError(f"unknown SAE format {fmt}")


class SAE:
    """encode / decode / decoder directions / threshold, all in fp32 on one device."""

    def __init__(self, W_enc, b_enc, W_dec_rows, b_dec, threshold: float, k: int | None,
                 norm_factor: float = 1.0):
        self.W_enc = W_enc            # [n, d]
        self.b_enc = b_enc            # [n]
        self.W_dec = W_dec_rows       # [n, d]  (decoder.weight.T)
        self.b_dec = b_dec            # [d]
        self.threshold = float(threshold)
        self.k = k
        self.norm_factor = float(norm_factor)
        self.n, self.d = W_enc.shape

    @classmethod
    def from_state_dict(cls, sd: dict, device) -> "SAE":
        norm_factor = 1.0
        if "ae" in sd and isinstance(sd["ae"], dict):        # dictionary_learning backup format
            norm_factor = float(sd.get("norm_factor") or 1.0)
            sd = sd["ae"]
        for key in ("norm_factor", "scaling_factor", "activation_scale"):
            if key in sd:
                norm_factor = float(torch.as_tensor(sd[key]).item())
        need = ["encoder.weight", "encoder.bias", "decoder.weight", "b_dec", "threshold"]
        miss = [k for k in need if k not in sd]
        if miss:
            raise RuntimeError(f"SAE checkpoint missing keys {miss}; has {sorted(sd)[:20]}")
        thr = float(torch.as_tensor(sd["threshold"]).item())
        if thr < 0:
            raise RuntimeError("SAE threshold is unset (< 0); cannot run as JumpReLU")
        f32 = lambda t: torch.as_tensor(t).to(device=device, dtype=torch.float32)  # noqa: E731
        k = int(torch.as_tensor(sd["k"]).item()) if "k" in sd else None
        return cls(f32(sd["encoder.weight"]), f32(sd["encoder.bias"]), f32(sd["decoder.weight"]).T.contiguous(),
                   f32(sd["b_dec"]), thr, k, norm_factor)

    @classmethod
    def load(cls, path: Path, device) -> "SAE":
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sae = cls.from_state_dict(sd, device)
        LOG.info("SAE loaded: %d latents x d=%d, threshold=%.4f, k=%s, norm_factor=%.4f (%s)", sae.n, sae.d,
                 sae.threshold, sae.k, sae.norm_factor,
                 "normalisation folded into checkpoint" if sae.norm_factor == 1.0 else "explicit norm factor")
        return sae

    def to(self, device) -> "SAE":
        for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # x in raw activation space throughout
    def _pre(self, x, idx=None):
        x = x.float() / self.norm_factor
        if idx is None:
            return (x - self.b_dec) @ self.W_enc.T + self.b_enc
        return (x - self.b_dec) @ self.W_enc[idx].T + self.b_enc[idx]

    def encode(self, x, idx=None):
        a = torch.relu(self._pre(x, idx))
        return a * (a > self.threshold)

    def decode(self, f):
        return (f @ self.W_dec + self.b_dec) * self.norm_factor

    def directions(self, idx=None):
        W = self.W_dec if idx is None else self.W_dec[idx]
        return W * self.norm_factor
