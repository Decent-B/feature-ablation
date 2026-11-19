"""v2 SAE resolution: one SAE file per (model, layer), both release layouts, hub or local (dev tiny).

dictionary_learning  <root>/resid_post_layer_<N>/trainer_<t>/{ae.pt, config.json}; the trainer is picked by
                     width and L0 from its config.json (never hardcoded).
reins                <folder>/layer_<NN>.pt
The file *named* N holds the SAE for 0-indexed decoder-layer output N (`layer_offset: 0`), with no input
scaling (`input_norm: none`), as both releases' model cards and code state (sources in config/v2.yaml).
The gate's fidelity check (FVE and next-token loss) still decides whether each SAE is included at each layer.
The checkpoint loader is v1's `sae.SAE`.
"""

from __future__ import annotations

import re
from pathlib import Path

from .sae import SAE
from .utils import LOG, read_json

_DL_RE = re.compile(r"^(?P<root>.*?)resid_post_layer_(?P<layer>\d+)/trainer_(?P<trainer>\d+)/ae\.pt$")
_REINS_RE = re.compile(r"^(?P<folder>.+)/layer_(?P<layer>\d+)\.pt$")


def _files(sae_cfg: dict) -> list[str]:
    if sae_cfg.get("local_dir"):
        root = Path(sae_cfg["local_dir"])
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(sae_cfg["repo"])


def _fetch(sae_cfg: dict, rel: str) -> Path:
    if sae_cfg.get("local_dir"):
        return Path(sae_cfg["local_dir"]) / rel
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(sae_cfg["repo"], rel, revision=sae_cfg.get("revision")))


def available(sae_cfg: dict, model_id: str, files: list[str] | None = None) -> dict[int, dict]:
    """{file layer name N: {file, ...}} for the configured width / L0."""
    files = files if files is not None else _files(sae_cfg)
    out = {}
    if sae_cfg["format"] == "dictionary_learning":
        cands = {}
        for f in files:
            m = _DL_RE.match(f)
            if m:
                cands.setdefault(int(m["layer"]), []).append((int(m["trainer"]), f))
        for N, lst in cands.items():
            for trainer, f in sorted(lst):
                cfg_file = f.replace("ae.pt", "config.json")
                c = read_json(_fetch(sae_cfg, cfg_file))["trainer"]
                if int(c["dict_size"]) == int(sae_cfg["width"]) and int(c["k"]) == int(sae_cfg.get("l0", c["k"])):
                    if c.get("lm_name") and c["lm_name"] != model_id:
                        raise RuntimeError(f"SAE {f} was trained on {c['lm_name']}, not {model_id}")
                    out[N] = {"file": f, "config_file": cfg_file, "trainer": trainer, "width": int(c["dict_size"]),
                              "k": int(c["k"]), "submodule": c.get("submodule_name")}
                    break
    elif sae_cfg["format"] == "reins":
        folder = sae_cfg["folder"].strip("/")
        for f in files:
            m = _REINS_RE.match(f)
            if m and m["folder"] == folder:
                out[int(m["layer"])] = {"file": f, "width": int(sae_cfg["width"])}
    else:
        raise ValueError(f"unknown SAE format {sae_cfg['format']}")
    return out


def resolve(sae_cfg: dict, model_id: str, layers: list[int], files: list[str] | None = None) -> dict[int, dict | None]:
    """For each 0-indexed hook layer L: the file named L + layer_offset, or None if the release lacks it."""
    off = int(sae_cfg.get("layer_offset", 0))
    av = available(sae_cfg, model_id, files)
    out = {}
    for L in layers:
        info = av.get(L + off)
        if info is not None:
            info = {**info, "repo": sae_cfg.get("repo", "local"), "format": sae_cfg["format"], "name_layer": L + off,
                    "hook_layer": L, "layer_offset": off, "available_name_layers": sorted(av)}
        out[L] = info
    return out


def load(sae_cfg: dict, info: dict, device) -> SAE:
    path = _fetch(sae_cfg, info["file"])
    sae = SAE.load(path, device)
    return sae


def set_input_norm(sae: SAE, mode: str, scale_stats: dict | None) -> SAE:
    """mode: none (normalisation folded into the checkpoint) | rms_unit (E||x||^2 = 1) | rms_d (E||x||^2 = d).
    scale_stats: {"mean_sq_norm": E||x||^2 at this layer on chat prompts}."""
    if mode == "none":
        sae.norm_factor = 1.0
        return sae
    msq = float(scale_stats["mean_sq_norm"])
    sae.norm_factor = (msq ** 0.5) if mode == "rms_unit" else (msq / sae.d) ** 0.5
    LOG.info("SAE input normalisation %s: scale %.4f", mode, sae.norm_factor)
    return sae
