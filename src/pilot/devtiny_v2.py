"""Developer plumbing test for v2 (`--v2 --smoke --dev-tiny`): tiny randomly initialised versions of
both architectures with the real tokenizers and chat templates, plus SAEs in both release layouts.

- Qwen/Qwen3-1.7B  -> tiny Qwen3 (8 layers, d=64), SAEs in the dictionary_learning layout
                      (<root>/resid_post_layer_<L>/trainer_2/{ae.pt, config.json}) at layers 2, 4, 6.
- Qwen/Qwen3.5-4B  -> tiny hybrid Qwen3.5 (8 layers: 3 Gated DeltaNet + 1 attention, twice), SAEs in the
                      REINS flat layout (<folder>/layer_XX.pt) at every layer.
Each tiny SAE spans the top principal directions of its own layer (decoder rows [P; -P], threshold 0,
plus dead latents), so it reconstructs its layer well and exercises the SAE basis end to end. Model ids
stay logical; only weights and SAE locations change. Behavioural gates cannot pass on random weights and
are relaxed (and logged) in this mode only. Not part of the spec's acceptance.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .utils import LOG, RUNS_DIR, write_json

TINY = RUNS_DIR / "_dev_tiny_v2"
SPECS = {
    "Qwen/Qwen3-1.7B": {"dir": TINY / "qwen3", "tok": "Qwen/Qwen3-1.7B", "arch": "qwen3", "sae_layers": [2, 4, 6],
                        "sae": {"format": "dictionary_learning", "width": 256, "l0": 16}},
    "Qwen/Qwen3.5-4B": {"dir": TINY / "qwen3_5", "tok": "Qwen/Qwen3.5-4B", "arch": "qwen3_5",
                        "sae_layers": list(range(8)),
                        "sae": {"format": "reins", "folder": "qwen3.5-4b-base", "width": 256}},
}
D, N_LAT, N_LIVE = 64, 256, 48


def dev_tiny_overrides(cfg: dict) -> dict:
    out = {"models": {}}
    for mid, s in SPECS.items():
        sae = dict(s["sae"])
        sae["local_dir"] = str(s["dir"] / "sae")
        out["models"][mid] = {"weights": str(s["dir"]), "sae": sae}
    return out


def _build_model(spec: dict):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["tok"])
    torch.manual_seed(0)
    if spec["arch"] == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=D, intermediate_size=128, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                          tie_word_embeddings=True)
        model = Qwen3ForCausalLM(cfg)
    else:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=D, intermediate_size=128, num_hidden_layers=8,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_value_heads=4,
                                linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                                linear_conv_kernel_dim=4, tie_word_embeddings=True, max_position_embeddings=4096,
                                layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3
                                + ["full_attention"])
        model = Qwen3_5ForCausalLM(cfg)
    model.eval()
    return model, tok


@torch.no_grad()
def _layer_acts(model, tok, layers: list[int]) -> dict:
    from . import prompts as P
    from .models import decoder_layers

    texts = [f"Answer with True or False only.\nStatement: The city of City{i} is in Country{i % 7}." for i in range(24)]
    texts += [f"Write a short note about topic number {i} and why it matters." for i in range(24)]
    store = {L: [] for L in layers}
    hs = []
    for L in layers:
        hs.append(decoder_layers(model)[L].register_forward_hook(
            lambda m, a, o, L=L: store[L].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float())))
    try:
        for t in texts:
            ids = torch.tensor([P.encode(tok, t)])
            model(input_ids=ids, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return {L: torch.cat(v) for L, v in store.items()}


def _tiny_sae(X: torch.Tensor, seed: int) -> dict:
    mu = X.mean(0)
    Xc = X - mu
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    Pm = Vt[:N_LIVE]                                           # top principal directions of this layer
    g = torch.Generator().manual_seed(seed)
    dead = torch.randn(N_LAT - 2 * N_LIVE, D, generator=g)
    dead = dead / dead.norm(dim=-1, keepdim=True)
    rows = torch.cat([Pm, -Pm, dead])                           # [n, d]
    b_enc = torch.cat([torch.zeros(2 * N_LIVE), torch.full((N_LAT - 2 * N_LIVE,), -1e4)])
    return {"encoder.weight": rows.clone(), "encoder.bias": b_enc, "decoder.weight": rows.T.contiguous(),
            "b_dec": mu, "threshold": torch.tensor(0.0), "k": torch.tensor(16)}


def ensure_tiny_models() -> dict:
    """Create (once) both tiny models and their SAEs; returns {model_id: weights dir}."""
    out = {}
    for mid, s in SPECS.items():
        d = s["dir"]
        marker = d / "sae" / "DONE"
        out[mid] = d
        if (d / "config.json").exists() and marker.exists():
            continue
        model, tok = _build_model(s)
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d)
        tok.save_pretrained(d)
        acts = _layer_acts(model, tok, s["sae_layers"])
        for L in s["sae_layers"]:
            sd = _tiny_sae(acts[L], 100 + L)
            if s["sae"]["format"] == "dictionary_learning":
                p = d / "sae" / "saes_tiny_batch_top_k" / f"resid_post_layer_{L}" / "trainer_2" / "ae.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(sd, p)
                write_json(p.parent / "config.json", {"trainer": {"dict_size": N_LAT, "k": 16, "layer": L,
                                                                  "lm_name": mid, "submodule_name": f"resid_post_layer_{L}"}})
            else:
                p = d / "sae" / s["sae"]["folder"] / f"layer_{L:02d}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(sd, p)
        marker.write_text("ok\n")
        LOG.info("created dev tiny model for %s at %s (SAEs at layers %s)", mid, d, s["sae_layers"])
    return out
