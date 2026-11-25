"""Developer plumbing test for v3 (`--v3 --smoke --dev-tiny`): tiny randomly initialised versions of both base
architectures with the real tokenizers, plus SAEs in both release layouts. No gated download and no GPU needed.

- google/gemma-2-2b       -> tiny Gemma 2 (8 layers, d=64, eager attention, soft-capping as the real model), Gemma
                             tokenizer from the ungated `unsloth/gemma-2-2b-it` mirror (same vocabulary); SAEs in the
                             Gemma Scope layout (layer_L/width_65k|16k/average_l0_X/params.npz, several L0 per width).
- Qwen/Qwen3-1.7B-Base    -> tiny Qwen3 (8 layers, d=64); SAEs in the Qwen-Scope layout (main/ and fallback/ with
                             layer{L}.sae.pt + config.json).
Each tiny SAE spans the top principal directions of its own layer (decoder rows [P; -P] plus dead latents, encoder
biases centring the input), so it reconstructs its layer well and exercises the SAE basis end to end. Model ids stay
logical; only weights and SAE locations change. Gates cannot pass on random weights and are not enforced in smoke.
"""

from __future__ import annotations

import numpy as np
import torch

from .utils import LOG, RUNS_DIR, write_json

TINY = RUNS_DIR / "_dev_tiny_v3"
D, N_LIVE = 64, 48
SPECS = {
    "google/gemma-2-2b": {"dir": TINY / "gemma2", "tok": ["google/gemma-2-2b", "unsloth/gemma-2-2b-it"], "arch": "gemma2"},
    "Qwen/Qwen3-1.7B-Base": {"dir": TINY / "qwen3base", "tok": ["Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B"], "arch": "qwen3"},
}
GS_WIDTHS = {"65k": (256, [40, 90, 130]), "16k": (128, [60, 95])}
QS_REPOS = {"main": 48, "fallback": 24}


def dev_tiny_overrides(cfg: dict) -> dict:
    out = {"models": {}}
    for mid, s in SPECS.items():
        sae = {"local_dir": str(s["dir"] / "sae")}
        if s["arch"] == "qwen3":
            sae.update(local_repo="main", local_fallback_repo="fallback")
        out["models"][mid] = {"weights": str(s["dir"]), "sae": sae}
    return out


def _tokenizer(names: list[str]):
    from transformers import AutoTokenizer

    err = None
    for n in names:
        try:
            return AutoTokenizer.from_pretrained(n), n
        except Exception as e:  # noqa: BLE001  (gated or not cached: try the next source)
            err = e
    raise RuntimeError(f"no tokenizer available from {names}: {err}")


def _build_model(spec: dict):
    tok, src = _tokenizer(spec["tok"])
    torch.manual_seed(0)
    if spec["arch"] == "gemma2":
        from transformers import Gemma2Config, Gemma2ForCausalLM

        cfg = Gemma2Config(vocab_size=len(tok), hidden_size=D, intermediate_size=128, num_hidden_layers=8,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16,
                           sliding_window=4096, max_position_embeddings=8192, attn_logit_softcapping=50.0,
                           final_logit_softcapping=30.0, pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                           eos_token_id=tok.eos_token_id)
        model = Gemma2ForCausalLM(cfg)
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=D, intermediate_size=128, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=8192,
                          tie_word_embeddings=True)
        model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model, tok, src


@torch.no_grad()
def _layer_acts(model, tok) -> dict:
    from .models import decoder_layers

    texts = [f"Statement: The city of City{i} is in Country{i % 7}.\nTrue or false?\nAnswer:" for i in range(24)]
    texts += [f"Write a short note about topic number {i} and why it matters." for i in range(24)]
    n = len(decoder_layers(model))
    store = {L: [] for L in range(n)}
    hs = [decoder_layers(model)[L].register_forward_hook(
        lambda m, a, o, L=L: store[L].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float())) for L in range(n)]
    try:
        for t in texts:
            model(input_ids=torch.tensor([tok(t)["input_ids"]]), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return {L: torch.cat(v) for L, v in store.items()}


def _pca_rows(X: torch.Tensor, n: int, seed: int):
    mu = X.mean(0)
    _, _, Vt = torch.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:N_LIVE]
    g = torch.Generator().manual_seed(seed)
    dead = torch.randn(n - 2 * N_LIVE, D, generator=g)
    dead = dead / dead.norm(dim=-1, keepdim=True)
    rows = torch.cat([P, -P, dead])
    b_enc = torch.cat([-(P @ mu), (P @ mu), torch.full((n - 2 * N_LIVE,), -1e4)])   # centres the input (no b_dec
    return rows, b_enc, mu                                                            # subtraction in either release)


def ensure_tiny_models() -> dict:
    out = {}
    for mid, s in SPECS.items():
        d = s["dir"]
        marker = d / "sae" / "DONE"
        out[mid] = d
        if (d / "config.json").exists() and marker.exists():
            continue
        model, tok, src = _build_model(s)
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d)
        tok.save_pretrained(d)
        acts = _layer_acts(model, tok)
        for L, X in acts.items():
            if s["arch"] == "gemma2":
                for w, (n, l0s) in GS_WIDTHS.items():
                    rows, b_enc, mu = _pca_rows(X, n, 100 + L)
                    thr = torch.cat([torch.full((2 * N_LIVE,), 1e-6), torch.ones(n - 2 * N_LIVE)])
                    for l0 in l0s:
                        p = d / "sae" / f"layer_{L}" / f"width_{w}" / f"average_l0_{l0}" / "params.npz"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(p, W_enc=rows.T.numpy(), W_dec=rows.numpy(), b_enc=b_enc.numpy(), b_dec=mu.numpy(),
                                 threshold=thr.numpy())
            else:
                rows, b_enc, mu = _pca_rows(X, 256, 200 + L)
                for repo, k in QS_REPOS.items():
                    p = d / "sae" / repo / f"layer{L}.sae.pt"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"W_enc": rows.clone(), "W_dec": rows.T.contiguous(), "b_enc": b_enc.clone(),
                                "b_dec": mu.clone()}, p)
                    write_json(p.parent / "config.json", {"model_type": "topk_sae", "base_model": "Qwen/Qwen3-1.7B-Base",
                                                          "d_model": D, "d_sae": 256, "k": k, "num_layers": 8,
                                                          "layers": list(range(8)), "hook_point": "resid_post"})
        marker.write_text("ok\n")
        LOG.info("created dev tiny model for %s at %s (tokenizer from %s)", mid, d, src)
    return out
