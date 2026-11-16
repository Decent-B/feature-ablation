"""Developer-only plumbing test: a tiny randomly initialised Qwen3 (real Qwen3 tokenizer and chat
template) plus a random SAE in the same checkpoint format. No model download is needed, so
every stage can be exercised end to end in a few minutes on CPU. Behavioural checks that a
random model cannot pass (format check, refusal labels, SAE fidelity) are relaxed in this mode
only and the relaxation is logged and recorded in manifest.json. Not part of the spec's
acceptance: the real smoke test is `bash run_pilot.sh --smoke` with Qwen3-1.7B.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .utils import LOG, RUNS_DIR, write_json

TINY_DIR = RUNS_DIR / "_dev_tiny_model"
SAE_PATH = TINY_DIR / "sae" / "resid_post_layer_2" / "trainer_0" / "ae.pt"


def ensure_tiny_model() -> Path:
    if (TINY_DIR / "config.json").exists() and SAE_PATH.exists():
        return TINY_DIR
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    TINY_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(TINY_DIR)
    tok.save_pretrained(TINY_DIR)
    d, n = 64, 256
    g = torch.Generator().manual_seed(1)
    dec = torch.randn(d, n, generator=g)
    dec = dec / dec.norm(dim=0, keepdim=True)
    sd = {"encoder.weight": dec.T.clone(), "encoder.bias": torch.zeros(n), "decoder.weight": dec,
          "b_dec": torch.zeros(d), "threshold": torch.tensor(0.05), "k": torch.tensor(16)}
    SAE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, SAE_PATH)
    write_json(SAE_PATH.parent / "config.json", {"trainer": {"dict_size": n, "k": 16, "layer": 2,
                                                             "lm_name": str(TINY_DIR)}})
    LOG.info("created dev tiny model at %s", TINY_DIR)
    return TINY_DIR


def dev_overrides() -> dict:
    return {"model": str(TINY_DIR), "dev_tiny": True,
            "models": {str(TINY_DIR): {"sae": {"format": "local", "path": str(SAE_PATH), "layer": 2, "width": 256},
                                       "approx_nonembedding_params": 2.0e5}},
            "pca": {"n_tokens": 3000}, "batch_size": 32}
