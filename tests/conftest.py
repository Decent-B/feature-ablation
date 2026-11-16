import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def tiny_qwen():
    """A 3-layer random Qwen3 (no download) for hook / fast-path tests."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    m = Qwen3ForCausalLM(cfg).eval()
    m.requires_grad_(False)
    m.config._attn_implementation = "sdpa"
    return m


@pytest.fixture
def toy_sae():
    from pilot.sae import SAE

    g = torch.Generator().manual_seed(3)
    d, n = 16, 48
    dec = torch.randn(d, n, generator=g)
    dec = dec / dec.norm(dim=0, keepdim=True)
    sd = {"encoder.weight": dec.T.clone() * 1.3, "encoder.bias": torch.randn(n, generator=g) * 0.1,
          "decoder.weight": dec, "b_dec": torch.randn(d, generator=g) * 0.2, "threshold": torch.tensor(0.1),
          "k": torch.tensor(8)}
    return SAE.from_state_dict(sd, "cpu")
