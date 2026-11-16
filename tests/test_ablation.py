"""Ablation identity (v_i = r_i(x) leaves x unchanged), the SAE error term is preserved, the
hook only touches the T intervention positions, and the layer-split fast path matches the
full forward on a tiny random Qwen3."""

import numpy as np
import torch

from pilot import bases as BZ
from pilot import models as M


def _apply(basis, x, idx, v):
    """Call the ablation hook on a [B, S, d] tensor as a forward-hook would."""
    T = x.shape[1]
    idx_t = torch.as_tensor(idx)
    hook = M.AblationHook(T, basis.directions(idx_t), basis.reader(idx_t), values_src=v,
                          src_index=np.arange(x.shape[0]))
    hook.batch_idx = np.arange(x.shape[0])
    return hook(None, None, x)


def _bases(toy_sae, d=16):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(200, d, generator=g)
    mu, U, ev = BZ.fit_pca(X)
    return {"dims": BZ.DimsBasis(d, "cpu"), "pca": BZ.PCABasis(mu, U, ev), "sae": BZ.SAEBasis(toy_sae),
            "dir": BZ.DirectionBasis(torch.randn(3, d, generator=g))}


def test_identity_leaves_x_unchanged(toy_sae):
    x = torch.randn(4, 5, 16)
    for key, basis in _bases(toy_sae).items():
        idx = np.arange(min(7, basis.n_units))
        v = basis.read_sel(x, None, torch.as_tensor(idx))          # v_i = r_i(x)
        out = _apply(basis, x, idx, v)
        assert torch.allclose(out, x, atol=1e-5), key


def test_ablation_formula_and_sae_error_term(toy_sae):
    x = torch.randn(3, 4, 16) * 2
    basis = BZ.SAEBasis(toy_sae)
    idx = np.array([1, 5, 9])
    r = basis.read_sel(x, None, torch.as_tensor(idx))
    v = torch.zeros_like(r)
    out = _apply(basis, x, idx, v)
    expected = x + (v - r) @ toy_sae.directions(torch.as_tensor(idx))
    assert torch.allclose(out, expected, atol=1e-5)
    # the reconstruction error term survives: x' is not replaced by a reconstruction
    err_before = x - toy_sae.decode(toy_sae.encode(x))
    assert err_before.norm() > 1e-2
    assert not torch.allclose(out, toy_sae.decode(toy_sae.encode(out)), atol=1e-3)


def test_hook_touches_only_last_T_positions_and_skips_decode_steps(toy_sae):
    basis = BZ.DimsBasis(16, "cpu")
    x = torch.randn(2, 9, 16)
    T = 3
    idx = torch.arange(16)
    hook = M.AblationHook(T, basis.directions(idx), basis.reader(idx), values_static=torch.zeros(T, 16))
    hook.expected_len = 9
    out = hook(None, None, x)
    assert torch.equal(out[:, :-T], x[:, :-T])
    assert torch.allclose(out[:, -T:], torch.zeros_like(out[:, -T:]))
    step = torch.randn(2, 1, 16)
    assert hook(None, None, step) is step          # single-token decode step untouched


def _batches(model, n=6, seed=0):
    rng = np.random.default_rng(seed)
    toks = [list(rng.integers(1, 96, size=int(rng.integers(6, 14)))) for _ in range(n)]
    return M.make_batches(toks, pad_id=0, device=torch.device("cpu"), batch_size=4, max_batch_tokens=10_000)


def test_left_padding_is_batch_invariant(tiny_qwen):
    rng = np.random.default_rng(1)
    toks = [list(rng.integers(1, 96, size=L)) for L in (5, 11, 8)]
    together = M.make_batches(toks, 0, torch.device("cpu"), 8, 10_000)
    with torch.inference_mode():
        lt = {int(i): row for b in together for i, row in zip(b.idx, M.last_logits(tiny_qwen, b))}
        for i, t in enumerate(toks):
            alone = M.make_batches([t], 0, torch.device("cpu"), 8, 10_000)[0]
            assert torch.allclose(M.last_logits(tiny_qwen, alone)[0], lt[i], atol=1e-4)


def test_fast_path_matches_full_forward_with_and_without_hook(tiny_qwen):
    L, T = 1, 3
    layers = M.decoder_layers(tiny_qwen)
    batches = _batches(tiny_qwen)
    cap = M.Capture(layers, L, None)
    with torch.inference_mode():
        full = {}
        for b in batches:
            full[id(b)] = M.last_logits(tiny_qwen, b)
            b.cache["h"] = cap.h.clone()
    cap.remove()
    basis = BZ.DimsBasis(32, "cpu")
    idx = torch.arange(5)
    for use_hook in (False, True):
        ref = {}
        if use_hook:
            hook = M.AblationHook(T, basis.directions(idx), basis.reader(idx), values_static=torch.zeros(T, 5))
            hs = M.attach_hook(layers, L, hook, False)
            with torch.inference_mode():
                for b in batches:
                    hook.expected_len, hook.batch_idx = b.S, b.idx
                    ref[id(b)] = M.last_logits(tiny_qwen, b)
            for h in hs:
                h.remove()
        else:
            ref = full
        with M.LayerSplit(tiny_qwen, L) as sp, torch.inference_mode():
            hs = []
            if use_hook:
                hook = M.AblationHook(T, basis.directions(idx), basis.reader(idx), values_static=torch.zeros(T, 5))
                hs = M.attach_hook(layers, L, hook, True)
            for b in batches:
                sp.inject.value = b.cache["h"]
                if use_hook:
                    hook.expected_len, hook.batch_idx = b.S, b.idx
                fast = M.last_logits(tiny_qwen, b)
                assert (fast - ref[id(b)]).abs().max() < 1e-4
            for h in hs:
                h.remove()
    # layers restored after the context
    assert all(type(l).__name__ == "Qwen3DecoderLayer" for l in layers)


def test_identity_on_model_logits(tiny_qwen):
    L, T = 1, 3
    layers = M.decoder_layers(tiny_qwen)
    batches = _batches(tiny_qwen, seed=4)
    cap = M.Capture(layers, L, T)
    clean, acts = {}, {}
    with torch.inference_mode():
        for b in batches:
            clean[id(b)] = M.last_logits(tiny_qwen, b)
            acts[id(b)] = cap.h.clone()
    cap.remove()
    g = torch.Generator().manual_seed(0)
    basis = BZ.PCABasis(*BZ.fit_pca(torch.randn(300, 32, generator=g)))
    idx = torch.arange(10)
    for b in batches:
        v = basis.read_sel(acts[id(b)], None, idx)
        hook = M.AblationHook(T, basis.directions(idx), basis.reader(idx), values_src=v, src_index=np.arange(len(b.idx)))
        hs = M.attach_hook(layers, L, hook, False)
        with torch.inference_mode():
            hook.expected_len, hook.batch_idx = b.S, np.arange(len(b.idx))
            out = M.last_logits(tiny_qwen, b)
        for h in hs:
            h.remove()
        assert (out - clean[id(b)]).abs().max() < 1e-4
        assert hook.fired == 1
