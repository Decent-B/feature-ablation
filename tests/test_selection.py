"""Toy data with a planted unit is recovered by each selection rule; statistics helpers."""

import numpy as np
import torch

from pilot import bases as BZ
from pilot import selection as SEL
from pilot import stats


def _planted(n=400, p=60, unit=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    R = torch.randn(n, p, generator=g)
    R[:, unit] += 2.5 * (2 * y.float() - 1)
    return R, y, unit


def test_meandiff_recovers_planted_unit():
    R, y, unit = _planted()
    order, score, _ = SEL.rank_meandiff(R, y)
    assert order[0] == unit


def test_l1probe_recovers_planted_unit_first():
    R, y, unit = _planted(seed=1)
    d_abs = SEL.cohens_d(R, y).abs().numpy()
    cfg = {"n_steps": 20, "lambda_min_ratio": 1e-3, "max_iter": 400, "tol": 1e-5}
    order, score, diag = SEL.rank_l1probe(R, y, d_abs, cfg)
    assert order[0] == unit
    assert diag["nonzero_at_weakest"] >= 1


def test_l1probe_prefilter_keeps_planted_unit():
    R, y, unit = _planted(seed=2, p=300)
    d_abs = SEL.cohens_d(R, y).abs().numpy()
    cfg = {"n_steps": 10, "lambda_min_ratio": 1e-2, "max_iter": 200, "tol": 1e-5}
    order, _, diag = SEL.rank_l1probe(R, y, d_abs, cfg, prefilter=50)
    assert diag["prefiltered"] and order[0] == unit


def test_attrib_recovers_planted_unit():
    # S(x) = w* . x summed over positions: only direction e_unit matters for the score.
    d, T, n, unit = 12, 3, 64, 5
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, T, d, generator=g) + 1.0
    grad = torch.zeros(n, T, d)
    grad[:, :, unit] = 1.0                     # dS/dx
    basis = BZ.DimsBasis(d, "cpu")
    v = torch.zeros(T, d)                      # ablating to 0 removes positive mass -> negative change in S
    keep = torch.ones(n, T, dtype=torch.bool)
    C = SEL.attrib_contrib(basis, x, grad, None, v, keep)
    order, score = SEL.rank_attrib(C.mean(0).numpy())
    assert order[0] == unit


def test_attrib_sae_matches_explicit_formula(toy_sae):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(3, 2, 16, generator=g) * 2
    gr = torch.randn(3, 2, 16, generator=g)
    basis = BZ.SAEBasis(toy_sae)
    v = torch.rand(2, basis.n_units, generator=g)
    C = SEL.attrib_contrib(basis, x, gr, None, v, torch.ones(3, 2, dtype=torch.bool))
    r = toy_sae.encode(x)
    explicit = ((v - r) * (gr @ toy_sae.directions().T)).sum(1)
    assert torch.allclose(C, explicit, atol=1e-5)


def test_random_excludes_dead_latents():
    allowed = np.zeros(100, bool)
    allowed[:30] = True
    order, _ = SEL.rank_random(100, np.random.default_rng(0), allowed)
    assert set(order[:30]) == set(range(30))
    assert sorted(order) == list(range(100))


def test_jaccard():
    a, b = np.arange(10), np.r_[np.arange(5), np.arange(20, 25)]
    jm, jmin, n = SEL.jaccard_topk([a, b], 10)
    assert n == 1 and abs(jm - 5 / 15) < 1e-9


# --------------------------------------------------------------------- statistics helpers

def _auroc_brute(s, y):
    pos, neg = s[y == 1], s[y == 0]
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def test_auroc_with_ties_matches_brute_force():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 200).astype(float)
    y = rng.integers(0, 2, 200)
    assert abs(stats.auroc(s, y) - _auroc_brute(s, y)) < 1e-12


def test_weighted_auroc_equals_resampled_auroc():
    rng = np.random.default_rng(1)
    s = np.round(rng.normal(size=80), 1)
    y = np.r_[np.ones(40), np.zeros(40)].astype(int)
    W = stats.bootstrap_counts(rng, 80, 5, strata=y)
    a = stats.auroc_weighted(s, y, W)
    for b in range(5):
        idx = np.repeat(np.arange(80), W[b].astype(int))
        assert abs(a[b] - _auroc_brute(s[idx], y[idx])) < 1e-12
        assert W[b, y == 1].sum() == 40


def test_value_at_cost_envelope_and_flags():
    C = np.array([[0.01, 0.1, 1.0]])
    D = np.array([[0.1, 0.05, 0.4]])
    v, f = stats.value_at_cost(C, D, np.sqrt(0.01 * 0.1))      # midpoint in log cost, envelope keeps 0.1
    assert abs(v[0] - 0.1) < 1e-9 and f[0] == stats.FLAG_OK
    v, f = stats.value_at_cost(C, D, 10.0)
    assert v[0] == 0.4 and f[0] == stats.FLAG_NEVER
    v, f = stats.value_at_cost(C, D, 0.005)
    assert abs(v[0] - 0.05) < 1e-9 and f[0] == stats.FLAG_BELOW


def test_bh_and_kendall():
    q = stats.bh_qvalues(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)
    assert stats.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert stats.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
