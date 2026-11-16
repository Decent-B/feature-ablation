"""Numpy-only statistics: tie-aware AUROC, paired bootstrap weights, BH, Kendall tau,
and the damage-at-reference-cost interpolation.

Bootstrap replicates are represented as count matrices W [B, n] (how many times each
example is drawn). Every metric is then a weighted statistic, so the same W can be
shared across all configurations (a paired bootstrap) and AUROC needs only one sort
per score vector instead of one per replicate.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- AUROC

def auroc_weighted(scores, labels, W=None) -> np.ndarray:
    """AUROC = P(s_pos > s_neg) + 0.5 P(s_pos == s_neg), with optional count weights.

    scores, labels: [n]; W: [B, n] counts or None. Returns [B] (or shape [1] if W is None).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n = s.shape[0]
    if W is None:
        W = np.ones((1, n), dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ys = y[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    Ws = W[:, order]
    pos = Ws * ys
    neg = Ws * (~ys)
    Pg = np.add.reduceat(pos, starts, axis=1)
    Ng = np.add.reduceat(neg, starts, axis=1)
    Ncum = np.cumsum(Ng, axis=1) - Ng
    num = (Pg * (Ncum + 0.5 * Ng)).sum(axis=1)
    den = Pg.sum(axis=1) * Ng.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def auroc(scores, labels) -> float:
    return float(auroc_weighted(scores, labels)[0])


# --------------------------------------------------------------------------- bootstrap

def bootstrap_counts(rng: np.random.Generator, n: int, B: int, strata=None) -> np.ndarray:
    """Multinomial resampling counts [B, n]; resampled within each stratum if given."""
    W = np.zeros((B, n), dtype=np.float32)
    if strata is None:
        groups = [np.arange(n)]
    else:
        strata = np.asarray(strata)
        groups = [np.flatnonzero(strata == u) for u in np.unique(strata)]
    for idx in groups:
        m = len(idx)
        if m == 0:
            continue
        W[:, idx] = rng.multinomial(m, np.full(m, 1.0 / m), size=B)
    return W


def weighted_mean(values, W) -> np.ndarray:
    """values [n] or [C, n]; W [B, n] -> [B] or [C, B]."""
    v = np.asarray(values, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    tot = W.sum(axis=1)
    tot = np.where(tot > 0, tot, np.nan)
    return (v @ W.T) / tot


def percentile_ci(samples, alpha: float = 0.05, axis: int = -1):
    s = np.asarray(samples, dtype=np.float64)
    lo = np.nanpercentile(s, 100 * alpha / 2, axis=axis)
    hi = np.nanpercentile(s, 100 * (1 - alpha / 2), axis=axis)
    return lo, hi


def boot_pvalue(samples) -> float:
    """Two-sided bootstrap p-value for H0: statistic == 0 (sign-flip proportion)."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    B = s.size
    if B == 0:
        return 1.0
    le = (np.sum(s <= 0) + 1) / (B + 1)
    ge = (np.sum(s >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(le, ge)))


def bh_qvalues(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


# --------------------------------------------------------------------------- rank correlation

def kendall_tau(a, b) -> np.ndarray:
    """Kendall tau-b. a, b: [n] or [B, n] (row-wise). Returns scalar or [B]."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    da = np.sign(a[:, :, None] - a[:, None, :])
    db = np.sign(b[:, :, None] - b[:, None, :])
    iu = np.triu_indices(a.shape[1], k=1)
    da = da[:, iu[0], iu[1]]
    db = db[:, iu[0], iu[1]]
    s = (da * db).sum(axis=1)
    na = (da != 0).sum(axis=1)
    nb = (db != 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = s / np.sqrt(na * nb)
    tau = np.where((na > 0) & (nb > 0), tau, np.nan)
    return tau if tau.size > 1 else float(tau[0])


# --------------------------------------------------------------------------- curves

FLAG_OK, FLAG_BELOW, FLAG_NEVER = 0, 1, 2


def value_at_cost(C, D, cstar, envelope: bool = True, eps: float = 1e-12):
    """Interpolate damage D at reference cost cstar along the curve (C, D).

    C, D: [R, K] (R curves, K budget points); cstar: [R] or scalar.
    Linear in log cost between measured points; with envelope=True the damage is first
    replaced by its running maximum in cost order (monotone envelope).
    Below the cheapest point the curve is joined linearly to the origin (k=0: zero cost,
    zero damage). Above the most expensive point the curve's maximum is reported and
    flagged as never reaching c*.
    Returns (values [R], flags [R]).
    """
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    R, K = C.shape
    cs = np.broadcast_to(np.asarray(cstar, dtype=np.float64), (R,)).copy()
    C = np.maximum(C, eps)
    cs = np.maximum(cs, eps)
    o = np.argsort(C, axis=1, kind="mergesort")
    C = np.take_along_axis(C, o, axis=1)
    D = np.take_along_axis(D, o, axis=1)
    E = np.maximum.accumulate(D, axis=1) if envelope else D
    rows = np.arange(R)
    if K == 1:
        below = cs < C[:, 0]
        vals = np.where(below, E[:, 0] * cs / C[:, 0], E[:, 0])
        flags = np.where(below, FLAG_BELOW, np.where(cs > C[:, 0], FLAG_NEVER, FLAG_OK))
        return vals, flags
    j = (C <= cs[:, None]).sum(axis=1) - 1
    below = j < 0
    above = cs > C[:, -1]
    jj = np.clip(j, 0, K - 2)
    lc = np.log(cs)
    c0 = np.log(C[rows, jj])
    c1 = np.log(C[rows, jj + 1])
    e0 = E[rows, jj]
    e1 = E[rows, jj + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(c1 > c0, (lc - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    inter = e0 + t * (e1 - e0)
    top = E[:, -1] if not envelope else E.max(axis=1)
    vals = np.where(below, E[:, 0] * cs / C[:, 0], np.where(above, top, inter))
    flags = np.where(below, FLAG_BELOW, np.where(above, FLAG_NEVER, FLAG_OK))
    return vals, flags
