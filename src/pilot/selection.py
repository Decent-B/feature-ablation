"""Selection rules (computed on discovery data at the final prompt token).

  meandiff  Cohen's d between classes; rank by |d|
  l1probe   L1 logistic regression on standardised r along a 20-step path from strong to
            weak; rank by the step at which a unit first becomes non-zero, ties by |coef|
            at the weakest step (then |d|). Warm starts, capped iterations, convergence logged.
  attrib    A_i = E_x[ sum_t (v_i,t - r_i,t(x)) * (w_i . dS/dx_t) ]; rank by most negative A_i
  random    uniform permutation (SAE: only latents firing on >= 1% of discovery tokens)

Rankings are full orderings of unit indices; `score` is rule-specific and higher means
ranked earlier (meandiff |d|; l1probe (n_steps - entry_step) + tanh|coef|; attrib -A).
"""

from __future__ import annotations

import numpy as np
import torch

from .utils import LOG


def cohens_d(R: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    R = R.double()
    a, b = R[y == 1], R[y == 0]
    n1, n0 = a.shape[0], b.shape[0]
    m1, m0 = a.mean(0), b.mean(0)
    v1 = a.var(0, unbiased=True) if n1 > 1 else torch.zeros_like(m1)
    v0 = b.var(0, unbiased=True) if n0 > 1 else torch.zeros_like(m0)
    sp = torch.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / max(1, n1 + n0 - 2))
    d = torch.where(sp > 1e-12, (m1 - m0) / sp.clamp_min(1e-12), torch.zeros_like(sp))
    return d.float()


def rank_meandiff(R, y):
    d = cohens_d(R, y)
    score = d.abs()
    order = torch.argsort(score, descending=True, stable=True)
    return order.cpu().numpy(), score.cpu().numpy(), d


def _power_lipschitz(Z: torch.Tensor, iters: int = 50) -> float:
    v = torch.randn(Z.shape[1], 1, device=Z.device, dtype=Z.dtype, generator=None)
    v /= v.norm()
    for _ in range(iters):
        v = Z.T @ (Z @ v)
        v /= v.norm().clamp_min(1e-12)
    s2 = float((Z @ v).norm() ** 2)
    return s2 / (4.0 * Z.shape[0]) + 1e-12


def l1_logistic_path(Z: torch.Tensor, y: torch.Tensor, n_steps: int = 20, lambda_min_ratio: float = 1e-3,
                     max_iter: int = 400, tol: float = 1e-5):
    """FISTA on (1/n) sum logloss + lam ||w||_1 with an unpenalised intercept.
    Returns entry_step [p] (n_steps if never non-zero), coef at weakest step [p], diagnostics."""
    torch.manual_seed(0)
    dt = torch.float32 if Z.is_cuda else torch.float64     # GB10 FP64 throughput is low
    Z = Z.to(dt)
    y = y.to(dt)
    n, p = Z.shape
    ybar = float(y.mean().clamp(1e-6, 1 - 1e-6))
    lam_max = float((Z.T @ (y - ybar)).abs().max() / n)
    lams = lam_max * lambda_min_ratio ** (np.arange(1, n_steps + 1) / n_steps)
    L = _power_lipschitz(Z)
    step = 1.0 / L
    w = torch.zeros(p, dtype=Z.dtype, device=Z.device)
    b = torch.tensor(np.log(ybar / (1 - ybar)), dtype=Z.dtype, device=Z.device)
    entry = torch.full((p,), n_steps, dtype=torch.long, device=Z.device)
    unconverged = 0
    iters_used = []
    for j, lam in enumerate(lams):
        wv, bv, t = w.clone(), b.clone(), 1.0
        it = 0
        for it in range(1, max_iter + 1):
            r = torch.sigmoid(Z @ wv + bv) - y
            gw = Z.T @ r / n
            gb = r.mean()
            w_new = wv - step * gw
            w_new = torch.sign(w_new) * torch.clamp(w_new.abs() - step * lam, min=0.0)
            b_new = bv - step * gb
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            diff = float(torch.max((w_new - w).abs().max(), (b_new - b).abs()))
            scale = max(1.0, float(w_new.abs().max()))
            wv = w_new + ((t - 1) / t_new) * (w_new - w)
            bv = b_new + ((t - 1) / t_new) * (b_new - b)
            w, b, t = w_new, b_new, t_new
            if diff < tol * scale:
                break
        else:
            unconverged += 1
        iters_used.append(it)
        nz = (w != 0) & (entry == n_steps)
        entry[nz] = j
    diag = {"lambda_max": lam_max, "lambdas": [float(x) for x in lams], "unconverged_steps": unconverged,
            "iters": iters_used, "nonzero_at_weakest": int((w != 0).sum())}
    return entry.cpu().numpy(), w.float().cpu().numpy(), diag


def standardise(R: torch.Tensor) -> torch.Tensor:
    mu = R.mean(0)
    sd = R.std(0, unbiased=False)
    return torch.where(sd > 1e-8, (R - mu) / sd.clamp_min(1e-8), torch.zeros_like(R))


def rank_l1probe(R, y, d_abs: np.ndarray, cfg: dict, prefilter: int | None = None, tag: str = ""):
    p = R.shape[1]
    cols = np.arange(p)
    if prefilter is not None and p > prefilter:
        cols = np.argsort(-d_abs, kind="stable")[:prefilter]
        LOG.info("l1probe %s: prefiltered to the top %d of %d units by |d|", tag, prefilter, p)
    Z = standardise(R[:, torch.as_tensor(cols, device=R.device)].float())
    entry_s, coef_s, diag = l1_logistic_path(Z, y, cfg["n_steps"], cfg["lambda_min_ratio"],
                                             cfg["max_iter"], cfg["tol"])
    if diag["unconverged_steps"]:
        LOG.warning("l1probe %s: %d of %d path steps hit max_iter=%d without converging", tag,
                    diag["unconverged_steps"], cfg["n_steps"], cfg["max_iter"])
    n_steps = cfg["n_steps"]
    entry = np.full(p, n_steps + 1, dtype=np.int64)     # not prefiltered -> after everything
    coef = np.zeros(p)
    entry[cols] = entry_s
    coef[cols] = np.abs(coef_s)
    order = np.lexsort((-d_abs, -coef, entry))
    score = (n_steps - entry) + np.tanh(coef)
    diag["prefiltered"] = bool(len(cols) < p)
    diag["n_candidates"] = int(len(cols))
    return order, score, diag


def rank_attrib(A: np.ndarray):
    order = np.argsort(A, kind="stable")             # most negative predicted change in S first
    return order, -A


def rank_random(n_units: int, rng: np.random.Generator, allowed: np.ndarray | None = None):
    pool = np.arange(n_units) if allowed is None else np.flatnonzero(allowed)
    perm = rng.permutation(pool)
    if allowed is not None:                         # never-firing latents go last, never selected in practice
        rest = rng.permutation(np.flatnonzero(~allowed))
        perm = np.concatenate([perm, rest])
    return perm, np.zeros(n_units)


def attrib_contrib(basis, x, g, m, v, keep) -> torch.Tensor:
    """Per-example first-order predicted change in S from ablating each unit alone.
    x, g [B,T,d]; m [B,T,f]|None; v [T,n]; keep [B,T] -> [B, n]."""
    r = basis.read(x, m)                               # [B,T,n]
    if basis.key == "dims":
        gw = g
    elif basis.key == "pca":
        gw = g @ basis.U.T
    elif basis.key == "sae":
        gw = g @ basis.sae.directions().T
    elif basis.key == "mlp":
        gw = g @ basis.W
    else:
        gw = g @ basis.directions(torch.arange(basis.n_units)).T
    c = (v.unsqueeze(0) - r) * gw * keep.unsqueeze(-1).float()
    return c.sum(1)


def jaccard_topk(orders: list[np.ndarray], k: int) -> tuple[float, float, int]:
    sets = [set(o[:k].tolist()) for o in orders]
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            vals.append(len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])))
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.min(vals)), len(vals)
