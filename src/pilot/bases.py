"""Unit bases. Each unit i has a read value r_i(x) and a write direction w_i in residual space;
every basis is ablated through the same formula x' = x + sum_{i in S} (v_i - r_i(x)) w_i.

  sae   r = SAE encoder activation (JumpReLU)     w = decoder row, raw scale
  pca   r = u_i . (x - mu)                         w = u_i (orthonormal)
  dims  r = x_i                                    w = e_i
  mlp   r = input to down_proj at layer L          w = column i of down_proj   (different site)
  dir   r = u_j . x                                w = u_j                     (anchors)
"""

from __future__ import annotations

import torch


class Basis:
    key = "base"
    site = "resid_post"
    needs_mlp = False
    n_units: int
    d: int

    def read(self, x, m=None):                   # [..., d] -> [..., n_units]
        raise NotImplementedError

    def read_sel(self, x, m, idx):               # [..., d] -> [..., k]
        raise NotImplementedError

    def directions(self, idx):                   # -> [k, d] fp32
        raise NotImplementedError

    def reader(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)

        def f(x, m=None):
            return self.read_sel(x, m, idx.to(x.device))
        return f

    def to(self, device):
        return self


class DimsBasis(Basis):
    key = "dims"

    def __init__(self, d: int, device):
        self.d = self.n_units = d
        self.device = device

    def read(self, x, m=None):
        return x.float()

    def read_sel(self, x, m, idx):
        return x.float()[..., idx]

    def directions(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        W = torch.zeros(len(idx), self.d, device=self.device)
        W[torch.arange(len(idx), device=self.device), idx] = 1.0
        return W


class PCABasis(Basis):
    key = "pca"

    def __init__(self, mu, U, evals=None):
        self.mu = mu.float()            # [d]
        self.U = U.float()              # [n, d] rows orthonormal, by decreasing variance
        self.evals = evals
        self.n_units, self.d = U.shape

    def read(self, x, m=None):
        return (x.float() - self.mu) @ self.U.T

    def read_sel(self, x, m, idx):
        return (x.float() - self.mu) @ self.U[idx].T

    def directions(self, idx):
        return self.U[torch.as_tensor(idx, dtype=torch.long, device=self.U.device)]

    def to(self, device):
        self.mu, self.U = self.mu.to(device), self.U.to(device)
        return self


class SAEBasis(Basis):
    key = "sae"

    def __init__(self, sae):
        self.sae = sae
        self.n_units, self.d = sae.n, sae.d

    def read(self, x, m=None):
        return self.sae.encode(x)

    def read_sel(self, x, m, idx):
        return self.sae.encode(x, idx)

    def directions(self, idx):
        return self.sae.directions(torch.as_tensor(idx, dtype=torch.long, device=self.sae.W_dec.device))

    def to(self, device):
        self.sae.to(device)
        return self


class MLPBasis(Basis):
    key = "mlp"
    site = "mlp_down_proj_input"
    needs_mlp = True

    def __init__(self, W_down):
        self.W = W_down.float()          # [d, f]
        self.d, self.n_units = self.W.shape

    def read(self, x, m=None):
        return m.float()

    def read_sel(self, x, m, idx):
        return m.float()[..., idx]

    def directions(self, idx):
        return self.W[:, torch.as_tensor(idx, dtype=torch.long, device=self.W.device)].T.contiguous()

    def to(self, device):
        self.W = self.W.to(device)
        return self


class DirectionBasis(Basis):
    """Projection ablation along unit directions (massmean / random_dir anchors)."""
    key = "dir"

    def __init__(self, U):
        U = U.float()
        self.U = U / U.norm(dim=-1, keepdim=True)
        self.n_units, self.d = self.U.shape

    def read(self, x, m=None):
        return x.float() @ self.U.T

    def read_sel(self, x, m, idx):
        return x.float() @ self.U[idx].T

    def directions(self, idx):
        return self.U[torch.as_tensor(idx, dtype=torch.long, device=self.U.device)]

    def to(self, device):
        self.U = self.U.to(device)
        return self


def fit_pca(X: torch.Tensor):
    """X [N, d] -> (mu [d], U [n, d] rows = components by decreasing variance, evals [n]).
    Computed in float64 on CPU; keeps min(d, N-1) components."""
    X = X.double().cpu()
    mu = X.mean(0)
    Xc = X - mu
    C = Xc.T @ Xc / max(1, X.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(C)
    order = torch.argsort(evals, descending=True)
    n = min(X.shape[1], X.shape[0] - 1)
    order = order[:n]
    return mu.float(), evecs[:, order].T.contiguous().float(), evals[order].float()


@torch.no_grad()
def masked_offset_mean(basis: Basis, X, M, keep, chunk: int = 64):
    """Mean of r over examples, per position offset, skipping masked (sink) tokens.
    X [N, T, d], M [N, T, f] | None, keep [N, T] bool -> [T, n_units] fp32."""
    N, T, _ = X.shape
    dev = X.device
    s = torch.zeros(T, basis.n_units, device=dev)
    c = torch.zeros(T, 1, device=dev)
    for i in range(0, N, chunk):
        x = X[i:i + chunk].float()
        m = None if M is None else M[i:i + chunk].float()
        k = keep[i:i + chunk].float().unsqueeze(-1)
        r = basis.read(x, m)
        s += (r * k).sum(0)
        c += k.sum(0)
    return s / c.clamp_min(1)


@torch.no_grad()
def firing_rate(basis: Basis, X, keep, chunk: int = 64):
    """Fraction of kept tokens on which each unit is non-zero ([n_units])."""
    N = X.shape[0]
    cnt = torch.zeros(basis.n_units, device=X.device)
    tot = 0.0
    for i in range(0, N, chunk):
        r = basis.read(X[i:i + chunk].float())
        k = keep[i:i + chunk]
        cnt += (r[k] != 0).float().sum(0)
        tot += float(k.sum())
    return cnt / max(tot, 1.0)
