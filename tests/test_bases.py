"""PCA and dims bases are orthonormal; read/write round-trips."""

import torch

from pilot import bases as BZ


def test_pca_orthonormal_and_sorted():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(500, 12, generator=g) * torch.linspace(3, 0.2, 12)
    mu, U, ev = BZ.fit_pca(X)
    assert U.shape == (12, 12)
    assert torch.allclose(U @ U.T, torch.eye(12), atol=1e-4)
    assert torch.all(ev[:-1] >= ev[1:])


def test_dims_orthonormal():
    b = BZ.DimsBasis(8, "cpu")
    W = b.directions(torch.arange(8))
    assert torch.equal(W, torch.eye(8))


def test_pca_round_trip():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(300, 10, generator=g)
    b = BZ.PCABasis(*BZ.fit_pca(X))
    x = torch.randn(5, 10, generator=g)
    r = b.read(x)
    x_rec = b.mu + r @ b.directions(torch.arange(b.n_units))
    assert torch.allclose(x_rec, x, atol=1e-4)


def test_dims_round_trip():
    b = BZ.DimsBasis(6, "cpu")
    x = torch.randn(4, 6)
    assert torch.allclose(b.read(x) @ b.directions(torch.arange(6)), x)


def test_sae_read_write_consistent_with_decode(toy_sae):
    b = BZ.SAEBasis(toy_sae)
    x = torch.randn(7, 16) * 2
    r = b.read(x)
    assert (r >= 0).all()
    assert ((r == 0) | (r > toy_sae.threshold)).all()          # JumpReLU with the fixed threshold
    rec = toy_sae.b_dec + r @ b.directions(torch.arange(b.n_units))
    assert torch.allclose(rec, toy_sae.decode(r), atol=1e-5)
    # batch independence: a token's code does not depend on its batch
    assert torch.allclose(b.read(x[:1]), r[:1], atol=1e-6)
    idx = torch.tensor([3, 0, 11])
    assert torch.allclose(b.read_sel(x, None, idx), r[:, idx], atol=1e-6)


def test_direction_basis_projection():
    u = torch.tensor([[3.0, 4.0, 0.0]])
    b = BZ.DirectionBasis(u)
    x = torch.tensor([[1.0, 2.0, 5.0]])
    assert torch.allclose(b.read(x), torch.tensor([[(3 + 8) / 5.0]]))
    assert torch.allclose(b.directions(torch.tensor([0])).norm(), torch.tensor(1.0))


def test_masked_offset_mean_excludes_masked_tokens():
    b = BZ.DimsBasis(2, "cpu")
    X = torch.tensor([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [1000.0, 1000.0]]])
    keep = torch.tensor([[True, True], [True, False]])
    m = BZ.masked_offset_mean(b, X, None, keep)
    assert torch.allclose(m, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
