"""Test 4 -- the Cholesky and matrix-free CG paths must agree.

Read `results/FINDING_gauge_degeneracy.md` before changing anything here. Because
every mask ablates exactly ``k`` components, every row of ``X`` sums to ``N - k``
and the all-ones vector is *exactly* in the null space of the centered design.
So ``w`` is identifiable only up to ``w -> w + c*1`` (with ``b`` absorbing the
rest), and the raw coefficients of two different solvers may legitimately differ
by a constant while agreeing on everything observable.

These tests therefore assert on the identifiable quantities -- shift-removed
coefficients and predictions -- and separately pin the degeneracy itself, so
that if the sampler ever changes the assumption breaks loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from naca.coar import fit_coar, predict_chunked, residuals_chunked
from naca.masks import MaskSpec, MaskStore
from naca.metrics import r2_score
from naca.solver import fit_ridge, normal_equations

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _degauge(W: np.ndarray) -> np.ndarray:
    """Remove the unidentifiable constant shift: centre coefficients per example."""
    return W - W.mean(axis=0, keepdims=True)


@pytest.fixture(scope="module")
def synthetic():
    """A linear problem with known coefficients, N=200."""
    rng = np.random.default_rng(0)
    N, E, m = 200, 5, 8_000
    masks = MaskStore.generate(MaskSpec(n_components=N, alpha=0.10, m=m, seed=7))
    X = masks.unpack().astype(np.float32)
    W = rng.normal(0, 1.0, size=(N, E)).astype(np.float32)
    b = rng.normal(0, 0.5, size=E).astype(np.float32)
    Y = X @ W + b[None, :] + rng.normal(0, 0.05, size=(m, E)).astype(np.float32)
    return masks, X, W, b, Y


# ------------------------------------------------- the degeneracy itself
def test_design_matrix_has_an_exact_null_direction(synthetic):
    """Fixed-size subset sampling makes the centered Gram rank N-1, exactly."""
    masks, X, _, _, Y = synthetic
    row_sums = np.unique(X.sum(axis=1))
    assert row_sums.size == 1, f"rows do not have constant sum: {row_sums[:5]}"
    assert row_sums[0] == masks.N - masks.spec.k_ablated

    ne = normal_equations(masks, Y, device=DEV, chunk=1000)
    G = ne.Gxx
    ones = torch.ones(G.shape[0], dtype=G.dtype, device=G.device)
    rel = float(torch.linalg.vector_norm(G @ ones)) / float(torch.linalg.matrix_norm(G))
    assert rel < 1e-10, f"expected an exact null direction, got relative norm {rel:.2e}"


# ------------------------------------------------------------------- test 4
def test_cholesky_and_cg_agree(synthetic):
    """Test 4: both solver paths agree to 1e-5 on everything identifiable."""
    masks, X, _, _, Y = synthetic
    lam = 1.0
    chol = fit_ridge(masks, Y, lam, method="cholesky", device=DEV)
    cg = fit_ridge(masks, Y, lam, method="cg", device=DEV, tol=1e-12, max_iter=3000)
    assert cg.info["converged"], f"CG did not converge: {cg.info}"

    scale = max(float(np.abs(chol.W).max()), 1e-8)
    d_gauged = float(np.abs(_degauge(chol.W) - _degauge(cg.W)).max()) / scale
    assert d_gauged < 1e-5, f"identifiable coefficients disagree: {d_gauged:.2e}"

    p_chol = X @ chol.W + chol.b[None, :]
    p_cg = X @ cg.W + cg.b[None, :]
    d_pred = float(np.abs(p_chol - p_cg).max()) / max(float(p_chol.std()), 1e-8)
    assert d_pred < 1e-5, f"predictions disagree: {d_pred:.2e}"


def test_solver_disagreement_lives_only_in_the_null_direction(synthetic):
    """The raw-coefficient gap must be explained by the constant shift."""
    masks, _, _, _, Y = synthetic
    chol = fit_ridge(masks, Y, 1.0, method="cholesky", device=DEV)
    cg = fit_ridge(masks, Y, 1.0, method="cg", device=DEV, tol=1e-12, max_iter=3000)
    raw = float(np.abs(chol.W - cg.W).max())
    gauged = float(np.abs(_degauge(chol.W) - _degauge(cg.W)).max())
    assert gauged < raw / 10, (
        f"removing the constant shift did not explain the gap "
        f"(raw={raw:.2e}, shift-removed={gauged:.2e})"
    )


def test_recovers_known_coefficients_up_to_the_gauge(synthetic):
    """Recovery is exact only modulo the unidentifiable shift -- assert both halves."""
    masks, _, W, _, Y = synthetic
    sol = fit_ridge(masks, Y, 1e-3, method="cholesky", device=DEV)
    D = sol.W - W
    assert D.std(axis=0).max() < 0.02, (
        f"residual is not a pure constant shift (per-example std {D.std(axis=0)})"
    )
    assert np.abs(_degauge(sol.W) - _degauge(W)).max() < 0.05


def test_predictions_are_recovered_even_though_coefficients_are_not(synthetic):
    masks, X, _, _, Y = synthetic
    sol = fit_ridge(masks, Y, 1e-3, method="cholesky", device=DEV)
    pred = X @ sol.W + sol.b[None, :]
    assert float(np.nanmean(r2_score(Y, pred))) > 0.999


def test_intercept_is_not_regularized(synthetic):
    """A huge ridge crushes the coefficients but must leave the bias free."""
    masks, _, _, _, Y = synthetic
    sol = fit_ridge(masks, Y, 1e8, method="cholesky", device=DEV)
    assert np.abs(sol.W).max() < 1e-2, "coefficients should be crushed by lambda=1e8"
    assert np.allclose(sol.b, Y.mean(axis=0), rtol=0, atol=1e-2), (
        f"intercept collapsed under regularization: got {sol.b}, "
        f"expected column means {Y.mean(axis=0)}"
    )


def test_centered_normal_equations_match_dense_reference(synthetic):
    """The chunked accumulation must equal an explicit centered computation."""
    masks, X, _, _, Y = synthetic
    ne = normal_equations(masks, Y, device=DEV, chunk=1000)
    Xd, Yd = X.astype(np.float64), Y.astype(np.float64)
    Xc = Xd - Xd.mean(axis=0, keepdims=True)
    Yc = Yd - Yd.mean(axis=0, keepdims=True)

    G_ref, rhs_ref = Xc.T @ Xc, Xc.T @ Yc
    assert np.abs(ne.Gxx.cpu().numpy() - G_ref).max() / np.abs(G_ref).max() < 1e-12
    # rhs uses fp32 matmuls with fp64 accumulation across chunks
    assert np.abs(ne.rhs.cpu().numpy() - rhs_ref).max() / np.abs(rhs_ref).max() < 1e-5
    assert np.abs(ne.diag.cpu().numpy() - (Xc * Xc).sum(axis=0)).max() < 1e-6
    assert np.allclose(ne.colsum.cpu().numpy(), Xd.sum(axis=0), atol=1e-9)


def test_chunk_size_does_not_change_the_identifiable_answer(synthetic):
    masks, X, _, _, Y = synthetic
    a = fit_ridge(masks, Y, 1.0, method="cholesky", device=DEV, chunk=997)
    b_ = fit_ridge(masks, Y, 1.0, method="cholesky", device=DEV, chunk=4096)
    scale = max(float(np.abs(b_.W).max()), 1e-8)
    assert float(np.abs(_degauge(a.W) - _degauge(b_.W)).max()) / scale < 1e-5
    pa, pb = X @ a.W + a.b[None, :], X @ b_.W + b_.b[None, :]
    assert float(np.abs(pa - pb).max()) / max(float(pb.std()), 1e-8) < 1e-5


def test_auto_method_selection(synthetic):
    masks, _, _, _, Y = synthetic
    assert fit_ridge(masks, Y, 1.0, method="auto", device=DEV, cholesky_threshold=10).info[
        "method"
    ] == "cg"
    assert fit_ridge(
        masks, Y, 1.0, method="auto", device=DEV, cholesky_threshold=10_000
    ).info["method"] == "cholesky"


def test_coar_wrapper_predict_and_residuals(synthetic):
    masks, X, _, _, Y = synthetic
    model = fit_coar(masks, Y, lam=1.0, device=DEV)
    pred = predict_chunked(model.W, model.b, masks, device=DEV)
    assert np.allclose(pred, X @ model.W + model.b[None, :], rtol=1e-3, atol=1e-3)

    R = residuals_chunked(model.W, model.b, masks, Y, device=DEV)
    assert R.shape == Y.shape
    assert np.allclose(R, Y - pred, atol=1e-5)
    assert float(np.nanmean(r2_score(Y, pred))) > 0.99


def test_bad_shapes_rejected(synthetic):
    masks, _, _, _, Y = synthetic
    with pytest.raises(ValueError, match="rows"):
        fit_ridge(masks, Y[:10], 1.0, device=DEV)
    with pytest.raises(ValueError, match="unknown method"):
        fit_ridge(masks, Y, 1.0, method="banana", device=DEV)
