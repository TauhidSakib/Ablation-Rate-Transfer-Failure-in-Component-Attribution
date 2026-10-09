"""Tests 5, 6, 7, 8 -- the estimator.

Test 5 (synthetic recovery) is the single most important test in the project:
it is what says the Stage-1 + Stage-3 pipeline can actually recover a known
interaction structure, and therefore that a measured NAF means something.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from naca.coar import fit_coar, predict_chunked, residuals_chunked
from naca.interaction import IndefiniteFM, fit_interaction, quad_naive
from naca.masks import MaskSpec, MaskStore
from naca.metrics import non_additive_fraction, r2_score
from naca.screening import residual_scores, select_pool

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------- test 6, 7
def test_fm_identity_matches_naive_double_loop():
    """Test 6: the O(Kr) FM identity equals the O(K^2) double sum."""
    torch.manual_seed(0)
    G, K, B = 3, 40, 64
    fm = IndefiniteFM(G, K, r_U=5, r_P=4, init_std=0.5)
    V = (torch.rand(B, G, K) < 0.85).float()

    fast = fm(V)  # (B, G)
    A = fm.interaction_matrix()  # (G, K, K)
    for g in range(G):
        slow = quad_naive(V[:, g, :], A[g])
        err = float((fast[:, g] - slow).abs().max())
        rel = err / max(float(slow.abs().max()), 1e-8)
        assert rel < 1e-5, f"group {g}: FM identity differs from naive sum by {rel:.2e}"


def test_diagonal_is_excluded():
    """Test 7: diag(A) == 0 exactly."""
    torch.manual_seed(1)
    fm = IndefiniteFM(4, 25, r_U=6, r_P=6, init_std=0.7)
    A = fm.interaction_matrix()
    d = torch.diagonal(A, dim1=-2, dim2=-1)
    assert torch.all(d == 0), f"diag(A) is not zero: max |diag| = {float(d.abs().max())}"
    assert torch.allclose(A, A.transpose(-1, -2), atol=1e-6), "A must be symmetric"


def test_diagonal_does_not_leak_into_the_quadratic():
    """A single kept component must contribute nothing to the quadratic term."""
    torch.manual_seed(2)
    fm = IndefiniteFM(1, 12, r_U=4, r_P=3, init_std=0.9)
    for i in range(12):
        V = torch.zeros(1, 1, 12)
        V[0, 0, i] = 1.0
        assert abs(float(fm(V))) < 1e-5, f"lone component {i} produced a quadratic term"


def test_psd_ablation_arm_has_no_negative_block():
    fm = IndefiniteFM(2, 16, r_U=4, r_P=0, init_std=0.5)
    A = fm.interaction_matrix(0)
    off_diag = A + torch.diag(torch.full((16,), float("nan")))
    assert torch.isfinite(off_diag[~torch.isnan(off_diag)]).all()
    # A = UU' with the diagonal removed; its full form must be PSD before removal
    ev = torch.linalg.eigvalsh(fm._U[0] @ fm._U[0].T)
    assert float(ev.min()) > -1e-5, "PSD arm produced a negative eigenvalue"


# -------------------------------------------------------- synthetic problem
def _make_synthetic(
    *,
    N=200,
    K_true=20,
    E=4,
    m=20_000,
    alpha=0.10,
    r_true=3,
    seed=0,
    quad_scale=1.0,
):
    """Ground-truth ``y = b + w'v + sum_{i<j in pool} A_ij v_i v_j``.

    ``A`` is built as ``UU' - PP'`` with distinct blocks, so it is genuinely
    indefinite and a PSD-only model cannot represent it.
    """
    rng = np.random.default_rng(seed)
    masks = MaskStore.generate(MaskSpec(n_components=N, alpha=alpha, m=m, seed=seed + 1))
    X = masks.unpack().astype(np.float32)

    true_pool = np.sort(rng.choice(N, size=K_true, replace=False))
    W = rng.normal(0, 1.0, size=(N, E)).astype(np.float32)
    b = rng.normal(0, 0.5, size=E).astype(np.float32)

    A_true = np.zeros((E, K_true, K_true), dtype=np.float32)
    for e in range(E):
        U = rng.normal(0, 1.0, size=(K_true, r_true)).astype(np.float32)
        P = rng.normal(0, 1.0, size=(K_true, r_true)).astype(np.float32)
        A = U @ U.T - P @ P.T
        np.fill_diagonal(A, 0.0)
        A_true[e] = quad_scale * A

    Vp = X[:, true_pool]  # (m, K_true)
    quad = np.einsum("mk,ekl,ml->me", Vp, A_true, Vp, optimize=True) * 0.5
    lin = X @ W + b[None, :]
    Y = (lin + quad).astype(np.float32)
    return {
        "masks": masks,
        "X": X,
        "W": W,
        "b": b,
        "pool": true_pool,
        "A_true": A_true,
        "Y": Y,
        "lin": lin,
        "quad": quad,
    }


# ------------------------------------------------------------------- test 5
@pytest.mark.slow
def test_synthetic_recovery_r2_above_099():
    """Test 5: Stage 1 + Stage 3 recover a known (w, A) with held-out R^2 > 0.99.

    Held-out, not in-sample: the model is fitted on one mask set and scored on a
    disjoint one drawn at the same alpha.
    """
    d = _make_synthetic(m=20_000, quad_scale=1.0)
    masks, Y, E = d["masks"], d["Y"], d["W"].shape[1]
    K_true = len(d["pool"])

    # the quadratic must actually matter, or this test proves nothing
    share = d["quad"].var(axis=0) / Y.var(axis=0)
    assert share.min() > 0.02, f"quadratic share of variance too small: {share}"

    coar = fit_coar(masks, Y, lam=1e-3, device=DEV)
    lin_pred = predict_chunked(coar.W, coar.b, masks, device=DEV)
    r2_lin_train = float(np.nanmean(r2_score(Y, lin_pred)))

    pool = np.broadcast_to(d["pool"], (E, K_true)).copy()
    model = fit_interaction(
        masks,
        Y,
        lin_pred,
        pool,
        coar.W,
        coar.b,
        r_U=4,
        r_P=4,
        lambda_fm=1e-6,
        lr=2e-2,
        epochs=80,
        group_size=E,
        device=DEV,
        seed=0,
        verbose=False,
    )

    # ---- held-out evaluation on a disjoint mask set ----------------------
    test_masks = MaskStore.generate(
        MaskSpec(n_components=masks.N, alpha=masks.spec.alpha, m=4_000, seed=999, kind="test")
    )
    Xt = test_masks.unpack().astype(np.float32)
    Vp = Xt[:, d["pool"]]
    quad_t = np.einsum("mk,ekl,ml->me", Vp, d["A_true"], Vp, optimize=True) * 0.5
    Y_test = (Xt @ d["W"] + d["b"][None, :] + quad_t).astype(np.float32)

    lin_test = predict_chunked(coar.W, coar.b, test_masks, device=DEV)
    naca_test = model.predict(test_masks, lin_test, device=DEV, group_size=E)

    r2_lin = r2_score(Y_test, lin_test)
    r2_naca = r2_score(Y_test, naca_test)
    naf = non_additive_fraction(r2_lin, r2_naca)

    print(
        f"\n[test5] in-sample linear R2={r2_lin_train:.4f} | held-out "
        f"linear R2={np.nanmean(r2_lin):.4f} -> NACA R2={np.nanmean(r2_naca):.4f} "
        f"(NAF={np.nanmean(naf):.3f})"
    )
    assert float(np.nanmean(r2_naca)) > 0.99, (
        f"synthetic recovery failed: held-out NACA R2={np.nanmean(r2_naca):.4f} "
        f"(linear was {np.nanmean(r2_lin):.4f})"
    )
    assert float(np.nanmean(r2_naca)) > float(np.nanmean(r2_lin))


# ------------------------------------------------------------------- test 8
@pytest.mark.slow
def test_fitted_A_is_indefinite():
    """Test 8: fitting genuinely indefinite data yields eigenvalues of both signs.

    A PSD-only factorization cannot do this, which is exactly what makes the
    indefinite/PSD ablation row meaningful.
    """
    d = _make_synthetic(E=2, m=15_000, r_true=3, quad_scale=1.5, seed=5)
    E, K_true = 2, len(d["pool"])
    ev_true = np.linalg.eigvalsh(d["A_true"][0])
    assert ev_true.min() < 0 < ev_true.max(), "the synthetic target is not indefinite"

    coar = fit_coar(d["masks"], d["Y"], lam=1e-3, device=DEV)
    lin_pred = predict_chunked(coar.W, coar.b, d["masks"], device=DEV)
    model = fit_interaction(
        d["masks"],
        d["Y"],
        lin_pred,
        np.broadcast_to(d["pool"], (E, K_true)).copy(),
        coar.W,
        coar.b,
        r_U=4,
        r_P=4,
        lambda_fm=1e-6,
        lr=2e-2,
        epochs=30,
        group_size=E,
        device=DEV,
        verbose=False,
    )
    for e in range(E):
        ev = np.linalg.eigvalsh(model.interaction_matrix(e))
        assert ev.min() < -1e-3, f"example {e}: no negative eigenvalue (min={ev.min():.3e})"
        assert ev.max() > 1e-3, f"example {e}: no positive eigenvalue (max={ev.max():.3e})"


@pytest.mark.slow
def test_psd_arm_underfits_indefinite_data():
    """The PSD ablation must do measurably worse -- that is the point of the arm."""
    d = _make_synthetic(E=2, m=15_000, r_true=3, quad_scale=1.5, seed=5)
    E, K_true = 2, len(d["pool"])
    pool = np.broadcast_to(d["pool"], (E, K_true)).copy()
    coar = fit_coar(d["masks"], d["Y"], lam=1e-3, device=DEV)
    lin_pred = predict_chunked(coar.W, coar.b, d["masks"], device=DEV)

    common = dict(
        lambda_fm=1e-6, lr=2e-2, epochs=30, group_size=E, device=DEV, verbose=False
    )
    indef = fit_interaction(
        d["masks"], d["Y"], lin_pred, pool, coar.W, coar.b, r_U=4, r_P=4, **common
    )
    psd = fit_interaction(
        d["masks"], d["Y"], lin_pred, pool, coar.W, coar.b, r_U=8, r_P=0, **common
    )

    r2_indef = np.nanmean(r2_score(d["Y"], indef.predict(d["masks"], lin_pred, device=DEV)))
    r2_psd = np.nanmean(r2_score(d["Y"], psd.predict(d["masks"], lin_pred, device=DEV)))
    print(f"\n[test8b] indefinite R2={r2_indef:.4f}  PSD R2={r2_psd:.4f}")
    assert r2_indef > r2_psd, (
        f"indefinite ({r2_indef:.4f}) did not beat PSD ({r2_psd:.4f}) on indefinite data, "
        f"despite matched parameter count"
    )


# ------------------------------------------------------------------ screening
@pytest.mark.slow
def test_residual_screening_finds_the_true_pool():
    """Residual screening should rank the interacting components highly."""
    d = _make_synthetic(N=200, K_true=20, E=3, m=20_000, quad_scale=1.5, seed=11)
    coar = fit_coar(d["masks"], d["Y"], lam=1e-3, device=DEV)
    R = residuals_chunked(coar.W, coar.b, d["masks"], d["Y"], device=DEV)
    S = residual_scores(d["masks"], R, device=DEV)

    pool = select_pool(coar.W, S, K=40, mode="residual")
    truth = set(d["pool"].tolist())
    recall = [len(truth & set(pool[e].tolist())) / len(truth) for e in range(pool.shape[0])]
    print(f"\n[screen] recall of the true pool at K=40: {recall}")
    assert min(recall) > 0.5, f"residual screening missed most of the true pool: {recall}"


def test_union_screening_returns_exactly_K_distinct():
    rng = np.random.default_rng(0)
    N, E, K = 120, 4, 30
    W = rng.normal(size=(N, E)).astype(np.float32)
    S = rng.random((N, E)).astype(np.float32)
    for mode in ("marginal", "residual", "union"):
        pool = select_pool(W, S, K, mode=mode)
        assert pool.shape == (E, K)
        for e in range(E):
            assert len(set(pool[e].tolist())) == K, f"{mode} produced duplicates"


# ------------------------------------------- fit path vs predict path
def test_predict_path_agrees_with_the_fitting_path():
    """``InteractionModel.predict`` and ``IndefiniteFM.forward`` must agree.

    They compute the same quantity two different ways: the fit path uses the
    O(Kr) FM identity on the factors, the predict path uses 0.5*v'Av on the
    materialized matrix. Both are exercised in every run -- the estimator is
    fitted through one and scored through the other -- so a discrepancy would
    silently corrupt every reported R^2 while every other test still passed.
    """
    from naca.interaction import InteractionModel

    torch.manual_seed(3)
    N, K, E, m = 60, 12, 4, 300
    masks = MaskStore.generate(MaskSpec(n_components=N, alpha=0.2, m=m, seed=4))
    pool = np.sort(np.random.default_rng(0).choice(N, size=K, replace=False))
    pool_e = np.broadcast_to(pool, (E, K)).copy()

    fm = IndefiniteFM(E, K, r_U=3, r_P=2, init_std=0.6)
    A = fm.interaction_matrix().numpy()

    model = InteractionModel(
        A=A.astype(np.float32),
        pool=pool_e,
        w_pool=np.zeros((E, K), dtype=np.float32),
        b=np.zeros(E, dtype=np.float32),
        shared_pool=False,
        freeze_linear=True,
    )
    lin = np.zeros((m, E), dtype=np.float32)
    via_predict = model.predict(masks, lin, device=DEV, group_size=E)

    X = torch.from_numpy(masks.unpack().astype(np.float32))
    V = torch.stack([X[:, torch.from_numpy(pool_e[e]).long()] for e in range(E)], dim=1)
    via_fit = fm(V).detach().numpy()

    err = np.abs(via_predict - via_fit).max()
    scale = max(float(np.abs(via_fit).max()), 1e-8)
    assert err / scale < 1e-5, (
        f"predict path and fit path disagree by {err:.3e} (relative {err / scale:.2e}) "
        f"-- 0.5*v'Av and the FM identity must give the same number"
    )


def test_predict_adds_the_linear_baseline_through_unchanged():
    """predict() must return lin_pred + quad, not overwrite lin_pred."""
    from naca.interaction import InteractionModel

    N, K, E, m = 40, 8, 3, 200
    masks = MaskStore.generate(MaskSpec(n_components=N, alpha=0.25, m=m, seed=6))
    pool_e = np.broadcast_to(np.arange(K), (E, K)).copy()
    model = InteractionModel(
        A=np.zeros((E, K, K), dtype=np.float32),  # no interactions at all
        pool=pool_e,
        w_pool=np.zeros((E, K), dtype=np.float32),
        b=np.zeros(E, dtype=np.float32),
        shared_pool=False,
        freeze_linear=True,
    )
    lin = np.random.default_rng(1).normal(size=(m, E)).astype(np.float32)
    out = model.predict(masks, lin, device=DEV, group_size=E)
    assert np.abs(out - lin).max() < 1e-6, "zero A must leave the linear prediction untouched"
