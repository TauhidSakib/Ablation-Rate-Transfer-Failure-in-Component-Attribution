"""Test 9 -- the O(K) greedy increment must match full re-evaluation of g."""

from __future__ import annotations

import numpy as np
import pytest

from naca.editing import (
    coar_edit_select,
    fdr_candidate_filter,
    greedy_reference_slow,
    naca_edit_select,
    surrogate_delta,
)
from naca.metrics import benjamini_hochberg


def _random_problem(E=6, K=20, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1.0, size=(E, K)).astype(np.float32)
    A = np.zeros((E, K, K), dtype=np.float32)
    for e in range(E):
        U = rng.normal(0, 0.4, size=(K, 3)).astype(np.float32)
        P = rng.normal(0, 0.4, size=(K, 3)).astype(np.float32)
        M = U @ U.T - P @ P.T
        np.fill_diagonal(M, 0.0)
        A[e] = M
    return w, A


# ------------------------------------------------------------------- test 9
def test_incremental_delta_matches_full_reevaluation():
    """Test 9: -(w_i + a_i) equals g(v with i ablated) - g(v), at every step."""
    E, K = 6, 20
    w, A = _random_problem(E, K)
    rng = np.random.default_rng(1)

    v = np.ones((E, K), dtype=np.float32)
    a = np.stack([A[e].sum(axis=1) for e in range(E)])

    for step in range(8):
        d_fast = surrogate_delta(w, A, v, a)  # (E, K)
        g_now = greedy_reference_slow(w, A, v)
        for j in range(K):
            if v[0, j] == 0:
                assert np.all(np.isnan(d_fast[:, j])), "ablated slot must be nan"
                continue
            v_try = v.copy()
            v_try[:, j] = 0.0
            d_slow = greedy_reference_slow(w, A, v_try) - g_now
            err = np.abs(d_fast[:, j] - d_slow).max()
            scale = max(float(np.abs(d_slow).max()), 1e-6)
            assert err / scale < 1e-5, (
                f"step {step}, component {j}: incremental delta differs from full "
                f"re-evaluation by {err:.3e} (relative {err / scale:.2e})"
            )
        # commit a random still-available component and update incrementally
        avail = np.nonzero(v[0] > 0)[0]
        j = int(rng.choice(avail))
        v[:, j] = 0.0
        a -= A[:, :, j]


def test_incremental_state_matches_recomputed_state():
    """The carried ``a = Av`` must not drift from a fresh recomputation."""
    E, K = 5, 16
    w, A = _random_problem(E, K, seed=3)
    v = np.ones((E, K), dtype=np.float32)
    a = np.stack([A[e].sum(axis=1) for e in range(E)])
    for j in (2, 7, 11, 0, 5):
        v[:, j] = 0.0
        a -= A[:, :, j]
        fresh = np.stack([A[e] @ v[e] for e in range(E)])
        assert np.abs(a - fresh).max() < 1e-4, "incremental a drifted from A @ v"


# ------------------------------------------------------------ NACA-EDIT
def test_greedy_selects_distinct_components_and_respects_k():
    E, K = 8, 25
    w, A = _random_problem(E, K, seed=5)
    pool = np.arange(100, 100 + K)
    sel = naca_edit_select(
        w, A, pool, np.arange(4), np.arange(4, 8), k=7, direction="increase"
    )
    assert len(sel) == 7
    assert len(set(sel.components.tolist())) == 7
    assert set(sel.components.tolist()) <= set(pool.tolist())
    assert [t["step"] for t in sel.trace] == list(range(7))


def test_greedy_improves_the_surrogate_objective_monotonically():
    """Each committed step must move the target margin the intended way."""
    E, K = 8, 25
    w, A = _random_problem(E, K, seed=6)
    pool = np.arange(K)
    tgt, ref = np.arange(4), np.arange(4, 8)

    for direction, sign in (("increase", 1.0), ("suppress", -1.0)):
        sel = naca_edit_select(
            w, A, pool, tgt, ref, k=6, direction=direction, lambda_edit=0.0
        )
        v = np.ones((E, K), dtype=np.float32)
        g0 = greedy_reference_slow(w, A, v)[tgt].mean()
        for c in sel.components:
            v[:, int(c)] = 0.0
        g1 = greedy_reference_slow(w, A, v)[tgt].mean()
        assert sign * (g1 - g0) > 0, (
            f"direction={direction}: target margin moved the wrong way "
            f"({g0:.4f} -> {g1:.4f})"
        )


def test_greedy_beats_independent_ranking_when_components_are_redundant():
    """The whole point of NACA-EDIT: a top-k ranking double-counts redundancy.

    Two components are built to be strongly redundant (a large negative
    interaction), so ablating both buys far less than their individual scores
    suggest. Independent ranking cannot see this; greedy can.
    """
    E, K = 4, 12
    rng = np.random.default_rng(11)
    w = rng.normal(0, 0.1, size=(E, K)).astype(np.float32)
    w[:, [0, 1]] = -3.0  # both look individually excellent to ablate
    A = np.zeros((E, K, K), dtype=np.float32)
    A[:, 0, 1] = A[:, 1, 0] = 6.0  # ...but they cancel when taken together
    w[:, 2] = -2.5  # a decent, non-redundant alternative

    pool = np.arange(K)
    tgt, ref = np.arange(2), np.arange(2, 4)
    sel = naca_edit_select(w, A, pool, tgt, ref, k=2, direction="increase", lambda_edit=0.0)

    v = np.ones((E, K), dtype=np.float32)
    g0 = greedy_reference_slow(w, A, v)[tgt].mean()

    def gain_of(components):
        vv = np.ones((E, K), dtype=np.float32)
        for c in components:
            vv[:, int(c)] = 0.0
        return greedy_reference_slow(w, A, vv)[tgt].mean() - g0

    naive = gain_of([0, 1])  # what independent top-k would pick
    greedy = gain_of(sel.components)
    assert greedy > naive, (
        f"greedy ({greedy:.3f}) failed to beat the redundant independent pick "
        f"({naive:.3f}); selection was {sel.components.tolist()}"
    )
    assert set(sel.components.tolist()) != {0, 1}


def test_candidate_filter_is_respected():
    E, K = 6, 20
    w, A = _random_problem(E, K, seed=7)
    pool = np.arange(K)
    allowed = pool[:5]
    sel = naca_edit_select(
        w, A, pool, np.arange(3), np.arange(3, 6), k=4, candidates=allowed
    )
    assert set(sel.components.tolist()) <= set(allowed.tolist())

    with pytest.raises(ValueError, match="no pool component"):
        naca_edit_select(
            w, A, pool, np.arange(3), np.arange(3, 6), k=2, candidates=np.array([999])
        )


# ------------------------------------------------------------ COAR-EDIT
def test_coar_edit_directions_are_opposite():
    rng = np.random.default_rng(2)
    N, E = 60, 20
    W = rng.normal(size=(N, E)).astype(np.float32)
    tgt, ref = np.arange(10), np.arange(10, 20)
    # Boost component 5 on the TARGET examples only -- boosting it everywhere
    # would leave tau near zero, which is the contrast the t-statistic measures.
    W[5, tgt] += 4.0

    up = coar_edit_select(W, tgt, ref, k=5, direction="increase")
    down = coar_edit_select(W, tgt, ref, k=5, direction="suppress")
    assert 5 in down.components.tolist(), "suppression should ablate the positive driver"
    assert 5 not in up.components.tolist()
    assert len(up) == len(down) == 5


def test_bh_fdr_is_monotone_and_bounded():
    p = np.array([0.001, 0.008, 0.02, 0.2, 0.5, 0.9])
    keep = benjamini_hochberg(p, q=0.10)
    assert keep[0] and not keep[-1]
    # BH is a step-up procedure: everything below the cutoff rank is kept
    assert list(keep) == sorted(list(keep), reverse=True)
    assert not benjamini_hochberg(np.ones(10), q=0.05).any()


def test_fdr_filter_returns_subset_and_tau():
    rng = np.random.default_rng(4)
    N, E = 80, 24
    W = rng.normal(size=(N, E)).astype(np.float32)
    W[:6] += 5.0 * (np.arange(E) < 12)[None, :]
    idx, tau = fdr_candidate_filter(W, np.arange(12), np.arange(12, 24), q=0.10)
    assert tau.shape == (N,)
    assert idx.size >= 6
    assert set(range(6)) <= set(idx.tolist()), "true signals should survive the filter"
