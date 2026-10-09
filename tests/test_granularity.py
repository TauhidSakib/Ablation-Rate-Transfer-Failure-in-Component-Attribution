"""Granularity maps and the exact ANOVA (Moebius) decomposition."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(r"D:\naca") / "scripts"))

from naca.components import build_component_index  # noqa: E402
from naca.granularity import LEVELS, UnitMaskView, build_unit_map  # noqa: E402
from naca.masks import MaskSpec, MaskStore  # noqa: E402


def _index(resnet18_cifar_model):
    return build_component_index(resnet18_cifar_model, arch="resnet18_cifar", verbose=False)


def test_every_level_partitions_all_components(resnet18_cifar_model):
    idx = _index(resnet18_cifar_model)
    for lvl in LEVELS:
        um = build_unit_map(idx, lvl)
        assert um.unit_of.shape == (idx.N,)
        assert um.unit_of.min() == 0
        assert um.unit_of.max() == um.n_units - 1
        assert set(np.unique(um.unit_of)) == set(range(um.n_units)), f"{lvl}: empty unit"
        assert um.sizes().sum() == idx.N, f"{lvl}: components lost or double-counted"


def test_unit_counts_are_as_expected(resnet18_cifar_model):
    idx = _index(resnet18_cifar_model)
    assert build_unit_map(idx, "filter").n_units == idx.N
    assert build_unit_map(idx, "layer").n_units == len(idx.layers)
    # ResNet-18: 4 stages x 2 blocks + stem
    assert build_unit_map(idx, "block").n_units == 9
    fg = build_unit_map(idx, "filter_group", group_size=8)
    assert fg.n_units == sum(-(-s.n_filters // 8) for s in idx.layers)


def test_filter_groups_never_straddle_a_layer(resnet18_cifar_model):
    idx = _index(resnet18_cifar_model)
    um = build_unit_map(idx, "filter_group", group_size=8)
    layer_of = np.empty(idx.N, dtype=object)
    for s in idx.layers:
        layer_of[s.start : s.stop] = s.name
    for u in range(um.n_units):
        members = um.members(u)
        assert len(set(layer_of[members])) == 1, "a unit spans two layers"


def test_expand_maps_units_to_their_components(resnet18_cifar_model):
    idx = _index(resnet18_cifar_model)
    um = build_unit_map(idx, "block")
    u = np.ones((3, um.n_units), dtype=np.uint8)
    u[0, 0] = 0          # ablate unit 0 in row 0
    u[1, :] = 0          # ablate everything in row 1
    f = um.expand(u)
    assert f.shape == (3, idx.N)
    assert (f[0][um.members(0)] == 0).all()
    assert f[0].sum() == idx.N - um.sizes()[0]
    assert f[1].sum() == 0
    assert f[2].sum() == idx.N


def test_unit_mask_view_presents_filter_level(resnet18_cifar_model):
    idx = _index(resnet18_cifar_model)
    um = build_unit_map(idx, "layer")
    store = MaskStore.generate(MaskSpec(n_components=um.n_units, alpha=0.2, m=32, seed=0))
    view = UnitMaskView(store, um)
    assert view.N == idx.N
    assert len(view) == 32
    dense = view.unpack()
    assert dense.shape == (32, idx.N)
    # every component of an ablated unit must be ablated, and only those
    for t in range(5):
        ablated_units = np.nonzero(store.unpack(t, t + 1)[0] == 0)[0]
        expected = sum(um.sizes()[u] for u in ablated_units)
        assert idx.N - dense[t].sum() == expected

    with pytest.raises(ValueError, match="unit map"):
        UnitMaskView(MaskStore.generate(MaskSpec(5, 0.2, 4, 0)), um)


# ------------------------------------------------------- the Moebius transform
def test_mobius_matches_brute_force_definition():
    """The fast transform must equal the textbook inclusion-exclusion sum.

    beta_S = sum_{T subset of S} (-1)^{|S|-|T|} f(T).  This is the load-bearing
    computation of the exhaustive experiment, so it is checked against the
    definition rather than trusted.
    """
    from wp2b_exhaustive_coarse import enumerate_all, mobius

    rng = np.random.default_rng(0)
    G, E = 6, 3
    M = enumerate_all(G)  # noqa: F841 - the subset ordering under test
    Y = rng.normal(size=(1 << G, E))
    beta = mobius(Y, G)

    for s in range(1 << G):
        S = [i for i in range(G) if (s >> i) & 1]
        acc = np.zeros(E)
        for r in range(len(S) + 1):
            for T in itertools.combinations(S, r):
                t_idx = sum(1 << i for i in T)
                acc += ((-1) ** (len(S) - len(T))) * Y[t_idx]
        assert np.abs(beta[s] - acc).max() < 1e-9, f"subset {S} disagrees"


def test_mobius_reconstructs_the_surface_exactly():
    from wp2b_exhaustive_coarse import enumerate_all, mobius

    rng = np.random.default_rng(1)
    G, E = 7, 2
    M = enumerate_all(G)
    Y = rng.normal(size=(1 << G, E))
    beta = mobius(Y, G)
    Yhat = np.zeros_like(Y)
    for s in range(1 << G):
        ms = M[s].astype(bool)
        inc = np.all(M[:, ms] == 1, axis=1) if ms.any() else np.ones(len(M), bool)
        Yhat[inc] += beta[s][None, :]
    assert np.abs(Y - Yhat).max() < 1e-9, "full-order reconstruction is not exact"


def test_mobius_of_a_purely_additive_surface_has_no_high_orders():
    """An additive f must produce zero coefficients at every order >= 2."""
    from wp2b_exhaustive_coarse import enumerate_all, mobius

    rng = np.random.default_rng(2)
    G, E = 6, 2
    M = enumerate_all(G)
    w = rng.normal(size=(G, E))
    b = rng.normal(size=E)
    Y = M.astype(np.float64) @ w + b[None, :]
    beta = mobius(Y, G)
    order = M.sum(axis=1)
    assert np.abs(beta[order >= 2]).max() < 1e-9, "additive surface produced interactions"
    assert np.abs(beta[order == 0] - b).max() < 1e-9
    for i in range(G):
        assert np.abs(beta[1 << i] - w[i]).max() < 1e-9


def test_mobius_recovers_a_planted_pairwise_term():
    from wp2b_exhaustive_coarse import enumerate_all, mobius

    G = 5
    M = enumerate_all(G)
    Mf = M.astype(np.float64)
    Y = (2.0 * Mf[:, [0]] - 3.0 * Mf[:, [1]] + 7.0 * (Mf[:, [0]] * Mf[:, [2]]))
    beta = mobius(Y, G)
    assert abs(float(beta[1 << 0, 0]) - 2.0) < 1e-9
    assert abs(float(beta[1 << 1, 0]) + 3.0) < 1e-9
    assert abs(float(beta[(1 << 0) | (1 << 2), 0]) - 7.0) < 1e-9
    order = M.sum(axis=1)
    assert np.abs(beta[order >= 3]).max() < 1e-9


# ------------------------------------------- the orthogonal (Walsh) transform
def test_walsh_satisfies_parseval():
    """Sum of squared non-constant coefficients must equal the variance, exactly.

    This is the property the Moebius basis lacks and the reason variance shares
    are computed in the Walsh basis instead.
    """
    from wp2b_exhaustive_coarse import walsh

    rng = np.random.default_rng(0)
    G, E = 7, 3
    Y = rng.normal(size=(1 << G, E))
    f = walsh(Y, G)
    energy = (f**2).sum(axis=0) - f[0] ** 2
    assert np.abs(energy - Y.var(axis=0)).max() < 1e-9


def test_walsh_puts_all_energy_at_order_one_for_an_additive_surface():
    from wp2b_exhaustive_coarse import enumerate_all, walsh

    rng = np.random.default_rng(1)
    G, E = 6, 2
    M = enumerate_all(G)
    Y = M.astype(np.float64) @ rng.normal(size=(G, E)) + rng.normal(size=E)[None, :]
    f = walsh(Y, G)
    pc = np.array([bin(i).count("1") for i in range(1 << G)])
    total = (f**2).sum(axis=0) - f[0] ** 2
    order1 = (f[pc == 1] ** 2).sum(axis=0)
    assert np.abs(order1 / total - 1.0).max() < 1e-9, "additive surface leaked energy above order 1"


def test_walsh_cumulative_share_is_monotone_and_reaches_one():
    """The property the Moebius truncation violated on real data."""
    from wp2b_exhaustive_coarse import walsh

    rng = np.random.default_rng(2)
    G = 8
    Y = rng.normal(size=(1 << G, 1)) * 5.0
    f = walsh(Y, G)
    pc = np.array([bin(i).count("1") for i in range(1 << G)])
    total = float((f**2).sum() - float(f[0, 0]) ** 2)
    cum, prev = 0.0, -1.0
    for k in range(1, G + 1):
        cum += float((f[pc == k] ** 2).sum()) / total
        assert cum >= prev - 1e-12, f"cumulative share decreased at order {k}"
        prev = cum
    assert abs(cum - 1.0) < 1e-9


def test_walsh_locates_a_planted_pairwise_term():
    from wp2b_exhaustive_coarse import enumerate_all, walsh

    G = 5
    M = enumerate_all(G).astype(np.float64)
    Y = (3.0 * M[:, [1]] * M[:, [3]])          # pure pairwise, nothing else
    f = walsh(Y, G)
    pc = np.array([bin(i).count("1") for i in range(1 << G)])
    total = float((f**2).sum() - float(f[0, 0]) ** 2)
    # a product of two binaries carries order-1 and order-2 energy, never order>=3
    assert float((f[pc >= 3] ** 2).sum()) / total < 1e-12
    assert float((f[pc == 2] ** 2).sum()) / total > 0.2
