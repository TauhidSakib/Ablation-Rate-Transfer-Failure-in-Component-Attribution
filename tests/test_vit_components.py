"""ViT component enumeration and ablation (WP9).

The component *type* is what WP9 changes, so these are the load-bearing checks:
if head ablation zeroes the wrong columns, every WP9 number is wrong and nothing
downstream would notice -- the pipeline would return a well-typed answer to a
question it was not asked.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from naca.ablation import Ablator
from naca.components import ComponentIndex, build_component_index
from naca.vit import vit_tiny_cifar


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return vit_tiny_cifar(num_classes=10).eval()


@pytest.fixture(scope="module")
def index(model):
    return build_component_index(model, arch="vit_tiny_cifar", verbose=False)


def test_component_count_is_612(index):
    """12 blocks x (3 heads + 48 MLP groups). Pre-registered in WP8_WP9_prereg.md."""
    assert index.N == 612
    kinds = {s.kind for s in index.layers}
    assert kinds == {"attn_head", "mlp_group"}
    assert sum(s.n_filters for s in index.layers if s.kind == "attn_head") == 36
    assert sum(s.n_filters for s in index.layers if s.kind == "mlp_group") == 576


def test_blocks_are_contiguous_runs(index):
    """The coarse granularity (12 units) needs each block contiguous."""
    groups = index.depth_group_indices()
    assert len(groups) == 12
    for g, idx in groups.items():
        assert idx.max() - idx.min() + 1 == len(idx), f"{g} is not contiguous"
        assert len(idx) == 51


def test_round_trip_is_bitwise_identical(model, index):
    abl = Ablator(model, index)
    before = abl.original_weight_checksum()
    v = torch.ones(index.N)
    v[[0, 5, 100, 611]] = 0.0
    abl.apply(v)
    abl.restore()
    assert abl.weights_match_original()
    assert abl.original_weight_checksum() == before


def test_ablating_one_head_zeroes_exactly_its_proj_columns(model, index):
    """Head h of block i owns proj.weight[:, h*head_dim:(h+1)*head_dim]."""
    abl = Ablator(model, index)
    head_dim = model.head_dim
    spec = next(s for s in index.layers if s.name == "blocks.3.attn.proj")
    target = spec.start + 1  # head 1 of block 3
    v = torch.ones(index.N)
    v[target] = 0.0
    try:
        abl.apply(v)
        w = dict(model.named_modules())["blocks.3.attn.proj"].weight.data
        lo, hi = 1 * head_dim, 2 * head_dim
        assert torch.all(w[:, lo:hi] == 0), "head's own columns not zeroed"
        assert torch.all(w[:, :lo] != 0), "columns before the head were touched"
        assert torch.all(w[:, hi:] != 0), "columns after the head were touched"
    finally:
        abl.restore()


def test_ablating_one_mlp_group_zeroes_exactly_its_fc2_columns(model, index):
    abl = Ablator(model, index)
    width = 16
    spec = next(s for s in index.layers if s.name == "blocks.7.mlp.fc2")
    g = 5
    v = torch.ones(index.N)
    v[spec.start + g] = 0.0
    try:
        abl.apply(v)
        w = dict(model.named_modules())["blocks.7.mlp.fc2"].weight.data
        lo, hi = g * width, (g + 1) * width
        assert torch.all(w[:, lo:hi] == 0)
        assert torch.all(w[:, :lo] != 0)
        assert torch.all(w[:, hi:] != 0)
    finally:
        abl.restore()


def test_components_touch_disjoint_weights(model, index):
    """Ablating i must not disturb any weight owned by j != i."""
    abl = Ablator(model, index)
    v_all = torch.ones(index.N)
    try:
        abl.apply(v_all)
        assert abl.weights_match_original(), "all-ones mask must be a no-op"
    finally:
        abl.restore()

    # union of two single ablations == joint ablation
    a, b = 4, 300
    snaps = []
    for sel in ([a], [b], [a, b]):
        v = torch.ones(index.N)
        v[sel] = 0.0
        abl.apply(v)
        snaps.append({n: m.weight.data.clone()
                      for n, m in model.named_modules()
                      if n in {s.name for s in index.layers}})
        abl.restore()
    za, zb, zab = ({n: (w == 0) for n, w in s.items()} for s in snaps)
    for n in za:
        assert torch.equal(za[n] | zb[n], zab[n]), f"{n}: ablations are not disjoint/additive"


def test_ablation_changes_the_output(model, index):
    torch.manual_seed(1)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y0 = model(x).clone()
    abl = Ablator(model, index)
    v = torch.ones(index.N)
    v[index.N // 2] = 0.0
    try:
        abl.apply(v)
        with torch.no_grad():
            y1 = model(x)
        assert not torch.allclose(y0, y1), "ablating a component did nothing"
    finally:
        abl.restore()
    with torch.no_grad():
        assert torch.allclose(y0, model(x)), "restore did not recover the output"


def test_index_survives_serialisation(index):
    rt = ComponentIndex.from_dict(index.to_dict())
    assert rt.N == index.N
    for a, b in zip(index.layers, rt.layers):
        assert (a.name, a.param_axis, a.unit_width, a.kind) == (
            b.name, b.param_axis, b.unit_width, b.kind
        )
