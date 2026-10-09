"""Test 3 -- component enumeration and the count convention."""

from __future__ import annotations

import pytest
import torch

from naca.components import (
    KNOWN_COUNTS,
    ComponentIndex,
    build_component_index,
    verify_resnet50_convention,
)


def test_resnet50_matches_paper_count():
    """The convention is pinned by ResNet-50: main-path convs only == 22,720.

    This is the anchor for the whole enumeration scheme. If it ever breaks, the
    component index no longer means what the base paper's index means.
    """
    without_ds, with_ds = verify_resnet50_convention()
    assert without_ds == 22_720, (
        f"main-path conv filters = {without_ds}, base paper says 22,720. "
        f"(with downsample convs: {with_ds})"
    )
    assert with_ds == 26_560, f"sanity: with downsample expected 26,560, got {with_ds}"


def test_resnet18_cifar_count_is_logged(resnet18_cifar_model, capsys):
    """Test 3: enumerate, and report against BOTH numbers the paper quotes.

    The base paper is internally inconsistent here (section 4 says 2,306;
    Appendix A.2 says 2,344). We resolve it empirically and record what we get
    rather than forcing a match.
    """
    index = build_component_index(
        resnet18_cifar_model, arch="resnet18_cifar", include_downsample=False
    )
    out = capsys.readouterr().out
    assert "enumerated N=" in out

    known = KNOWN_COUNTS["resnet18_cifar"]
    matches = [label for label, val in known.items() if val == index.N]
    print(
        f"\n[test] resnet18_cifar enumerated N={index.N}; "
        f"paper section 4={known['paper_section4']}, "
        f"appendix A.2={known['paper_appendixA2']}; "
        f"matches={matches or 'NEITHER'}"
    )
    # Structural invariants must hold regardless of which number the paper meant.
    assert index.N > 0
    assert index.layers[0].start == 0
    assert index.layers[-1].stop == index.N


def test_index_is_contiguous_and_covers_every_filter(resnet18_cifar_model):
    index = build_component_index(
        resnet18_cifar_model, arch="resnet18_cifar", verbose=False
    )
    total = sum(spec.n_filters for spec in index.layers)
    assert total == index.N
    cursor = 0
    for spec in index.layers:
        assert spec.start == cursor
        cursor = spec.stop
    assert cursor == index.N


def test_downsample_convs_are_excluded_by_default(resnet18_cifar_model):
    excl = build_component_index(resnet18_cifar_model, verbose=False)
    incl = build_component_index(resnet18_cifar_model, include_downsample=True, verbose=False)
    assert incl.N > excl.N
    assert not any(s.is_downsample for s in excl.layers)
    assert any(s.is_downsample for s in incl.layers)


def test_owner_and_depth_groups(resnet18_cifar_model):
    index = build_component_index(resnet18_cifar_model, verbose=False)
    for probe in (0, index.N // 2, index.N - 1):
        spec, ch = index.owner(probe)
        assert 0 <= ch < spec.n_filters
        assert spec.start + ch == probe

    groups = index.depth_group_indices()
    assert sum(len(v) for v in groups.values()) == index.N
    assert {"stem", "layer1", "layer2", "layer3", "layer4"} <= set(groups)

    arr = index.depth_group_array()
    assert arr.shape == (index.N,)
    assert arr[0] == "stem"

    with pytest.raises(IndexError):
        index.owner(index.N)


def test_expected_n_mismatch_raises(resnet18_cifar_model):
    with pytest.raises(ValueError, match="component count mismatch"):
        build_component_index(resnet18_cifar_model, expected_n=12345, verbose=False)


def test_index_round_trips_through_dict(resnet18_cifar_model):
    index = build_component_index(resnet18_cifar_model, arch="resnet18_cifar", verbose=False)
    back = ComponentIndex.from_dict(index.to_dict())
    assert back.N == index.N
    assert back.layer_names == index.layer_names


def test_empty_model_rejected():
    with pytest.raises(ValueError, match="no convolutional components"):
        build_component_index(torch.nn.Linear(4, 4), verbose=False)
