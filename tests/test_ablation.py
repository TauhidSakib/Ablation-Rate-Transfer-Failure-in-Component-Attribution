"""Tests 1, 2, 11 -- ablation round-trip, ablation is real, sign convention."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from naca.ablation import Ablator, ablated_from_mask, mask_from_ablated
from naca.components import build_component_index
from naca.metrics import correct_class_margin


# ------------------------------------------------------------------ test 11
def test_sign_convention_on_hand_built_toy_case(tiny_net):
    """Test 11: ``v_i = 0`` ablates component ``i``.

    ``TinyConvNet`` has three 1x1 filters that scale the input by 1, 2, 3, and an
    identity head, so with a constant input of 1.0 the logits are exactly
    ``[1, 2, 3]`` and ablating component ``i`` must zero **that** logit.
    """
    index = build_component_index(tiny_net, arch="tiny", verbose=False)
    assert index.N == 3
    ablator = Ablator(tiny_net, index)
    x = torch.ones(1, 1, 4, 4)

    baseline = tiny_net(x).detach().flatten()
    assert torch.allclose(baseline, torch.tensor([1.0, 2.0, 3.0]), atol=1e-6)

    for i in range(3):
        v = torch.ones(3)
        v[i] = 0.0  # ablate component i
        with ablator.ablated(v):
            out = tiny_net(x).detach().flatten()
        expected = baseline.clone()
        expected[i] = 0.0
        assert torch.allclose(out, expected, atol=1e-6), (
            f"ablating component {i} gave {out.tolist()}, expected {expected.tolist()} "
            f"-- sign convention is inverted"
        )


def test_inverted_convention_would_fail(tiny_net):
    """Guards the guard: passing the ablation *indicator* must give a different answer."""
    index = build_component_index(tiny_net, arch="tiny", verbose=False)
    ablator = Ablator(tiny_net, index)
    x = torch.ones(1, 1, 4, 4)

    v = torch.tensor([0.0, 1.0, 1.0])  # ablate component 0
    with ablator.ablated(v):
        correct = tiny_net(x).detach().flatten().clone()
    with ablator.ablated(1.0 - v):  # the wrong convention
        inverted = tiny_net(x).detach().flatten().clone()
    assert not torch.allclose(correct, inverted)


def test_mask_helpers_round_trip():
    v = mask_from_ablated([1, 3], n=5)
    assert v.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert ablated_from_mask(v).tolist() == [1, 3]
    assert mask_from_ablated([], n=4).sum() == 4
    with pytest.raises(IndexError):
        mask_from_ablated([9], n=4)


# ------------------------------------------------------------------- test 1
def test_ablation_round_trip_is_bitwise_identical(resnet18_cifar_model):
    """Test 1: apply then restore leaves every conv weight bitwise identical."""
    index = build_component_index(resnet18_cifar_model, verbose=False)
    ablator = Ablator(resnet18_cifar_model, index)
    rng = np.random.default_rng(0)

    assert ablator.weights_match_original()
    for _ in range(5):
        v = torch.ones(index.N, device=ablator.device)
        idx = rng.choice(index.N, size=index.N // 10, replace=False)
        v[torch.as_tensor(idx, device=ablator.device)] = 0.0
        ablator.apply(v)
        assert ablator.is_dirty
        ablator.restore()
        assert not ablator.is_dirty
        assert ablator.weights_match_original(), "restore is not bitwise exact"


def test_context_manager_restores_on_exception(resnet18_cifar_model):
    index = build_component_index(resnet18_cifar_model, verbose=False)
    ablator = Ablator(resnet18_cifar_model, index)
    v = torch.ones(index.N, device=ablator.device)
    v[:50] = 0.0

    with pytest.raises(RuntimeError, match="boom"), ablator.ablated(v):
        raise RuntimeError("boom")
    assert ablator.weights_match_original()


# ------------------------------------------------------------------- test 2
def test_ablation_is_real(resnet18_cifar_model, device):
    """Test 2: 0% ablation is a no-op; 100% ablation changes the margin."""
    index = build_component_index(resnet18_cifar_model, verbose=False)
    ablator = Ablator(resnet18_cifar_model, index)

    torch.manual_seed(0)
    x = torch.randn(16, 3, 32, 32, device=device).contiguous(
        memory_format=torch.channels_last
    )
    y = torch.randint(0, 10, (16,), device=device)

    @torch.inference_mode()
    def margins() -> torch.Tensor:
        return correct_class_margin(resnet18_cifar_model(x).float(), y)

    base = margins()

    keep_all = torch.ones(index.N, device=device)
    with ablator.ablated(keep_all):
        same = margins()
    assert torch.allclose(base, same, atol=1e-5, rtol=0), "0% ablation must be a no-op"

    kill_all = torch.zeros(index.N, device=device)
    with ablator.ablated(kill_all):
        dead = margins()
    assert not torch.allclose(base, dead, atol=1e-3), "100% ablation changed nothing"

    after = margins()
    assert torch.allclose(base, after, atol=1e-5, rtol=0), "state leaked after restore"


def test_partial_ablation_changes_output_monotonically_often(resnet18_cifar_model, device):
    """Ablating more filters should, on average, move the margin further."""
    index = build_component_index(resnet18_cifar_model, verbose=False)
    ablator = Ablator(resnet18_cifar_model, index)
    torch.manual_seed(1)
    x = torch.randn(8, 3, 32, 32, device=device).contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (8,), device=device)

    @torch.inference_mode()
    def margins():
        return correct_class_margin(resnet18_cifar_model(x).float(), y)

    base = margins()
    rng = np.random.default_rng(3)
    drift = []
    for frac in (0.05, 0.25, 0.75):
        v = torch.ones(index.N, device=device)
        idx = rng.choice(index.N, size=int(frac * index.N), replace=False)
        v[torch.as_tensor(idx, device=device)] = 0.0
        with ablator.ablated(v):
            drift.append(float((margins() - base).abs().mean()))
    assert drift[0] < drift[-1], f"expected larger drift with more ablation, got {drift}"


# ---------------------------------------------------------------- guards
def test_non_binary_mask_is_rejected(tiny_net):
    index = build_component_index(tiny_net, arch="tiny", verbose=False)
    ablator = Ablator(tiny_net, index)
    with pytest.raises(ValueError, match="binary"):
        ablator.apply(torch.tensor([0.5, 1.0, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        ablator.apply(torch.ones(7))


def test_index_model_mismatch_is_caught(tiny_net, resnet18_cifar_model):
    bad_index = build_component_index(resnet18_cifar_model, verbose=False)
    with pytest.raises(KeyError, match="out of sync"):
        Ablator(tiny_net, bad_index)


def test_numpy_mask_accepted(tiny_net):
    index = build_component_index(tiny_net, arch="tiny", verbose=False)
    ablator = Ablator(tiny_net, index)
    ablator.apply(np.array([1, 0, 1], dtype=np.float32))
    ablator.restore()
    assert ablator.weights_match_original()
