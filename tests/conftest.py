"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("NACA_ROOT", r"D:\naca")


def pytest_configure(config):  # noqa: ARG001
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)


class TinyConvNet(nn.Module):
    """Three conv filters in one layer, then a fixed linear head.

    Small enough that the ablation sign convention can be checked by hand
    (test 11): filter ``i`` of ``conv`` is component ``i``.
    """

    def __init__(self, n_filters: int = 3, n_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_filters, n_classes, bias=False)
        with torch.no_grad():
            # filter i multiplies the input by (i + 1)
            for i in range(n_filters):
                self.conv.weight[i, 0, 0, 0] = float(i + 1)
            self.fc.weight.copy_(torch.eye(n_classes, n_filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.conv(x)).flatten(1)
        return self.fc(h)


@pytest.fixture
def tiny_net() -> TinyConvNet:
    return TinyConvNet()


@pytest.fixture(scope="session")
def resnet18_cifar_model(device):
    from naca.models import build_model

    torch.manual_seed(0)
    model = build_model("resnet18_cifar", num_classes=10, device=device)
    model.eval()
    return model
