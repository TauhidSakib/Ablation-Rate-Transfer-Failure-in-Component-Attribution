"""Datasets, backdoor injection, and GPU-resident tensor batches.

Windows forces ``num_workers=0`` (hard rule 1), which makes a normal DataLoader
the bottleneck for small images. CIFAR-10 is only 150 MB as uint8, so we skip
DataLoader entirely: the whole split lives on the GPU and augmentation is done
there too. This is the same trick that replaces FFCV during collection
(CLAUDE.md 5.1) -- once the data is resident, the run is purely compute-bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import BackdoorConfig

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "GPUTensorDataset",
    "load_cifar10",
    "load_cifar100",
    "load_dataset",
    "num_classes_for",
    "apply_backdoor_patch",
    "make_backdoor_train_split",
    "normalize",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class GPUTensorDataset:
    """A whole split resident on the GPU as uint8 ``(n, 3, H, W)`` + int64 labels."""

    images: torch.Tensor  # uint8, (n, 3, H, W)
    labels: torch.Tensor  # int64, (n,)
    mean: torch.Tensor  # (1, 3, 1, 1) float32
    std: torch.Tensor  # (1, 3, 1, 1) float32
    name: str = ""
    poisoned: torch.Tensor | None = None  # bool (n,), set for backdoor splits

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def device(self) -> torch.device:
        return self.images.device

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.images.shape[-2]), int(self.images.shape[-1])

    def nbytes(self) -> int:
        return self.images.numel() + self.labels.numel() * 8

    def to(self, device: torch.device | str) -> GPUTensorDataset:
        return GPUTensorDataset(
            images=self.images.to(device),
            labels=self.labels.to(device),
            mean=self.mean.to(device),
            std=self.std.to(device),
            name=self.name,
            poisoned=None if self.poisoned is None else self.poisoned.to(device),
        )

    def subset(self, idx: torch.Tensor | np.ndarray) -> GPUTensorDataset:
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.images.device)
        return GPUTensorDataset(
            images=self.images[idx],
            labels=self.labels[idx],
            mean=self.mean,
            std=self.std,
            name=f"{self.name}[subset:{idx.numel()}]",
            poisoned=None if self.poisoned is None else self.poisoned[idx],
        )

    # ------------------------------------------------------------------
    def normalized(
        self,
        lo: int = 0,
        hi: int | None = None,
        *,
        channels_last: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Slice ``[lo, hi)`` as a normalized float tensor ready for the model."""
        hi = len(self) if hi is None else hi
        x = self.images[lo:hi].to(dtype=torch.float32).div_(255.0)
        x = (x - self.mean) / self.std
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x.to(dtype=dtype)

    def normalized_all(
        self, *, channels_last: bool = True, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return self.normalized(0, len(self), channels_last=channels_last, dtype=dtype)


def normalize(
    images_uint8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    x = images_uint8.float().div_(255.0)
    return (x - mean) / std


# --------------------------------------------------------------------------


def load_cifar10(
    root: str | Path,
    *,
    train: bool,
    device: torch.device | str = "cuda",
    download: bool = True,
) -> GPUTensorDataset:
    """Load a CIFAR-10 split straight into GPU memory as uint8."""
    from torchvision.datasets import CIFAR10

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ds = CIFAR10(root=str(root), train=train, download=download)

    # ds.data is (n, 32, 32, 3) uint8 numpy
    images = torch.from_numpy(np.ascontiguousarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(ds.targets, dtype=torch.long)

    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)

    out = GPUTensorDataset(
        images=images,
        labels=labels,
        mean=mean,
        std=std,
        name=f"cifar10-{'train' if train else 'test'}",
    )
    return out.to(device)


def load_cifar100(
    root: str | Path,
    *,
    train: bool,
    device: torch.device | str = "cuda",
    download: bool = True,
) -> GPUTensorDataset:
    """Load a CIFAR-100 split straight into GPU memory as uint8.

    Identical mechanics to :func:`load_cifar10` -- same 32x32 uint8 layout, same
    resident-on-GPU trick -- with the channel statistics for this split. The only
    substantive difference downstream is that the correct-class margin is computed
    against 99 competing logits rather than 9, which changes its scale. WP8 reports
    Var(f_M) at every alpha so that difference is visible rather than confounding.
    """
    from torchvision.datasets import CIFAR100

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ds = CIFAR100(root=str(root), train=train, download=download)

    images = torch.from_numpy(np.ascontiguousarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(ds.targets, dtype=torch.long)

    mean = torch.tensor(CIFAR100_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, dtype=torch.float32).view(1, 3, 1, 1)

    out = GPUTensorDataset(
        images=images,
        labels=labels,
        mean=mean,
        std=std,
        name=f"cifar100-{'train' if train else 'test'}",
    )
    return out.to(device)


def num_classes_for(dataset: str) -> int:
    """Class count for a dataset name. Raises rather than defaulting to 10."""
    key = dataset.lower().replace("-", "")
    if key not in _NUM_CLASSES:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(_NUM_CLASSES)}")
    return _NUM_CLASSES[key]


def load_dataset(
    dataset: str,
    root: str | Path,
    *,
    train: bool,
    device: torch.device | str = "cuda",
    download: bool = True,
) -> GPUTensorDataset:
    """Dispatch on dataset name.

    Every script routes through here so that adding a dataset does not mean
    editing a hardcoded ``load_cifar10`` and ``num_classes=10`` in seven places --
    which is exactly the kind of default that produced the wrong-checkpoint bug in
    results/BUG_precision_floor_wrong_checkpoint.md.
    """
    key = dataset.lower().replace("-", "")
    if key == "cifar10":
        return load_cifar10(root, train=train, device=device, download=download)
    if key == "cifar100":
        return load_cifar100(root, train=train, device=device, download=download)
    raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(_NUM_CLASSES)}")


# ----------------------------------------------------------------- backdoor
def apply_backdoor_patch(images_uint8: torch.Tensor, cfg: BackdoorConfig) -> torch.Tensor:
    """Stamp the trigger patch onto a copy of ``images_uint8``.

    CLAUDE.md 3: a blue ``patch_size x patch_size`` square in the top-left corner.
    """
    if images_uint8.dtype != torch.uint8:
        raise TypeError(f"expected uint8 images, got {images_uint8.dtype}")
    out = images_uint8.clone()
    r0, c0 = cfg.patch_row, cfg.patch_col
    r1, c1 = r0 + cfg.patch_size, c0 + cfg.patch_size
    h, w = out.shape[-2], out.shape[-1]
    if r1 > h or c1 > w:
        raise ValueError(f"patch {cfg.patch_size}px at ({r0},{c0}) does not fit in {h}x{w}")
    rgb = torch.tensor(
        [int(round(255 * c)) for c in cfg.patch_rgb], dtype=torch.uint8, device=out.device
    ).view(1, 3, 1, 1)
    out[:, :, r0:r1, c0:c1] = rgb
    return out


def make_backdoor_train_split(
    ds: GPUTensorDataset, cfg: BackdoorConfig, seed: int = 0
) -> GPUTensorDataset:
    """Poison ``poison_fraction`` of the trigger class's training images.

    The label is left unchanged: this is the base paper's setup, where the
    trigger becomes a shortcut *for the true class* rather than a label flip.
    Which images were poisoned is recorded in ``.poisoned`` so the backdoor
    editing task (base paper 5.4) can build its target/reference splits.
    """
    if not cfg.enabled:
        return ds
    device = ds.images.device
    target_mask = ds.labels == cfg.target_class
    target_idx = torch.nonzero(target_mask, as_tuple=False).flatten()
    if target_idx.numel() == 0:
        raise ValueError(f"no images of class {cfg.target_class} to poison")

    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(target_idx.numel(), generator=g).to(device)
    n_poison = int(round(cfg.poison_fraction * target_idx.numel()))
    chosen = target_idx[perm[:n_poison]]

    images = ds.images.clone()
    images[chosen] = apply_backdoor_patch(images[chosen], cfg)

    poisoned = torch.zeros(len(ds), dtype=torch.bool, device=device)
    poisoned[chosen] = True

    print(
        f"[backdoor] poisoned {n_poison}/{target_idx.numel()} images of class "
        f"{cfg.target_class} ({CIFAR10_CLASSES[cfg.target_class]}), "
        f"{cfg.patch_size}x{cfg.patch_size} patch rgb={cfg.patch_rgb}"
    )
    return GPUTensorDataset(
        images=images,
        labels=ds.labels.clone(),
        mean=ds.mean,
        std=ds.std,
        name=ds.name + "+backdoor",
        poisoned=poisoned,
    )
