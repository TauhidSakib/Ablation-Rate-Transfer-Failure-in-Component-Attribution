"""Model construction and training.

Setup A is the CIFAR variant of ResNet-18: a 3x3 stride-1 stem with the maxpool
removed, which is standard for 32x32 inputs. Training is fully GPU-resident --
augmentation runs on the GPU on uint8 tensors, so no DataLoader is touched and
hard rule 1 (``num_workers=0``) costs nothing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig
from .datasets import GPUTensorDataset
from .metrics import correct_class_margin

__all__ = [
    "build_model",
    "resnet18_cifar",
    "vit_tiny_cifar",
    "train_classifier",
    "evaluate",
    "compute_margins",
    "save_checkpoint",
    "load_checkpoint",
    "TrainResult",
]


# ----------------------------------------------------------------- models
def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-18 adapted to 32x32: 3x3 stem, no maxpool."""
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def resnet50_imagenet(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet-50 with a fresh head, for the Waterbirds / CelebA fine-tunes."""
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def vit_tiny_cifar(num_classes: int = 10) -> nn.Module:
    """ViT-Tiny adapted to 32x32 (WP9). See naca.vit for the ablation targets."""
    from .vit import vit_tiny_cifar as _build

    return _build(num_classes=num_classes)


_ARCHS = {
    "resnet18_cifar": resnet18_cifar,
    "resnet50": resnet50_imagenet,
    "vit_tiny_cifar": vit_tiny_cifar,
}


def build_model(
    arch: str,
    num_classes: int = 10,
    *,
    device: torch.device | str = "cuda",
    channels_last: bool = True,
    **kwargs,
) -> nn.Module:
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {sorted(_ARCHS)}")
    model = _ARCHS[arch](num_classes=num_classes, **kwargs)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)  # hard rule 5
    return model


# ------------------------------------------------------- GPU augmentation
def _random_crop_flip(
    x_uint8: torch.Tensor, padding: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Random crop with reflection padding + random horizontal flip, on GPU.

    Fully vectorised: one gather for the crop, one masked flip. Per-sample crop
    offsets are drawn independently, matching torchvision's ``RandomCrop``.
    """
    b, c, h, w = x_uint8.shape
    dev = x_uint8.device
    padded = F.pad(x_uint8.float(), (padding,) * 4, mode="reflect")

    span = 2 * padding + 1
    i = torch.randint(0, span, (b,), device=dev, generator=generator)
    j = torch.randint(0, span, (b,), device=dev, generator=generator)

    ar = torch.arange(h, device=dev)
    ac = torch.arange(w, device=dev)
    rows = (i[:, None] + ar[None, :]).view(b, 1, h, 1).expand(b, c, h, w)
    cols = (j[:, None] + ac[None, :]).view(b, 1, 1, w).expand(b, c, h, w)
    bidx = torch.arange(b, device=dev).view(b, 1, 1, 1).expand(b, c, h, w)
    cidx = torch.arange(c, device=dev).view(1, c, 1, 1).expand(b, c, h, w)

    out = padded[bidx, cidx, rows, cols]

    flip = torch.rand(b, device=dev, generator=generator) < 0.5
    out = torch.where(flip.view(b, 1, 1, 1), out.flip(-1), out)
    return out


def _triangular_lr(epoch: int, cfg: TrainConfig) -> float:
    """Linear warmup to ``lr`` at ``lr_peak_epoch``, linear decay to 0 after."""
    if epoch < cfg.lr_peak_epoch:
        return cfg.lr * (epoch + 1) / max(1, cfg.lr_peak_epoch)
    remaining = cfg.epochs - cfg.lr_peak_epoch
    if remaining <= 0:
        return cfg.lr
    return cfg.lr * max(0.0, 1.0 - (epoch - cfg.lr_peak_epoch) / remaining)


def _cosine_lr(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(1, cfg.warmup_epochs)
    t = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * t))


def lr_at(epoch: int, cfg: TrainConfig) -> float:
    if cfg.lr_schedule == "triangular":
        return _triangular_lr(epoch, cfg)
    if cfg.lr_schedule == "cosine":
        return _cosine_lr(epoch, cfg)
    if cfg.lr_schedule == "step":
        return cfg.lr * (0.1 ** (epoch // max(1, cfg.epochs // 3)))
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")


# -------------------------------------------------------------- training
@dataclass
class TrainResult:
    test_acc: float
    train_acc: float
    epochs: int
    seconds: float
    history: list[dict]

    def to_dict(self) -> dict:
        return {
            "test_acc": self.test_acc,
            "train_acc": self.train_acc,
            "epochs": self.epochs,
            "seconds": self.seconds,
            "history": self.history,
        }


def train_classifier(
    model: nn.Module,
    train_ds: GPUTensorDataset,
    test_ds: GPUTensorDataset,
    cfg: TrainConfig,
    *,
    seed: int = 0,
    amp_dtype: torch.dtype = torch.bfloat16,
    crop_padding: int = 4,
    log_every: int = 1,
) -> TrainResult:
    """SGD or AdamW, scheduled LR, GPU-resident augmentation, bf16 autocast."""
    device = train_ds.images.device
    n = len(train_ds)
    steps = max(1, n // cfg.batch_size)
    gen = torch.Generator(device=device).manual_seed(seed)

    if getattr(cfg, "optimizer", "sgd") == "adamw":
        # No weight decay on norms/biases/tokens -- standard for transformers and
        # worth several points of accuracy at this scale.
        decay, no_decay = [], []
        for name, prm in model.named_parameters():
            if not prm.requires_grad:
                continue
            (no_decay if prm.ndim <= 1 or name.endswith(("cls_token", "pos_embed"))
             else decay).append(prm)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr,
            betas=(0.9, 0.999),
        )
    else:
        opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    history: list[dict] = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        lr = lr_at(epoch, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        perm = torch.randperm(n, device=device, generator=gen)
        running, correct, seen = 0.0, 0, 0
        for step in range(steps):
            idx = perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            raw = train_ds.images[idx]
            aug = _random_crop_flip(raw, crop_padding, generator=gen)
            x = ((aug / 255.0) - train_ds.mean) / train_ds.std
            x = x.contiguous(memory_format=torch.channels_last)
            y = train_ds.labels[idx]

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
                loss = loss_fn(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach()) * y.numel()
            correct += int((out.detach().argmax(1) == y).sum())
            seen += y.numel()

        train_acc = correct / max(1, seen)
        test_acc = evaluate(model, test_ds, amp_dtype=amp_dtype)
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": running / max(1, seen),
                "train_acc": train_acc,
                "test_acc": test_acc,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        if log_every and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            print(
                f"[train] epoch {epoch + 1:>3d}/{cfg.epochs}  lr={lr:.4f}  "
                f"loss={running / max(1, seen):.4f}  train_acc={train_acc:.4f}  "
                f"test_acc={test_acc:.4f}  ({time.time() - t0:.0f}s)"
            )

    return TrainResult(
        test_acc=history[-1]["test_acc"],
        train_acc=history[-1]["train_acc"],
        epochs=cfg.epochs,
        seconds=time.time() - t0,
        history=history,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    ds: GPUTensorDataset,
    *,
    batch_size: int = 1000,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> float:
    """Top-1 accuracy."""
    model.eval()
    correct = 0
    for lo in range(0, len(ds), batch_size):
        hi = min(lo + batch_size, len(ds))
        x = ds.normalized(lo, hi)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(x)
        correct += int((out.float().argmax(1) == ds.labels[lo:hi]).sum())
    return correct / len(ds)


@torch.inference_mode()
def compute_margins(
    model: nn.Module,
    ds: GPUTensorDataset,
    *,
    batch_size: int = 500,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Correct-class margins for a whole split, fp32."""
    model.eval()
    out = torch.empty(len(ds), dtype=torch.float32, device=ds.images.device)
    for lo in range(0, len(ds), batch_size):
        hi = min(lo + batch_size, len(ds))
        x = ds.normalized(lo, hi)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)
        out[lo:hi] = correct_class_margin(logits.float(), ds.labels[lo:hi])
    return out


# ----------------------------------------------------------- checkpoints
def save_checkpoint(
    model: nn.Module, path: str | Path, *, meta: dict | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    return path


def load_checkpoint(
    model: nn.Module, path: str | Path, *, strict: bool = True
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"], strict=strict)
    return blob.get("meta", {})
