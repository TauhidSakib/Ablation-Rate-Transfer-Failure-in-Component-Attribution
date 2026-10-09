"""ViT-Tiny for 32x32 inputs, written out rather than imported.

WP9 changes the *component type*: attention heads and MLP neuron groups instead
of conv filters. That makes module naming part of the experiment, not an
implementation detail -- the component index addresses weights by name, so the
names have to be stable and known. timm is not installed and pinning it would
add a dependency whose internal naming we do not control, so the ~90 lines are
written here.

Architecture (ViT-Tiny, adapted to CIFAR):
    patch 4 over 32x32   -> 64 tokens (+1 cls)
    dim 192, depth 12, heads 3 (head_dim 64), mlp_ratio 4 (hidden 768)

Ablation targets, and why each is the natural analogue of zeroing a conv filter:

    attention head h in block i -> ``blocks.{i}.attn.proj.weight[:, h*64:(h+1)*64]``
        The head's output occupies that slice of proj's *input*. Zeroing those
        columns removes exactly that head's contribution to the residual stream
        and nothing else.
    MLP neuron group g in block i -> ``blocks.{i}.mlp.fc2.weight[:, g*w:(g+1)*w]``
        Same logic one layer over: those columns carry the group's activations
        onward. Biases are untouched, mirroring the conv convention where
        BatchNorm affine parameters are left alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["VisionTransformer", "vit_tiny_cifar"]


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, T, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim, self.depth, self.num_heads = dim, depth, num_heads
        self.head_dim = dim // num_heads
        self.mlp_hidden = int(dim * mlp_ratio)
        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList(
            [Block(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # patch_embed is a Conv2d; a channels_last input would silently change
        # nothing here, but keep it contiguous so the reshape is well defined.
        x = self.patch_embed(x.contiguous()).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


def vit_tiny_cifar(num_classes: int = 10) -> nn.Module:
    return VisionTransformer(num_classes=num_classes)
