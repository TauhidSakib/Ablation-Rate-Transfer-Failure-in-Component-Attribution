"""Enumerate model components -> a flat, stable component index.

A *component* is one convolutional filter: one output channel of one conv layer
(CLAUDE.md 2.1).

Enumeration convention
----------------------
The base paper reports ``N = 22,720`` for ImageNet ResNet-50. Enumerating every
``nn.Conv2d`` output channel in torchvision's ResNet-50 gives 26,560; excluding
the residual *downsample* (shortcut) convolutions gives exactly::

    64 + 1152 + 3072 + 9216 + 9216 = 22,720

That exact match pins the convention: **main-path conv filters only, shortcut
convolutions excluded**. ``include_downsample`` exposes it as a flag so the
choice is auditable rather than buried, and :func:`verify_resnet50_convention`
turns the check into a test.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "LayerSpec",
    "ComponentIndex",
    "build_component_index",
    "build_vit_component_index",
    "verify_resnet50_convention",
    "KNOWN_COUNTS",
]

# Counts quoted by the base paper, for side-by-side logging (CLAUDE.md 8).
KNOWN_COUNTS: dict[str, dict[str, int]] = {
    "resnet18_cifar": {"paper_section4": 2306, "paper_appendixA2": 2344},
    "resnet50": {"paper": 22720},
}


def _depth_group(name: str) -> str:
    """Coarse depth bucket used for the H3 depth-localisation analysis."""
    head = name.split(".")[0]
    if head in ("layer1", "layer2", "layer3", "layer4"):
        return head
    if head in ("conv1", "bn1", "stem"):
        return "stem"
    return head or "other"


@dataclass(frozen=True)
class LayerSpec:
    """One conv layer's slice of the global component index."""

    name: str
    n_filters: int
    start: int
    stop: int  # exclusive
    depth_group: str
    is_downsample: bool
    # Generalisation for non-conv components (ViT). Defaults reproduce the
    # conv-filter behaviour exactly, so every existing index is unchanged.
    #   param_axis  which weight axis the component indexes (0 = output rows,
    #               as for a conv filter; 1 = input columns, which is how an
    #               attention head or an MLP unit is removed -- zero the weights
    #               that carry its output onward).
    #   unit_width  how many consecutive positions along that axis one component
    #               owns (1 for a conv filter, head_dim for an attention head,
    #               the group width for an MLP neuron group).
    param_axis: int = 0
    unit_width: int = 1
    kind: str = "conv"

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def param_span(self) -> int:
        """Positions occupied along ``param_axis`` by this layer's components."""
        return self.n_filters * self.unit_width

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LayerSpec({self.name!r}, n={self.n_filters}, "
            f"[{self.start}:{self.stop}), group={self.depth_group})"
        )


class ComponentIndex:
    """Flat index over conv filters, plus the metadata needed to ablate them.

    Holds *names*, never module references, so an index can be pickled and
    re-attached to a freshly constructed model.
    """

    def __init__(self, layers: list[LayerSpec], arch: str = "") -> None:
        if not layers:
            raise ValueError("no convolutional components found -- wrong model?")
        self.layers = layers
        self.arch = arch
        self._validate()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        cursor = 0
        for spec in self.layers:
            if spec.start != cursor:
                raise ValueError(
                    f"component index is not contiguous at {spec.name}: "
                    f"expected start={cursor}, got {spec.start}"
                )
            if spec.stop - spec.start != spec.n_filters:
                raise ValueError(f"layer {spec.name} slice width != n_filters")
            cursor = spec.stop

    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        return self.layers[-1].stop

    def __len__(self) -> int:
        return self.N

    def __iter__(self) -> Iterator[LayerSpec]:
        return iter(self.layers)

    @property
    def layer_names(self) -> list[str]:
        return [spec.name for spec in self.layers]

    # ------------------------------------------------------------------
    def owner(self, idx: int) -> tuple[LayerSpec, int]:
        """Global component index -> (layer, channel-within-layer)."""
        if not 0 <= idx < self.N:
            raise IndexError(f"component {idx} out of range [0,{self.N})")
        for spec in self.layers:
            if spec.start <= idx < spec.stop:
                return spec, idx - spec.start
        raise AssertionError("unreachable: contiguity validated in __init__")

    def layer_of(self, idx: int) -> str:
        return self.owner(idx)[0].name

    def depth_group_of(self, idx: int) -> str:
        return self.owner(idx)[0].depth_group

    def depth_group_array(self) -> np.ndarray:
        """``(N,)`` array of depth-group labels -- vectorised H3 stratification."""
        out = np.empty(self.N, dtype=object)
        for spec in self.layers:
            out[spec.start : spec.stop] = spec.depth_group
        return out

    def depth_group_indices(self) -> dict[str, np.ndarray]:
        """``{group: component indices}``, in depth order."""
        groups: dict[str, list[int]] = {}
        for spec in self.layers:
            groups.setdefault(spec.depth_group, []).extend(range(spec.start, spec.stop))
        return {g: np.asarray(v, dtype=np.int64) for g, v in groups.items()}

    def summary(self) -> str:
        lines = [f"ComponentIndex(arch={self.arch!r}, N={self.N}, layers={len(self.layers)})"]
        for spec in self.layers:
            lines.append(
                f"  {spec.name:<34s} {spec.n_filters:>5d}  "
                f"[{spec.start:>6d}:{spec.stop:>6d})  {spec.depth_group}"
            )
        by_group: dict[str, int] = {}
        for spec in self.layers:
            by_group[spec.depth_group] = by_group.get(spec.depth_group, 0) + spec.n_filters
        lines.append("  --- by depth group ---")
        for g, n in by_group.items():
            lines.append(f"  {g:<34s} {n:>5d}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "arch": self.arch,
            "N": self.N,
            "layers": [
                {
                    "name": s.name,
                    "n_filters": s.n_filters,
                    "start": s.start,
                    "stop": s.stop,
                    "depth_group": s.depth_group,
                    "is_downsample": s.is_downsample,
                    "param_axis": s.param_axis,
                    "unit_width": s.unit_width,
                    "kind": s.kind,
                }
                for s in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ComponentIndex:
        layers = [
            LayerSpec(
                name=x["name"],
                n_filters=x["n_filters"],
                start=x["start"],
                stop=x["stop"],
                depth_group=x["depth_group"],
                is_downsample=x["is_downsample"],
                param_axis=x.get("param_axis", 0),
                unit_width=x.get("unit_width", 1),
                kind=x.get("kind", "conv"),
            )
            for x in d["layers"]
        ]
        return cls(layers, arch=d.get("arch", ""))


# --------------------------------------------------------------------------


def _is_downsample(name: str) -> bool:
    """torchvision names shortcut convs ``layerX.Y.downsample.0``."""
    return "downsample" in name.split(".") or "shortcut" in name.split(".")


def build_vit_component_index(
    model: nn.Module,
    *,
    arch: str = "vit_tiny_cifar",
    mlp_group_width: int = 16,
    expected_n: int | None = None,
    verbose: bool = True,
) -> ComponentIndex:
    """Enumerate ViT components: attention heads + MLP neuron groups.

    Ordering is **per block**: block i contributes its heads, then its MLP
    groups, before block i+1 begins. That keeps every transformer block a
    contiguous run of the component index, which is what the coarse granularity
    (12 blocks, 2^12 masks) requires.

    For ViT-Tiny this gives 12 x (3 heads + 768/16 groups) = 12 x 51 = 612.
    """
    depth = int(getattr(model, "depth"))
    num_heads = int(getattr(model, "num_heads"))
    head_dim = int(getattr(model, "head_dim"))
    mlp_hidden = int(getattr(model, "mlp_hidden"))
    if mlp_hidden % mlp_group_width:
        raise ValueError(
            f"mlp_hidden {mlp_hidden} not divisible by group width {mlp_group_width}"
        )
    n_groups = mlp_hidden // mlp_group_width

    layers: list[LayerSpec] = []
    cursor = 0
    for i in range(depth):
        group = f"block{i}"
        layers.append(
            LayerSpec(
                name=f"blocks.{i}.attn.proj",
                n_filters=num_heads,
                start=cursor,
                stop=cursor + num_heads,
                depth_group=group,
                is_downsample=False,
                param_axis=1,        # heads occupy proj's INPUT columns
                unit_width=head_dim,
                kind="attn_head",
            )
        )
        cursor += num_heads
        layers.append(
            LayerSpec(
                name=f"blocks.{i}.mlp.fc2",
                n_filters=n_groups,
                start=cursor,
                stop=cursor + n_groups,
                depth_group=group,
                is_downsample=False,
                param_axis=1,        # groups occupy fc2's INPUT columns
                unit_width=mlp_group_width,
                kind="mlp_group",
            )
        )
        cursor += n_groups

    index = ComponentIndex(layers, arch=arch)
    if verbose:
        print(
            f"[components] arch={arch!r} N={index.N} = {depth} blocks x "
            f"({num_heads} heads + {n_groups} mlp groups of width {mlp_group_width})"
        )
    if expected_n is not None and expected_n != index.N:
        raise ValueError(
            f"component count mismatch: enumerated {index.N}, config expects {expected_n}"
        )
    return index


def build_component_index(
    model: nn.Module,
    *,
    arch: str = "",
    include_downsample: bool = False,
    expected_n: int | None = None,
    verbose: bool = True,
) -> ComponentIndex:
    """Walk ``model`` in definition order and index every conv filter.

    Parameters
    ----------
    include_downsample
        Include residual shortcut convolutions. ``False`` reproduces the base
        paper's ``N = 22,720`` for ResNet-50 (see module docstring).
    expected_n
        If given, assert the enumerated count matches. Mismatches raise -- a
        silently wrong ``N`` corrupts every downstream mask.
    """
    if arch.startswith("vit"):
        return build_vit_component_index(
            model, arch=arch, expected_n=expected_n, verbose=verbose
        )

    layers: list[LayerSpec] = []
    cursor = 0
    skipped: list[tuple[str, int]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        n_out = int(module.out_channels)
        down = _is_downsample(name)
        if down and not include_downsample:
            skipped.append((name, n_out))
            continue
        layers.append(
            LayerSpec(
                name=name,
                n_filters=n_out,
                start=cursor,
                stop=cursor + n_out,
                depth_group=_depth_group(name),
                is_downsample=down,
            )
        )
        cursor += n_out

    index = ComponentIndex(layers, arch=arch)

    if verbose:
        print(f"[components] arch={arch!r} enumerated N={index.N} over {len(layers)} conv layers")
        if skipped:
            n_skip = sum(n for _, n in skipped)
            print(
                f"[components] excluded {len(skipped)} downsample convs "
                f"({n_skip} filters); include_downsample=False"
            )
        known = KNOWN_COUNTS.get(arch, {})
        for label, val in known.items():
            verdict = "MATCH" if val == index.N else f"differs by {index.N - val:+d}"
            print(f"[components]   vs {label}: {val} -> {verdict}")

    if expected_n is not None and expected_n != index.N:
        raise ValueError(
            f"component count mismatch for arch={arch!r}: enumerated {index.N}, "
            f"config expected {expected_n}. Refusing to continue -- every mask "
            f"depends on this number."
        )
    return index


# --------------------------------------------------------------------------


def verify_resnet50_convention() -> tuple[int, int]:
    """Return ``(without_downsample, with_downsample)`` filter counts for ResNet-50.

    The first must be 22,720 -- the base paper's number. Used by test 3.
    """
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    without = build_component_index(
        model, arch="resnet50", include_downsample=False, verbose=False
    ).N
    with_ds = build_component_index(
        model, arch="resnet50", include_downsample=True, verbose=False
    ).N
    return without, with_ds


if __name__ == "__main__":  # pragma: no cover - manual inspection
    torch.manual_seed(0)
    without, with_ds = verify_resnet50_convention()
    print(f"ResNet-50: main-path only = {without} (paper: 22720), with downsample = {with_ds}")
