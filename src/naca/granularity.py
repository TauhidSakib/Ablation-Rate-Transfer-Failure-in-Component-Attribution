"""Component granularity -- one ablation pipeline, several unit definitions.

A *unit* is a set of conv filters ablated together. Changing granularity changes
only the map ``component -> unit``; the ablation machinery, the collection loop
and the estimators are untouched. Writing five ablation implementations would be
five chances to introduce an inconsistency that looks like a granularity effect.

Levels (ResNet-18 / CIFAR-10, N = 3,904 filters):

    filter        every filter its own unit                    3904
    filter_group  ``group_size`` adjacent filters in a layer    ~488
    layer         one unit per conv layer                         17
    block         one unit per residual block, stem separate        9

Coarser is not always fewer: blocks (9) are coarser than layers (17) because a
block contains two conv layers.

The severity confound
---------------------
At 9 units, an alpha of 0.10 means ablating 0.9 units, and one whole block is
catastrophically more damaging than 10% of filters. Fraction-matching across
granularities is therefore meaningless, and a naive comparison would confound
granularity with ablation severity -- any apparent trend would be an artifact.
:func:`severity_of` measures the actual damage so a severity-matched protocol can
be calibrated instead. Both protocols are reported; severity-matched is primary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .components import ComponentIndex

__all__ = [
    "LEVELS",
    "UnitMap",
    "build_unit_map",
    "UnitMaskView",
    "severity_of",
]

LEVELS = ("filter", "filter_group", "layer", "block")


def _block_of(layer_name: str) -> str:
    """Residual/transformer block a layer belongs to.

    ``layer2.1.conv1`` -> ``layer2.1``      (ResNet)
    ``blocks.3.attn.proj`` -> ``blocks.3``  (ViT -- both the attention and MLP
                                             layers of block 3 map to one unit,
                                             giving 12 units for ViT-Tiny)
    ``conv1`` -> ``stem``
    """
    parts = layer_name.split(".")
    if parts[0] == "blocks" and len(parts) >= 2:
        return f"blocks.{parts[1]}"
    if parts[0].startswith("layer") and len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return "stem"


@dataclass
class UnitMap:
    """``component -> unit`` assignment plus human-readable unit names."""

    unit_of: np.ndarray  # (N,) int64
    names: list[str]
    level: str
    N: int

    @property
    def n_units(self) -> int:
        return len(self.names)

    def members(self, u: int) -> np.ndarray:
        return np.nonzero(self.unit_of == u)[0]

    def sizes(self) -> np.ndarray:
        return np.bincount(self.unit_of, minlength=self.n_units)

    def expand(self, unit_masks: np.ndarray) -> np.ndarray:
        """``(rows, n_units)`` unit mask -> ``(rows, N)`` filter mask."""
        return unit_masks[:, self.unit_of]

    def summary(self) -> str:
        s = self.sizes()
        return (
            f"UnitMap(level={self.level!r}, n_units={self.n_units}, N={self.N}, "
            f"unit size min/median/max = {s.min()}/{int(np.median(s))}/{s.max()})"
        )


def build_unit_map(index: ComponentIndex, level: str, *, group_size: int = 8) -> UnitMap:
    """Partition the components of ``index`` into units at the given level."""
    if level not in LEVELS:
        raise ValueError(f"unknown granularity {level!r}; known: {LEVELS}")
    N = index.N
    unit_of = np.empty(N, dtype=np.int64)
    names: list[str] = []

    if level == "filter":
        unit_of = np.arange(N, dtype=np.int64)
        names = [f"c{i}" for i in range(N)]

    elif level == "filter_group":
        # groups never straddle a layer boundary: a unit must be a coherent set
        nxt = 0
        for spec in index.layers:
            n = spec.n_filters
            n_groups = max(1, (n + group_size - 1) // group_size)
            for g in range(n_groups):
                lo = spec.start + g * group_size
                hi = min(spec.start + (g + 1) * group_size, spec.stop)
                unit_of[lo:hi] = nxt
                names.append(f"{spec.name}[{g}]")
                nxt += 1

    elif level == "layer":
        for u, spec in enumerate(index.layers):
            unit_of[spec.start : spec.stop] = u
            names.append(spec.name)

    else:  # block
        seen: dict[str, int] = {}
        for spec in index.layers:
            b = _block_of(spec.name)
            if b not in seen:
                seen[b] = len(names)
                names.append(b)
            unit_of[spec.start : spec.stop] = seen[b]

    return UnitMap(unit_of=unit_of, names=names, level=level, N=N)


class UnitMaskView:
    """Presents a unit-level mask store as filter-level masks for ablation.

    The regression is fitted at **unit** level (``n_units`` coefficients); the
    ablation must happen at **filter** level. This adapter exposes the same
    ``unpack`` / ``__len__`` / ``N`` surface the collection loop expects, so
    `collect_counterfactuals` needs no granularity-specific branch.
    """

    def __init__(self, store, umap: UnitMap):
        if umap.n_units != store.N:
            raise ValueError(
                f"mask store has N={store.N} but the unit map has "
                f"{umap.n_units} units"
            )
        self.store = store
        self.umap = umap
        self.spec = store.spec

    def __len__(self) -> int:
        return len(self.store)

    @property
    def N(self) -> int:
        return self.umap.N

    def unpack(self, lo: int = 0, hi: int | None = None) -> np.ndarray:
        return self.umap.expand(self.store.unpack(lo, hi))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"UnitMaskView(level={self.umap.level!r}, units={self.umap.n_units}, "
            f"filters={self.umap.N}, m={len(self)})"
        )


@torch.inference_mode()
def severity_of(
    model,
    index: ComponentIndex,
    umap: UnitMap,
    k_units: int,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    clean: np.ndarray,
    *,
    n_probe: int = 64,
    sub_batch: int = 1000,
    seed: int = 0,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Mean |change in margin| when ``k_units`` units are ablated.

    This is the quantity that must be matched across granularities. Fraction of
    components ablated is not comparable; observed damage is.
    """
    from .ablation import Ablator
    from .collect import _forward_margins

    rng = np.random.default_rng(seed)
    ablator = Ablator(model, index)
    buf = torch.empty(int(x_eval.shape[0]), dtype=torch.float32, device=x_eval.device)
    deltas = []
    frac_components = []
    try:
        for _ in range(n_probe):
            units = rng.choice(umap.n_units, size=k_units, replace=False)
            um = np.ones(umap.n_units, dtype=np.uint8)
            um[units] = 0
            fm = um[umap.unit_of].astype(np.float32)
            frac_components.append(1.0 - fm.mean())
            ablator.apply(torch.from_numpy(fm).to(x_eval.device))
            _forward_margins(model, x_eval, y_eval, buf, sub_batch, amp_dtype)
            deltas.append(buf.cpu().numpy() - clean)
    finally:
        ablator.restore()
    D = np.stack(deltas)
    return {
        "level": umap.level,
        "n_units": umap.n_units,
        "k_units": int(k_units),
        "mean_abs_delta": float(np.abs(D).mean()),
        "mean_signed_delta": float(D.mean()),
        "mean_component_fraction": float(np.mean(frac_components)),
        "std_abs_delta": float(np.abs(D).mean(axis=1).std()),
    }
