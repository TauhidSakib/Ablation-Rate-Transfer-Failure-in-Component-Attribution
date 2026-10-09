"""Apply and restore component ablations.

Ablation = zero every weight that carries a component's output onward
(the design spec 2.1). For a conv filter that is its own output channel; for an
attention head or MLP neuron group (WP9) it is the corresponding input columns
of the projection that consumes it. ``LayerSpec.param_axis`` and
``LayerSpec.unit_width`` carry that distinction, and default to the conv case. BatchNorm affine
parameters and running statistics are **left untouched**, and the model stays in
``eval()`` so nothing is recomputed.

Sign convention -- the single most common source of bugs here::

    v_i = 0  ->  component i is ABLATED
    v_i = 1  ->  component i is KEPT

which is the base paper's ``0_{C'}`` convention. :func:`Ablator.apply` asserts
the mask is binary so a float "fraction kept" can never be passed by accident.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from .components import ComponentIndex, LayerSpec

__all__ = ["Ablator", "mask_from_ablated", "ablated_from_mask"]


def mask_from_ablated(ablated: Sequence[int], n: int, device=None) -> torch.Tensor:
    """Indices to ablate -> mask ``v`` with 0 at those positions, 1 elsewhere."""
    v = torch.ones(n, dtype=torch.float32, device=device)
    idx = torch.as_tensor(list(ablated), dtype=torch.long, device=device)
    if idx.numel():
        if int(idx.max()) >= n or int(idx.min()) < 0:
            raise IndexError(f"ablation index out of range [0,{n})")
        v[idx] = 0.0
    return v


def ablated_from_mask(v: torch.Tensor) -> torch.Tensor:
    """Mask ``v`` -> indices of ablated components (where ``v == 0``)."""
    return torch.nonzero(v == 0, as_tuple=False).flatten()


class Ablator:
    """Holds pristine copies of every conv weight and swaps masked ones in.

    The original weights are cloned once onto the model's device, so applying a
    mask is a handful of fused multiply-writes (~1 ms) rather than a reload.
    """

    def __init__(
        self,
        model: nn.Module,
        index: ComponentIndex,
        *,
        verify_names: bool = True,
    ) -> None:
        self.model = model
        self.index = index
        named = dict(model.named_modules())

        self._entries: list[tuple[nn.Module, torch.Tensor, LayerSpec]] = []
        for spec in index.layers:
            module = named.get(spec.name)
            if module is None:
                raise KeyError(
                    f"layer {spec.name!r} from the component index is not present in this "
                    f"model -- index and model are out of sync"
                )
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                raise TypeError(
                    f"layer {spec.name!r} is {type(module).__name__}, expected Conv2d or Linear"
                )
            w = module.weight
            if spec.param_axis >= w.ndim:
                raise ValueError(
                    f"layer {spec.name!r}: param_axis {spec.param_axis} out of range "
                    f"for weight of shape {tuple(w.shape)}"
                )
            if verify_names and w.shape[spec.param_axis] != spec.param_span:
                raise ValueError(
                    f"layer {spec.name!r}: weight axis {spec.param_axis} has "
                    f"{w.shape[spec.param_axis]} positions, index says "
                    f"{spec.n_filters} components x width {spec.unit_width} = "
                    f"{spec.param_span}"
                )
            # clone() preserves memory format (channels_last stays channels_last)
            orig = w.detach().clone()
            self._entries.append((module, orig, spec))

        self._dirty = False
        self.n_applies = 0

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._entries[0][1].device

    @property
    def N(self) -> int:
        return self.index.N

    @property
    def is_dirty(self) -> bool:
        """True if a mask is currently applied."""
        return self._dirty

    # ------------------------------------------------------------------
    def apply(self, v: torch.Tensor | np.ndarray) -> None:
        """Apply mask ``v`` (``v_i = 0`` ablates component ``i``)."""
        v = self._as_mask(v)
        for module, orig, spec in self._entries:
            m = v[spec.start : spec.stop]
            if spec.unit_width != 1:
                # One component owns a run of positions (an attention head owns
                # head_dim columns; an MLP group owns its group width).
                m = m.repeat_interleave(spec.unit_width)
            shape = [1] * orig.ndim
            shape[spec.param_axis] = -1
            # out= writes in place: no allocation per mask, and preserves the
            # weight's existing memory format.
            torch.mul(orig, m.view(shape), out=module.weight.data)
        self._dirty = True
        self.n_applies += 1

    def restore(self) -> None:
        """Put the pristine weights back. Bitwise identical to construction time."""
        if not self._dirty:
            return
        for module, orig, _ in self._entries:
            module.weight.data.copy_(orig)
        self._dirty = False

    @contextmanager
    def ablated(self, v: torch.Tensor | np.ndarray) -> Iterator[nn.Module]:
        """``with ablator.ablated(v) as model:`` -- always restores, even on error."""
        try:
            self.apply(v)
            yield self.model
        finally:
            self.restore()

    # ------------------------------------------------------------------
    def _as_mask(self, v: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(np.ascontiguousarray(v))
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"mask must be a tensor or ndarray, got {type(v)}")
        if v.ndim != 1 or v.numel() != self.N:
            raise ValueError(f"mask must have shape ({self.N},), got {tuple(v.shape)}")
        v = v.to(device=self.device, dtype=self._entries[0][1].dtype, non_blocking=True)
        # Binary check: catches "fraction kept" floats and inverted conventions.
        if not torch.all((v == 0) | (v == 1)):
            bad = v[(v != 0) & (v != 1)][:5].tolist()
            raise ValueError(
                f"mask must be binary in {{0,1}} (v_i=0 ablates); saw values {bad}. "
                f"If you passed an ablation *indicator*, use 1 - m."
            )
        return v

    # ------------------------------------------------------------------
    def original_weight_checksum(self) -> float:
        """Cheap scalar over the pristine weights, for round-trip assertions."""
        return float(sum(float(o.double().abs().sum()) for _, o, _ in self._entries))

    def current_weight_checksum(self) -> float:
        return float(
            sum(float(mod.weight.data.double().abs().sum()) for mod, _, _ in self._entries)
        )

    def weights_match_original(self) -> bool:
        """Bitwise equality of every conv weight against its pristine copy."""
        return all(
            torch.equal(module.weight.data, orig) for module, orig, _ in self._entries
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Ablator(N={self.N}, layers={len(self._entries)}, "
            f"device={self.device}, dirty={self._dirty})"
        )
