"""Phase 3 -- collect component counterfactuals ``y_t = f_M(z, C_t)``.

This is the only expensive stage in the project; everything downstream re-uses
what it produces (that re-use is contribution 2: **zero additional forward
passes**). Two decisions carry the performance:

**Loop order: masks outer, examples inner** (the design spec 5.1). Weight-writing is
per-mask and forward cost is per-example; reversing the loops would pay the
weight write ``E`` times per mask.

**No restore between masks.** ``Ablator.apply`` writes ``orig * mask`` into the
weights and never reads their current value, so consecutive masks can be applied
back to back. Restoring in between would double the weight traffic for nothing.
The weights are restored once at the end, and on any exception.

Output ``Y`` is an ``(m, E)`` float32 ``.npy`` **memmap**: it is simultaneously
the result and the checkpoint, so an interrupted run resumes from the last
flushed mask instead of restarting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .ablation import Ablator
from .components import ComponentIndex
from .masks import MaskStore
from .metrics import correct_class_margin

__all__ = ["CollectionState", "collect_counterfactuals", "clean_margins", "preload_eval"]

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class CollectionState:
    """Resume record written next to the ``Y`` memmap."""

    path: Path
    m: int
    n_examples: int
    completed: int
    mask_spec: dict
    seconds: float = 0.0

    def save(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "m": self.m,
                    "n_examples": self.n_examples,
                    "completed": self.completed,
                    "mask_spec": self.mask_spec,
                    "seconds": self.seconds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CollectionState | None:
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            m=d["m"],
            n_examples=d["n_examples"],
            completed=d["completed"],
            mask_spec=d["mask_spec"],
            seconds=d.get("seconds", 0.0),
        )


def preload_eval(
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the eval set once, keep it resident in ``channels_last`` fp32.

    This is what replaces FFCV: after this call no DataLoader is touched, so the
    collection loop is purely compute-bound.
    """
    x = images_uint8.float().div_(255.0)
    x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
    return x, labels.long()


@torch.inference_mode()
def _forward_margins(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor,
    sub_batch: int,
    amp_dtype: torch.dtype,
) -> None:
    """Fill ``out`` (E,) with correct-class margins, in VRAM-sized sub-batches."""
    n = x.shape[0]
    for lo in range(0, n, sub_batch):
        hi = min(lo + sub_batch, n)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            logits = model(x[lo:hi])
        out[lo:hi] = correct_class_margin(logits.float(), y[lo:hi])


@torch.inference_mode()
def clean_margins(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sub_batch: int = 500,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> np.ndarray:
    """``f_M(z, empty)`` -- the unablated baseline margin for every eval example."""
    model.eval()
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    _forward_margins(model, x, y, out, sub_batch, amp_dtype)
    return out.detach().cpu().numpy()


def collect_counterfactuals(
    model: nn.Module,
    index: ComponentIndex,
    masks: MaskStore,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    out_path: str | Path,
    *,
    sub_batch: int = 500,
    dtype_forward: str = "bf16",
    checkpoint_every: int = 1_000,
    mask_chunk: int = 1_000,
    resume: bool = True,
    log_every: int = 2_000,
) -> tuple[np.ndarray, CollectionState]:
    """Evaluate every mask in ``masks`` against every example in ``x_eval``.

    Returns ``(Y, state)`` with ``Y`` of shape ``(m, E)``, ``Y[t, e] = f_M(z_e, C_t)``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(".state.json")

    m, n_eval = len(masks), int(x_eval.shape[0])
    if masks.N != index.N:
        raise ValueError(f"mask store has N={masks.N} but component index has N={index.N}")

    amp_dtype = _DTYPES[dtype_forward]
    device = x_eval.device

    # --- open or create the memmap that is both output and checkpoint -----
    state = CollectionState.load(state_path) if resume else None
    if state is not None and out_path.exists():
        if (state.m, state.n_examples) != (m, n_eval):
            raise ValueError(
                f"checkpoint at {out_path} is for shape ({state.m},{state.n_examples}), "
                f"this run needs ({m},{n_eval}). Move it aside or pass resume=False."
            )
        Y = np.lib.format.open_memmap(out_path, mode="r+")
        start = state.completed
        print(f"[collect] resuming at mask {start}/{m} ({start / m:.1%} done)")
    else:
        Y = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(m, n_eval)
        )
        state = CollectionState(
            path=state_path,
            m=m,
            n_examples=n_eval,
            completed=0,
            mask_spec=masks.spec.to_dict(),
        )
        start = 0

    if start >= m:
        print(f"[collect] already complete ({m} masks)")
        return np.asarray(Y), state

    model.eval()
    ablator = Ablator(model, index)
    margin_buf = torch.empty(n_eval, dtype=torch.float32, device=device)

    t_start = time.time()
    done_this_run = 0
    # `seconds` must accumulate across resumes. Storing only this run's elapsed
    # time made `completed / seconds` in the state file a meaningless ratio --
    # a total numerator over a partial denominator, which overstated the rate.
    elapsed_offset = state.seconds
    try:
        for lo in range(start, m, mask_chunk):
            hi = min(lo + mask_chunk, m)
            # Unpack a chunk of masks on the CPU, then move once to the GPU.
            chunk = torch.from_numpy(masks.unpack(lo, hi)).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            for j in range(hi - lo):
                # No restore(): apply() rebuilds each weight from the pristine copy.
                ablator.apply(chunk[j])
                _forward_margins(model, x_eval, y_eval, margin_buf, sub_batch, amp_dtype)
                Y[lo + j] = margin_buf.cpu().numpy()

            done_this_run += hi - lo
            state.completed = hi
            state.seconds = elapsed_offset + (time.time() - t_start)

            if hi % checkpoint_every == 0 or hi == m:
                Y.flush()
                state.save()

            if done_this_run and (hi % log_every == 0 or hi == m):
                elapsed = time.time() - t_start
                rate = done_this_run / max(elapsed, 1e-9)
                eta = (m - hi) / max(rate, 1e-9)
                print(
                    f"[collect] {hi:>6d}/{m}  {hi / m:5.1%}  "
                    f"{rate:6.1f} masks/s  elapsed {elapsed / 60:5.1f}m  "
                    f"ETA {eta / 60:5.1f}m"
                )
    finally:
        ablator.restore()
        Y.flush()
        state.save()

    # --- integrity: a NaN margin means a degenerate ablation (the design spec 5.4)
    arr = np.asarray(Y)
    bad = ~np.isfinite(arr)
    if bad.any():
        rows = np.unique(np.nonzero(bad)[0])[:10]
        raise FloatingPointError(
            f"{int(bad.sum())} non-finite counterfactuals in {out_path.name} "
            f"(first offending masks: {rows.tolist()}). This must be surfaced, "
            f"not dropped -- inspect those ablations."
        )

    total = time.time() - t_start
    print(
        f"[collect] done: {m} masks x {n_eval} examples in {total / 60:.1f} min "
        f"({done_this_run / max(total, 1e-9):.1f} masks/s), "
        f"{arr.nbytes / 1024**2:.0f} MB -> {out_path.name}"
    )
    return arr, state


def throughput_probe(
    model: nn.Module,
    index: ComponentIndex,
    masks: MaskStore,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    n_probe: int = 100,
    sub_batch: int = 500,
    dtype_forward: str = "bf16",
) -> dict:
    """Time ``n_probe`` masks and project the full run, before committing to it."""
    amp_dtype = _DTYPES[dtype_forward]
    model.eval()
    ablator = Ablator(model, index)
    buf = torch.empty(int(x_eval.shape[0]), dtype=torch.float32, device=x_eval.device)
    chunk = torch.from_numpy(masks.unpack(0, min(n_probe, len(masks)))).to(
        device=x_eval.device, dtype=torch.float32
    )

    # warm up cuDNN autotuning before timing anything
    for j in range(min(5, chunk.shape[0])):
        ablator.apply(chunk[j])
        _forward_margins(model, x_eval, y_eval, buf, sub_batch, amp_dtype)
    torch.cuda.synchronize()

    t0 = time.time()
    for j in range(chunk.shape[0]):
        ablator.apply(chunk[j])
        _forward_margins(model, x_eval, y_eval, buf, sub_batch, amp_dtype)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    ablator.restore()

    rate = chunk.shape[0] / elapsed
    return {
        "n_probe": int(chunk.shape[0]),
        "seconds": elapsed,
        "masks_per_sec": rate,
        "ms_per_mask": 1000.0 / rate,
        "projected_minutes_for_m": len(masks) / rate / 60.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
