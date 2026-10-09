"""Stage 2 -- candidate screening.

Screening by largest ``|w_i|`` picks strong *marginal* effects and structurally
excludes the components we are hunting: near-zero linear coefficient, large
interaction effect. That is the weak-marginal/strong-interaction problem
(the design spec 2.3), so the default scores instead how much each component
modulates the **magnitude** of the additive model's error:

    S[i,z] = | mean_t( X[t,i] * R[t,z]^2 ) - mean_t(X[t,i]) * mean_t(R[t,z]^2) |

which is exactly ``|Cov_t(X[:,i], R[:,z]^2)|``. All three arms of the required
ablation -- ``marginal`` / ``residual`` / ``union`` -- live behind one call.

Pool shape differs by task, and this is a deliberate design decision worth
stating in the paper:

* **attribution** (Phases 3-5): a per-example pool ``K(z)``.
* **editing** (Phases 6-7): one **shared** pool across all examples, ranked by
  ``sum_z S[i,z]``. Greedy selection in 2.5 is only well defined over a common
  index set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .masks import MaskStore

__all__ = ["ScreeningResult", "residual_scores", "select_pool", "shared_pool"]


@dataclass
class ScreeningResult:
    """Selected pools plus the scores that produced them."""

    pool: np.ndarray  # (E, K) per-example, or (K,) if shared
    scores: np.ndarray  # (N, E) residual scores
    mode: str
    K: int
    shared: bool

    @property
    def E(self) -> int:
        return int(self.scores.shape[1])

    def coverage(self) -> dict:
        """How much the per-example pools overlap -- context for the shared-pool choice."""
        if self.shared:
            return {"shared": True, "K": int(self.pool.size)}
        counts = np.bincount(self.pool.ravel(), minlength=self.scores.shape[0])
        chosen = int((counts > 0).sum())
        return {
            "shared": False,
            "K": self.K,
            "distinct_components_used": chosen,
            "fraction_of_all_components": chosen / self.scores.shape[0],
            "max_examples_sharing_one_component": int(counts.max()),
            "mean_selection_count": float(counts[counts > 0].mean()),
        }


def residual_scores(
    masks: MaskStore,
    R: np.ndarray,
    *,
    device: str = "cuda",
    chunk: int = 8192,
) -> np.ndarray:
    """``|Cov_t(X[:,i], R[:,z]^2)|`` as an ``(N, E)`` array, accumulated in chunks."""
    m, N = len(masks), masks.N
    if R.shape[0] != m:
        raise ValueError(f"R has {R.shape[0]} rows, mask store has {m}")
    E = int(R.shape[1])
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    xr2 = torch.zeros((N, E), dtype=torch.float32, device=dev)  # sum_t X[t,i] R[t,z]^2
    xsum = torch.zeros(N, dtype=torch.float32, device=dev)  # sum_t X[t,i]
    r2sum = torch.zeros(E, dtype=torch.float32, device=dev)  # sum_t R[t,z]^2

    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        Xc = torch.from_numpy(masks.unpack(lo, hi)).to(dev, torch.float32)
        Rc = torch.from_numpy(np.ascontiguousarray(R[lo:hi])).to(dev, torch.float32)
        R2c = Rc * Rc
        xr2 += Xc.T @ R2c
        xsum += Xc.sum(dim=0)
        r2sum += R2c.sum(dim=0)
        del Xc, Rc, R2c

    S = (xr2 / m) - torch.outer(xsum / m, r2sum / m)
    return S.abs().cpu().numpy()


def _topk_per_column(scores: np.ndarray, K: int) -> np.ndarray:
    """``(N, E)`` scores -> ``(E, K)`` indices of the top-K per example."""
    N, E = scores.shape
    K = min(K, N)
    part = np.argpartition(-scores, K - 1, axis=0)[:K]  # (K, E), unordered
    taken = np.take_along_axis(scores, part, axis=0)
    order = np.argsort(-taken, axis=0)
    return np.take_along_axis(part, order, axis=0).T.astype(np.int64)  # (E, K)


def select_pool(
    W: np.ndarray,
    scores: np.ndarray,
    K: int,
    mode: str = "residual",
) -> np.ndarray:
    """Per-example candidate pool ``K(z)``, shape ``(E, K)``.

    ``union`` takes top-``K//2`` from each arm and back-fills any duplicates
    from the residual ranking, so every example still gets exactly ``K`` distinct
    components -- the conservative arm must not silently be a smaller pool.
    """
    if mode == "marginal":
        return _topk_per_column(np.abs(W), K)
    if mode == "residual":
        return _topk_per_column(scores, K)
    if mode != "union":
        raise ValueError(f"unknown screening mode {mode!r}")

    N, E = scores.shape
    half = max(1, K // 2)
    marg = _topk_per_column(np.abs(W), half)  # (E, half)
    resid_full = _topk_per_column(scores, min(N, K + half))  # deep enough to back-fill
    out = np.empty((E, K), dtype=np.int64)
    for e in range(E):
        seen: dict[int, None] = {}
        for i in marg[e]:
            seen.setdefault(int(i), None)
        for i in resid_full[e]:
            if len(seen) >= K:
                break
            seen.setdefault(int(i), None)
        picked = list(seen)[:K]
        if len(picked) < K:  # pathological: N < K
            raise ValueError(f"cannot build a pool of {K} distinct components from N={N}")
        out[e] = picked
    return out


def shared_pool(
    W: np.ndarray,
    scores: np.ndarray,
    K: int,
    mode: str = "residual",
) -> np.ndarray:
    """One pool for all examples, shape ``(K,)`` -- required for greedy editing."""
    if mode == "marginal":
        agg = np.abs(W).sum(axis=1)
    elif mode == "residual":
        agg = scores.sum(axis=1)
    elif mode == "union":
        half = max(1, K // 2)
        a = np.argsort(-np.abs(W).sum(axis=1))[:half]
        b = np.argsort(-scores.sum(axis=1))[: K + half]
        seen: dict[int, None] = {}
        for i in np.concatenate([a, b]):
            if len(seen) >= K:
                break
            seen.setdefault(int(i), None)
        return np.asarray(list(seen)[:K], dtype=np.int64)
    else:
        raise ValueError(f"unknown screening mode {mode!r}")
    return np.argsort(-agg)[:K].astype(np.int64)


def screen(
    masks: MaskStore,
    R: np.ndarray,
    W: np.ndarray,
    K: int,
    *,
    mode: str = "residual",
    shared: bool = False,
    device: str = "cuda",
    chunk: int = 8192,
    scores: np.ndarray | None = None,
) -> ScreeningResult:
    """Score, then select. ``scores`` can be passed in to reuse one sweep across modes."""
    if scores is None:
        scores = residual_scores(masks, R, device=device, chunk=chunk)
    pool = (
        shared_pool(W, scores, K, mode)
        if shared
        else select_pool(W, scores, K, mode)
    )
    res = ScreeningResult(pool=pool, scores=scores, mode=mode, K=K, shared=shared)
    cov = res.coverage()
    print(f"[screen] mode={mode} shared={shared} K={K} -> {cov}")
    return res
