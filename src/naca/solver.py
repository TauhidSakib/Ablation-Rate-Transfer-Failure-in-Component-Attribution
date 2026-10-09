"""Ridge least squares over a *shared* binary design matrix.

The design matrix ``X in {0,1}^{m x N}`` is the same for every evaluation
example -- only the targets differ (CLAUDE.md 2.2). So the normal equations are
formed once and all ``E`` examples are solved simultaneously.

Conditioning -- why this is solved centered
-------------------------------------------
Writing the system as ``(X1'X1 + lambda I*) B = X1'Y`` with ``X1 = [X | 1]`` and
a zero in ``I*``'s intercept slot is the textbook statement, but it is a bad
thing to *compute*. At ``alpha = 0.1`` the mask matrix is ~90% ones, so the
appended intercept column is very nearly inside the span of the mask columns,
and the unpenalized intercept direction is left almost singular. In fp32 that
showed up immediately: identical solves at different chunk sizes disagreed in
the second decimal place.

Ridge with an unpenalized intercept is *exactly* equivalent to centering and
dropping the intercept::

    minimize ||y - Xw - b||^2 + lambda||w||^2
      <=>  b = ybar - xbar'w,  and w solves  (Xc'Xc + lambda I) w = Xc'yc

with ``Xc = X - 1 xbar'`` and ``yc = y - ybar``. Working in that form removes the
near-singular direction analytically rather than asking the solver to survive it.

Centering never materializes ``Xc``:

* ``Xc'Xc = X'X - m xbar xbar'``, and ``X'X`` is a matrix of **integer
  co-occurrence counts**, accumulated exactly in fp32 (integers below 2^24 are
  exact), so the cancellation is done once in fp64 against an exact operand.
* ``Xc'yc = X'yc`` once ``yc`` is centered, because ``mean(yc) = 0``. Centering
  the targets before accumulating also shrinks the accumulated magnitudes, which
  is where the fp32 precision actually goes.

The Cholesky solve itself runs in fp64 (122 MB at ``N = 3,904``, and this GPU
still does it in well under a second). TF32 is disabled throughout: 10 mantissa
bits would round co-occurrence counts in the tens of thousands.

Two paths behind one interface, auto-selected on ``N``:

* ``N <= 10,000`` -- form the centered Gram and Cholesky-solve it.
* ``N > 10,000``  -- **matrix-free** preconditioned block-CG. ResNet-50's Gram
  would be 2.06 GB and OOM at 4 GB, so it is never formed; only
  ``u -> Xc'(Xc u)`` is applied, chunked over rows, at ``O(N)`` memory.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .masks import MaskStore

__all__ = [
    "RidgeSolution",
    "fit_ridge",
    "normal_equations",
    "predict",
]


@contextlib.contextmanager
def _full_fp32():
    """Disable TF32 for the duration -- see module docstring."""
    mm, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = mm
        torch.backends.cudnn.allow_tf32 = cudnn


@dataclass
class RidgeSolution:
    """Per-example linear component model: ``g(v) = b + w'v``."""

    W: np.ndarray  # (N, E) coefficients
    b: np.ndarray  # (E,)   intercepts
    info: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return int(self.W.shape[0])

    @property
    def E(self) -> int:
        return int(self.W.shape[1])

    def predict(self, V: np.ndarray) -> np.ndarray:
        return V.astype(np.float32) @ self.W + self.b[None, :]

    def save(self, path) -> None:
        np.savez_compressed(path, W=self.W, b=self.b)

    @classmethod
    def load(cls, path) -> RidgeSolution:
        with np.load(path, allow_pickle=False) as z:
            return cls(W=z["W"], b=z["b"], info={"loaded_from": str(path)})


def predict(W: np.ndarray, b: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V.astype(np.float32) @ W + b[None, :]


# --------------------------------------------------------------------------
@dataclass
class NormalEquations:
    """Sufficient statistics for the centered system."""

    Gxx: torch.Tensor | None  # (N, N) fp64 centered Gram, or None on the CG path
    rhs: torch.Tensor  # (N, E) fp64  == Xc' yc
    colsum: torch.Tensor  # (N,)  fp64  == sum_t X[t, i]
    ymean: torch.Tensor  # (E,)  fp64
    m: int
    diag: torch.Tensor  # (N,)  fp64 diagonal of Xc'Xc, for Jacobi preconditioning


def normal_equations(
    masks: MaskStore,
    Y: np.ndarray,
    *,
    device: torch.device | str = "cuda",
    chunk: int = 4096,
    form_gram: bool = True,
) -> NormalEquations:
    """Accumulate the centered normal equations in row chunks.

    ``X`` is never materialized in full: at ``N = 22,720``, ``m = 50,000`` that
    would be 4.5 GB in fp32 (CLAUDE.md 5.3).
    """
    m, N = len(masks), masks.N
    if Y.shape[0] != m:
        raise ValueError(f"Y has {Y.shape[0]} rows but the mask store has {m}")
    E = int(Y.shape[1])
    dev = torch.device(device)

    Y = np.ascontiguousarray(Y, dtype=np.float32)
    ymean = torch.from_numpy(Y.mean(axis=0, dtype=np.float64)).to(dev, torch.float64)
    ymean32 = ymean.to(torch.float32)

    Gxx = torch.zeros((N, N), dtype=torch.float32, device=dev) if form_gram else None
    # fp64 accumulator: the per-chunk matmul stays fp32 (fp64 matmul is ~1/64
    # throughput on Ada), but cross-chunk summation is exact, which is what made
    # the result depend on chunk size.
    rhs = torch.zeros((N, E), dtype=torch.float64, device=dev)
    colsum = torch.zeros(N, dtype=torch.float64, device=dev)
    sqsum = torch.zeros(N, dtype=torch.float64, device=dev)

    with _full_fp32():
        for lo in range(0, m, chunk):
            hi = min(lo + chunk, m)
            Xc = torch.from_numpy(masks.unpack(lo, hi)).to(dev, torch.float32)
            Yc = torch.from_numpy(Y[lo:hi]).to(dev, torch.float32) - ymean32.unsqueeze(0)
            if Gxx is not None:
                # binary x binary: integer counts, exact in fp32 below 2^24
                Gxx += Xc.T @ Xc
            rhs += (Xc.T @ Yc).to(torch.float64)
            colsum += Xc.sum(dim=0, dtype=torch.float64)
            sqsum += (Xc * Xc).sum(dim=0, dtype=torch.float64)
            del Xc, Yc

    if Gxx is not None:
        if m >= (1 << 24):
            raise ValueError(
                f"m={m} exceeds 2^24; fp32 accumulation of the Gram is no longer exact"
            )
        Gxx = Gxx.to(torch.float64) - torch.outer(colsum, colsum) / m

    diag = sqsum - colsum * colsum / m
    return NormalEquations(
        Gxx=Gxx,
        rhs=rhs,
        colsum=colsum,
        ymean=ymean,
        m=m,
        diag=diag,
    )


# --------------------------------------------------------------- Gram path
def _cholesky_solve(
    Gc: torch.Tensor, rhs: torch.Tensor, lam: float, *, estimate_cond: bool = False
) -> tuple[torch.Tensor, dict]:
    n = Gc.shape[0]
    A = Gc.clone()
    A.diagonal().add_(lam)  # every coefficient penalised; intercept is not in A

    info: dict = {"method": "cholesky", "lambda": lam, "n": n, "dtype": "float64"}
    if estimate_cond:
        ev = torch.linalg.eigvalsh(A)
        info["eig_min"] = float(ev[0])
        info["eig_max"] = float(ev[-1])
        info["cond"] = float(ev[-1] / max(float(ev[0]), 1e-300))

    jitter, W = 0.0, None
    for attempt in range(6):
        try:
            L = torch.linalg.cholesky(A)
            W = torch.cholesky_solve(rhs, L)
            break
        except RuntimeError as exc:
            jitter = 1e-8 * float(A.diagonal().mean()) if jitter == 0.0 else jitter * 100
            A.diagonal().add_(jitter)
            print(
                f"[solver] cholesky failed (attempt {attempt + 1}): {exc}. "
                f"retrying with jitter {jitter:g}"
            )
    if W is None:
        raise RuntimeError("cholesky failed even with jitter; the Gram is badly conditioned")
    info["jitter"] = jitter

    denom = max(float(torch.linalg.matrix_norm(rhs)), 1e-300)
    info["relative_residual"] = float(torch.linalg.matrix_norm(A @ W - rhs) / denom)
    return W, info


# ----------------------------------------------------------- matrix-free CG
def _centered_matvec_factory(masks: MaskStore, dev, chunk: int, lam: float, colsum, m):
    """``U -> Xc'(Xc U) + lambda U``, with ``Xc = X - 1 xbar'``, never forming ``Xc``."""
    xbar = (colsum / m).to(torch.float32)

    def matvec(U: torch.Tensor) -> torch.Tensor:
        U32 = U.to(torch.float32)
        xbu = xbar @ U32  # (E,)  == xbar' U
        out = torch.zeros(U.shape, dtype=torch.float64, device=dev)
        tsum = torch.zeros(U.shape[1], dtype=torch.float64, device=dev)
        for lo in range(0, len(masks), chunk):
            hi = min(lo + chunk, len(masks))
            Xc = torch.from_numpy(masks.unpack(lo, hi)).to(dev, torch.float32)
            t = Xc @ U32 - xbu.unsqueeze(0)  # (B, E)  == Xc_rows @ U
            out += (Xc.T @ t).to(torch.float64)
            tsum += t.sum(dim=0, dtype=torch.float64)
            del Xc, t
        # Xc' t = X' t - xbar (1' t)
        out -= torch.outer(xbar.to(torch.float64), tsum)
        return out + lam * U

    return matvec


def _block_pcg(
    matvec, rhs: torch.Tensor, diag: torch.Tensor, *, tol: float, max_iter: int
) -> tuple[torch.Tensor, dict]:
    """Jacobi-preconditioned CG, all ``E`` right-hand sides sharing each matvec."""
    X = torch.zeros_like(rhs)
    R = rhs.clone()
    Minv = 1.0 / diag.clamp_min(1e-30)
    Z = Minv.unsqueeze(1) * R
    P = Z.clone()
    rz = (R * Z).sum(dim=0)
    b_norm = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-300)

    n_iter, rel = 0, torch.ones_like(b_norm)
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - used after the loop, in info
        AP = matvec(P)
        alpha = rz / (P * AP).sum(dim=0).clamp_min(1e-300)
        X += alpha.unsqueeze(0) * P
        R -= alpha.unsqueeze(0) * AP
        rel = torch.linalg.vector_norm(R, dim=0) / b_norm
        if float(rel.max()) < tol:
            break
        Z = Minv.unsqueeze(1) * R
        rz_new = (R * Z).sum(dim=0)
        P = Z + (rz_new / rz.clamp_min(1e-300)).unsqueeze(0) * P
        rz = rz_new

    info = {
        "method": "cg",
        "iterations": n_iter,
        "converged": bool(float(rel.max()) < tol),
        "relative_residual_max": float(rel.max()),
        "relative_residual_mean": float(rel.mean()),
        "tol": tol,
    }
    if not info["converged"]:
        print(
            f"[solver] WARNING: block-CG hit max_iter={max_iter} without reaching "
            f"tol={tol:g}; worst relative residual {info['relative_residual_max']:.3e}"
        )
    return X, info


# ------------------------------------------------------------------- driver
def fit_ridge(
    masks: MaskStore,
    Y: np.ndarray,
    lam: float,
    *,
    method: str = "auto",
    device: torch.device | str = "cuda",
    chunk: int = 4096,
    tol: float = 1e-8,
    max_iter: int = 500,
    cholesky_threshold: int = 10_000,
    estimate_cond: bool = False,
) -> RidgeSolution:
    """Fit ``g(v) = b + w'v`` for every example at once, bias unpenalized."""
    if method not in ("auto", "cholesky", "cg"):
        raise ValueError(f"unknown method {method!r}")
    N = masks.N
    if method == "auto":
        method = "cholesky" if cholesky_threshold >= N else "cg"

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    with _full_fp32():
        ne = normal_equations(
            masks, Y, device=dev, chunk=chunk, form_gram=(method == "cholesky")
        )
        if method == "cholesky":
            gram_gb = ne.Gxx.numel() * 8 / 1024**3
            print(
                f"[solver] cholesky path (fp64, centered): Gram {tuple(ne.Gxx.shape)} "
                f"({gram_gb:.2f} GB), lambda={lam:g}"
            )
            Wc, info = _cholesky_solve(ne.Gxx, ne.rhs, lam, estimate_cond=estimate_cond)
            info["gram_gb"] = gram_gb
            ne.Gxx = None
        else:
            print(f"[solver] matrix-free block-CG (centered): N={N}, E={ne.rhs.shape[1]}")
            matvec = _centered_matvec_factory(masks, dev, chunk, lam, ne.colsum, ne.m)
            Wc, info = _block_pcg(
                matvec, ne.rhs, ne.diag + lam, tol=tol, max_iter=max_iter
            )

        # b = ybar - xbar' w  -- recovers the unpenalized intercept exactly
        xbar = ne.colsum / ne.m
        b = ne.ymean - xbar @ Wc

    W_np = Wc.to(torch.float32).cpu().numpy()
    b_np = b.to(torch.float32).cpu().numpy()
    info.update(
        {
            "seconds": time.time() - t0,
            "m": len(masks),
            "N": N,
            "E": int(W_np.shape[1]),
            "alpha_train": masks.spec.alpha,
            "centered": True,
        }
    )
    print(
        f"[solver] {info['method']} solved ({info['E']} examples, N={N}) "
        f"in {info['seconds']:.1f}s"
    )
    del Wc, b, ne
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return RidgeSolution(W=W_np, b=b_np, info=info)
