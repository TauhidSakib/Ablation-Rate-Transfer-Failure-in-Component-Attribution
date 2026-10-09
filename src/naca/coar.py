"""Stage 1 -- linear COAR: the baseline, and the screening signal for Stage 2.

This is the base paper reproduced unmodified (CLAUDE.md 2.2). Two things it
produces are load-bearing downstream:

* ``W`` -- the additive attribution, and the ``|w_i|`` marginal-screening
  ablation arm.
* ``R = Y - X1 B`` -- the Stage-1 residuals, which is what residual screening
  actually scores (2.3). Residuals are computed chunked and never held at full
  ``(m, E)`` in GPU memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .masks import MaskStore
from .metrics import mse, pearson_r, r2_offset_corrected, r2_score, variance
from .solver import RidgeSolution, fit_ridge

__all__ = [
    "CoarModel",
    "fit_coar",
    "predict_chunked",
    "residuals_chunked",
    "evaluate_predictions",
    "evaluate_across_alpha",
]


@dataclass
class CoarModel:
    """A fitted per-example additive component model."""

    W: np.ndarray  # (N, E)
    b: np.ndarray  # (E,)
    alpha_train: float
    info: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return int(self.W.shape[0])

    @property
    def E(self) -> int:
        return int(self.W.shape[1])

    def predict(self, masks: MaskStore, *, device="cuda", chunk: int = 8192) -> np.ndarray:
        return predict_chunked(self.W, self.b, masks, device=device, chunk=chunk)

    def residuals(self, masks: MaskStore, Y: np.ndarray, **kw) -> np.ndarray:
        return residuals_chunked(self.W, self.b, masks, Y, **kw)

    def save(self, path) -> None:
        np.savez_compressed(
            path, W=self.W, b=self.b, alpha_train=np.float32(self.alpha_train)
        )

    @classmethod
    def load(cls, path) -> CoarModel:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                W=z["W"],
                b=z["b"],
                alpha_train=float(z["alpha_train"]),
                info={"loaded_from": str(path)},
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CoarModel(N={self.N}, E={self.E}, alpha_train={self.alpha_train})"


def fit_coar(
    masks: MaskStore,
    Y: np.ndarray,
    *,
    lam: float = 1.0,
    method: str = "auto",
    device: str = "cuda",
    chunk: int = 4096,
    tol: float = 1e-6,
    max_iter: int = 500,
    estimate_cond: bool = False,
) -> CoarModel:
    """Fit linear COAR for all examples simultaneously off the shared Gram."""
    sol: RidgeSolution = fit_ridge(
        masks,
        Y,
        lam,
        method=method,
        device=device,
        chunk=chunk,
        tol=tol,
        max_iter=max_iter,
        estimate_cond=estimate_cond,
    )
    return CoarModel(W=sol.W, b=sol.b, alpha_train=masks.spec.alpha, info=sol.info)


# ------------------------------------------------------------- prediction
def predict_chunked(
    W: np.ndarray,
    b: np.ndarray,
    masks: MaskStore,
    *,
    device: str = "cuda",
    chunk: int = 8192,
) -> np.ndarray:
    """``(m, E)`` predictions without materializing the dense design matrix."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Wt = torch.from_numpy(np.ascontiguousarray(W)).to(dev, torch.float32)
    bt = torch.from_numpy(np.ascontiguousarray(b)).to(dev, torch.float32)
    out = np.empty((len(masks), W.shape[1]), dtype=np.float32)
    for lo in range(0, len(masks), chunk):
        hi = min(lo + chunk, len(masks))
        Xc = torch.from_numpy(masks.unpack(lo, hi)).to(dev, torch.float32)
        out[lo:hi] = (Xc @ Wt + bt.unsqueeze(0)).cpu().numpy()
        del Xc
    return out


def residuals_chunked(
    W: np.ndarray,
    b: np.ndarray,
    masks: MaskStore,
    Y: np.ndarray,
    *,
    device: str = "cuda",
    chunk: int = 8192,
) -> np.ndarray:
    """``R = Y - (X W + b)`` -- the signal residual screening scores."""
    pred = predict_chunked(W, b, masks, device=device, chunk=chunk)
    return np.asarray(Y, dtype=np.float32) - pred


# -------------------------------------------------------------- evaluation
def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, *, label: str = ""
) -> dict:
    """Per-example rho / R^2 / MSE plus the target-variance SNR diagnostic.

    All three are reported at every alpha, never rho alone: the base paper's
    Fig. 13 shows correlation decaying on *both* sides of alpha_train, and only
    R^2 and the raw variance can separate misspecification from the
    signal-to-noise collapse at small alpha (CLAUDE.md 2.7).
    """
    rho = pearson_r(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r2_oc = r2_offset_corrected(y_true, y_pred)
    err = mse(y_true, y_pred)
    var = variance(y_true)
    return {
        "label": label,
        "rho_mean": float(np.nanmean(rho)),
        "rho_median": float(np.nanmedian(rho)),
        "rho_std": float(np.nanstd(rho)),
        "r2_mean": float(np.nanmean(r2)),
        "r2_median": float(np.nanmedian(r2)),
        "r2_oc_mean": float(np.nanmean(r2_oc)),
        "r2_oc_median": float(np.nanmedian(r2_oc)),
        "mse_mean": float(np.nanmean(err)),
        "target_var_mean": float(np.nanmean(var)),
        "n_examples": int(np.asarray(rho).size),
        "per_example": {
            "rho": rho,
            "r2": r2,
            "r2_oc": r2_oc,
            "mse": err,
            "target_var": var,
        },
    }


def evaluate_across_alpha(
    model: CoarModel,
    test_sets: dict[float, tuple[MaskStore, np.ndarray]],
    *,
    device: str = "cuda",
    chunk: int = 8192,
) -> list[dict]:
    """Evaluate one fitted model over the whole ``alpha_test`` grid."""
    rows: list[dict] = []
    for alpha in sorted(test_sets):
        masks, Y = test_sets[alpha]
        t0 = time.time()
        pred = predict_chunked(model.W, model.b, masks, device=device, chunk=chunk)
        res = evaluate_predictions(Y, pred, label=f"alpha_test={alpha}")
        res["alpha_test"] = alpha
        res["alpha_train"] = model.alpha_train
        res["is_ood"] = alpha != model.alpha_train
        res["seconds"] = time.time() - t0
        rows.append(res)
        print(
            f"[coar] alpha_test={alpha:<6.3f} rho={res['rho_mean']:.4f} "
            f"R2={res['r2_mean']:.4f} R2_oc={res['r2_oc_mean']:.4f} "
            f"MSE={res['mse_mean']:.4f} Var(f_M)={res['target_var_mean']:.4f}"
        )
    return rows
