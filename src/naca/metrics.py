"""Metrics: margins, per-example rho / R^2 / MSE, NAF, (epsilon, delta), Wilcoxon.

Everything here is fp32 (CLAUDE.md 5.4) and column-wise: the convention across
the codebase is that a matrix of shape ``(m, E)`` holds ``m`` counterfactual
observations for each of ``E`` evaluation examples, so every metric returns a
length-``E`` vector -- one number per example ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "correct_class_margin",
    "pearson_r",
    "r2_score",
    "r2_offset_corrected",
    "mse",
    "variance",
    "non_additive_fraction",
    "paired_wilcoxon",
    "EditEffect",
    "edit_epsilon_delta",
    "benjamini_hochberg",
    "two_sample_t",
]


# ---------------------------------------------------------------- margins
def correct_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """``logit_y - max_{y' != y} logit_{y'}`` (CLAUDE.md 2.1).

    Returned in fp32 regardless of the autocast dtype used for the forward pass.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (B, C), got {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(f"labels must be (B,), got {tuple(labels.shape)}")
    logits = logits.float()
    idx = labels.long().unsqueeze(1)
    correct = logits.gather(1, idx).squeeze(1)
    masked = logits.scatter(1, idx, float("-inf"))
    other = masked.max(dim=1).values
    return correct - other


# ------------------------------------------------------- column-wise stats
def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation. Zero-variance columns yield ``nan``."""
    yt, yp = _as2d(y_true), _as2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    yt = yt - yt.mean(axis=0, keepdims=True)
    yp = yp - yp.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column coefficient of determination, ``1 - SS_res / SS_tot``.

    Not squared-correlation: this one punishes scale and offset error, which is
    exactly why it is the primary metric and rho is kept only for comparability.
    """
    yt, yp = _as2d(y_true), _as2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    ss_res = ((yt - yp) ** 2).sum(axis=0)
    ss_tot = ((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def r2_offset_corrected(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """R^2 after removing the best constant offset per column.

    Required by `results/FINDING_gauge_degeneracy.md`. Fixed-size subset sampling
    leaves the additive model identifiable only up to ``w -> w + c*1``, and at
    ``alpha_test != alpha_train`` that unidentifiable constant shows up as a pure
    offset of size ``c*(k_train - k_test)`` -- which inflates raw R^2 and MSE the
    further you extrapolate, exactly mimicking the signature H2 predicts.

    Correcting it is equivalent to letting the intercept refit at test time, so
    this is the *best case* for the additive model and the gauge-invariant number.
    Report it alongside raw R^2, never instead of it, and apply it identically to
    every model being compared.
    """
    yt, yp = _as2d(y_true), _as2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    resid = yt - yp
    resid = resid - resid.mean(axis=0, keepdims=True)
    ss_res = (resid**2).sum(axis=0)
    ss_tot = ((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt, yp = _as2d(y_true), _as2d(y_pred)
    return ((yt - yp) ** 2).mean(axis=0)


def variance(y: np.ndarray) -> np.ndarray:
    """Per-column ``Var_t(f_M(z, C_t))`` -- the signal-to-noise diagnostic of 2.7."""
    return _as2d(y).var(axis=0, ddof=1)


def non_additive_fraction(r2_lin: np.ndarray, r2_naca: np.ndarray) -> np.ndarray:
    """``NAF = (R2_NACA - R2_lin) / (1 - R2_lin)`` -- the headline number.

    Scale-free: the share of the variance the additive model *left on the table*
    that the interaction model recovers. Columns where the linear model already
    explains everything (``1 - R2_lin`` at or below zero) are ``nan``, not
    silently huge.
    """
    r2_lin = np.asarray(r2_lin, dtype=np.float64)
    r2_naca = np.asarray(r2_naca, dtype=np.float64)
    denom = 1.0 - r2_lin
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 1e-12, (r2_naca - r2_lin) / denom, np.nan)


# ----------------------------------------------------------- significance
def paired_wilcoxon(a: np.ndarray, b: np.ndarray, alternative: str = "greater") -> dict:
    """Paired Wilcoxon signed-rank of ``a`` vs ``b`` across examples.

    Paired because the same example ``z`` is scored under both models
    (CLAUDE.md 2.6) -- an unpaired test would be wrong here.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    n_nonzero = int(np.count_nonzero(d))
    if n_nonzero == 0:
        return {
            "statistic": float("nan"),
            "pvalue": 1.0,
            "n": int(ok.sum()),
            "n_nonzero": 0,
            "median_diff": 0.0,
            "alternative": alternative,
            "note": "all paired differences are exactly zero",
        }
    stat, p = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "n": int(ok.sum()),
        "n_nonzero": n_nonzero,
        "median_diff": float(np.median(d)),
        "mean_diff": float(np.mean(d)),
        "alternative": alternative,
    }


# ----------------------------------------------------------------- editing
@dataclass
class EditEffect:
    """Base paper Definition 3, measured with **real** ablations, never surrogate."""

    delta: float  # mean |change| on the target distribution D_T
    epsilon: float  # mean |change| on the reference distribution D_R
    n_target: int
    n_reference: int
    delta_signed: float
    epsilon_signed: float
    n_edit: int

    def to_dict(self) -> dict:
        return {
            "delta": self.delta,
            "epsilon": self.epsilon,
            "delta_signed": self.delta_signed,
            "epsilon_signed": self.epsilon_signed,
            "n_target": self.n_target,
            "n_reference": self.n_reference,
            "n_edit": self.n_edit,
        }


def edit_epsilon_delta(
    margins_target_before: np.ndarray,
    margins_target_after: np.ndarray,
    margins_ref_before: np.ndarray,
    margins_ref_after: np.ndarray,
    n_edit: int,
) -> EditEffect:
    """``delta`` over the target set, ``epsilon`` over the reference set.

    ``delta = E_{D_T}[|f_M(z, C_edit) - f_M(z, empty)|]`` and likewise for
    ``epsilon`` over ``D_R``. Signed means are carried alongside because the
    absolute value hides whether an edit pushed margins up or down.
    """
    dt = np.asarray(margins_target_after, np.float64) - np.asarray(
        margins_target_before, np.float64
    )
    dr = np.asarray(margins_ref_after, np.float64) - np.asarray(margins_ref_before, np.float64)
    return EditEffect(
        delta=float(np.abs(dt).mean()),
        epsilon=float(np.abs(dr).mean()),
        delta_signed=float(dt.mean()),
        epsilon_signed=float(dr.mean()),
        n_target=int(dt.size),
        n_reference=int(dr.size),
        n_edit=int(n_edit),
    )


# ------------------------------------------------------ COAR-EDIT machinery
def two_sample_t(a: np.ndarray, b: np.ndarray, axis: int = 0) -> np.ndarray:
    """Welch two-sample t-statistic, the base paper's ``tau(c_i)`` (Eq. 8).

    ``a`` holds per-component attributions over the target set, ``b`` over the
    reference set; both ``(n_examples, N)``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.shape[axis], b.shape[axis]
    if na < 2 or nb < 2:
        raise ValueError(f"need >=2 samples per group, got {na} and {nb}")
    ma, mb = a.mean(axis=axis), b.mean(axis=axis)
    va, vb = a.var(axis=axis, ddof=1), b.var(axis=axis, ddof=1)
    denom = np.sqrt(va / na + vb / nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (ma - mb) / denom, 0.0)


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Boolean mask of hypotheses passing BH-FDR at level ``q``."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"pvalues must be 1-D, got {p.shape}")
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * np.arange(1, n + 1) / n
    passing = ranked <= thresh
    keep = np.zeros(n, dtype=bool)
    if passing.any():
        cutoff = int(np.max(np.nonzero(passing)[0]))
        keep[order[: cutoff + 1]] = True
    return keep
