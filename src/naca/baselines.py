"""Competing estimators, including the one that could sink the paper.

**Multi-alpha linear COAR is the danger baseline** (the design spec Phase 6). The whole
interaction story rests on the claim that the additive model degrades away from
``alpha_train`` *because it is misspecified*. But a much duller explanation is
available: a linear model fitted at one ablation rate is simply mis-calibrated
for another, and fitting several and picking the nearest would fix it for free.
If that closes the out-of-distribution gap, the estimator contribution is
undercut and the scope needs revising before more work goes in. It is cheap.
Run it first, and believe the answer.

The rest -- a K-restricted MLP, gradient-boosted trees, and leave-one-out -- are
the "obvious alternative" arms. The PSD ablation is not here: it is the same code
path as the main estimator with ``r_P = 0``, which is exactly what makes it a
clean comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .ablation import Ablator
from .coar import CoarModel, fit_coar, predict_chunked
from .collect import _forward_margins
from .components import ComponentIndex
from .masks import MaskStore
from .metrics import mse, pearson_r, r2_offset_corrected, r2_score

__all__ = [
    "MultiAlphaCoar",
    "fit_multi_alpha",
    "fit_mlp_baseline",
    "fit_gbt_baseline",
    "leave_one_out",
    "score_predictions",
]


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rho_mean": float(np.nanmean(pearson_r(y_true, y_pred))),
        "r2_mean": float(np.nanmean(r2_score(y_true, y_pred))),
        "r2_oc_mean": float(np.nanmean(r2_offset_corrected(y_true, y_pred))),
        "mse_mean": float(np.nanmean(mse(y_true, y_pred))),
    }


# ------------------------------------------------------- DANGER BASELINE
@dataclass
class MultiAlphaCoar:
    """A bank of linear COAR models, one per training alpha."""

    models: dict[float, CoarModel] = field(default_factory=dict)

    @property
    def alphas(self) -> list[float]:
        return sorted(self.models)

    def nearest(self, alpha_test: float) -> float:
        return min(self.alphas, key=lambda a: abs(a - alpha_test))

    def predict_nearest(
        self, masks: MaskStore, alpha_test: float, *, device: str = "cuda"
    ) -> tuple[np.ndarray, float]:
        a = self.nearest(alpha_test)
        m = self.models[a]
        return predict_chunked(m.W, m.b, masks, device=device), a

    def predict_interpolated(
        self, masks: MaskStore, alpha_test: float, *, device: str = "cuda"
    ) -> np.ndarray:
        """Linearly interpolate the two bracketing models' coefficients.

        Interpolating *coefficients* rather than predictions is the stronger
        version of this baseline: it is what a practitioner would actually do
        with a bank of fitted models, and it is what we have to beat.
        """
        al = self.alphas
        if alpha_test <= al[0]:
            m = self.models[al[0]]
            return predict_chunked(m.W, m.b, masks, device=device)
        if alpha_test >= al[-1]:
            m = self.models[al[-1]]
            return predict_chunked(m.W, m.b, masks, device=device)
        hi_i = next(i for i, a in enumerate(al) if a >= alpha_test)
        lo_a, hi_a = al[hi_i - 1], al[hi_i]
        t = (alpha_test - lo_a) / max(hi_a - lo_a, 1e-12)
        lo_m, hi_m = self.models[lo_a], self.models[hi_a]
        W = (1 - t) * lo_m.W + t * hi_m.W
        b = (1 - t) * lo_m.b + t * hi_m.b
        return predict_chunked(W, b, masks, device=device)


def fit_multi_alpha(
    train_sets: dict[float, tuple[MaskStore, np.ndarray]],
    *,
    lam: float = 1.0,
    device: str = "cuda",
    method: str = "auto",
) -> MultiAlphaCoar:
    """Fit one linear COAR per available ``alpha_train``."""
    bank = MultiAlphaCoar()
    for alpha in sorted(train_sets):
        masks, Y = train_sets[alpha]
        print(f"[multi-alpha] fitting at alpha_train={alpha} (m={len(masks)})")
        bank.models[alpha] = fit_coar(masks, Y, lam=lam, method=method, device=device)
    return bank


# ------------------------------------------------------------- MLP / GBT
class _PoolMLP(nn.Module):
    """Two hidden layers on the K-restricted ablation vector."""

    def __init__(self, K: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_mlp_baseline(
    masks: MaskStore,
    Y: np.ndarray,
    lin_pred: np.ndarray,
    pool: np.ndarray,
    *,
    hidden: int = 64,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 1024,
    device: str = "cuda",
    seed: int = 0,
    max_examples: int | None = 64,
    verbose: bool = True,
) -> dict:
    """MLP on the pooled ablation vector, fitted to the Stage-1 residual.

    Fitted to the residual, not the raw target, so it is solving exactly the same
    problem as the interaction model -- otherwise the comparison would be about
    who models the linear part better, which is not the question.

    ``max_examples`` caps how many examples get their own MLP: this arm exists to
    answer "could a generic non-linear model do better", and a subsample answers
    that at a fraction of the cost.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    m, E = len(masks), int(Y.shape[1])
    shared = pool.ndim == 1
    K = int(pool.shape[-1] if not shared else pool.shape[0])
    n_fit = E if max_examples is None else min(E, max_examples)

    X = torch.from_numpy(masks.unpack()).to(dev)
    R = torch.from_numpy(
        np.ascontiguousarray(np.asarray(Y, np.float32) - np.asarray(lin_pred, np.float32))
    ).to(dev)

    preds = np.zeros((m, n_fit), dtype=np.float32)
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(seed)
    for e in range(n_fit):
        cols = torch.from_numpy(
            np.ascontiguousarray(pool if shared else pool[e])
        ).to(dev, torch.long)
        V = X[:, cols].to(torch.float32)
        target = R[:, e]

        net = _PoolMLP(K, hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        steps = max(1, m // batch_size)
        for _ in range(epochs):
            perm = torch.randperm(m, device=dev, generator=gen)
            for s in range(steps):
                idx = perm[s * batch_size : (s + 1) * batch_size]
                loss = ((net(V[idx]) - target[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            out = torch.cat([net(V[i : i + 8192]) for i in range(0, m, 8192)])
        preds[:, e] = out.cpu().numpy()
        del V, net, opt
    if verbose:
        print(f"[mlp] fitted {n_fit} examples in {(time.time() - t0) / 60:.1f} min")
    del X, R
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "residual_pred": preds,
        "n_examples": n_fit,
        "hidden": hidden,
        "seconds": time.time() - t0,
    }


def fit_gbt_baseline(
    masks: MaskStore,
    Y: np.ndarray,
    lin_pred: np.ndarray,
    pool: np.ndarray,
    *,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    max_examples: int | None = 32,
    subsample_rows: int | None = 20_000,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Gradient-boosted trees on the pooled ablation vector, fitted to the residual.

    Captures interactions non-parametrically, so it bounds how much of the
    residual is *reachable* structure rather than noise -- a useful ceiling to
    compare the low-rank model against.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    m, E = len(masks), int(Y.shape[1])
    shared = pool.ndim == 1
    n_fit = E if max_examples is None else min(E, max_examples)
    rows = np.arange(m) if subsample_rows is None else np.arange(min(m, subsample_rows))

    X = masks.unpack()
    R = np.asarray(Y, np.float32) - np.asarray(lin_pred, np.float32)
    preds = np.zeros((m, n_fit), dtype=np.float32)

    t0 = time.time()
    for e in range(n_fit):
        cols = pool if shared else pool[e]
        V = X[:, cols]
        gbt = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            early_stopping=False,
        )
        gbt.fit(V[rows], R[rows, e])
        preds[:, e] = gbt.predict(V)
    if verbose:
        print(
            f"[gbt] fitted {n_fit} examples on {len(rows)} rows in "
            f"{(time.time() - t0) / 60:.1f} min"
        )
    return {
        "residual_pred": preds,
        "n_examples": n_fit,
        "n_rows": int(len(rows)),
        "seconds": time.time() - t0,
    }


# ------------------------------------------------------------------- LOO
@torch.inference_mode()
def leave_one_out(
    model: nn.Module,
    index: ComponentIndex,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    clean: np.ndarray,
    *,
    sub_batch: int = 1000,
    amp_dtype: torch.dtype = torch.bfloat16,
    components: np.ndarray | None = None,
    log_every: int = 500,
) -> np.ndarray:
    """``LOO[i, e] = f_M(z_e, {c_i}) - f_M(z_e, empty)`` -- N real ablations.

    The base paper's simplest attribution baseline. Exact, and ``O(N)`` forward
    passes over the eval set, which is affordable only because ``N`` is small at
    Setup A.
    """
    comps = np.arange(index.N) if components is None else np.asarray(components)
    ablator = Ablator(model, index)
    buf = torch.empty(int(x_eval.shape[0]), dtype=torch.float32, device=x_eval.device)
    out = np.zeros((len(comps), int(x_eval.shape[0])), dtype=np.float32)

    model.eval()
    t0 = time.time()
    try:
        for j, ci in enumerate(comps):
            v = torch.ones(index.N, device=x_eval.device)
            v[int(ci)] = 0.0
            ablator.apply(v)
            _forward_margins(model, x_eval, y_eval, buf, sub_batch, amp_dtype)
            out[j] = buf.cpu().numpy() - clean
            if log_every and (j + 1) % log_every == 0:
                rate = (j + 1) / max(time.time() - t0, 1e-9)
                print(
                    f"[loo] {j + 1}/{len(comps)}  {rate:.1f} comp/s  "
                    f"ETA {(len(comps) - j - 1) / max(rate, 1e-9) / 60:.1f}m"
                )
    finally:
        ablator.restore()
    return out
