"""Stage 3 -- the indefinite low-rank interaction model.

A standard factorization machine parameterizes ``A = VV'``, which is positive
semi-definite. That is the wrong prior twice over (CLAUDE.md 2.4): there is no
reason redundant and complementary components should interact with the same
sign, and a convex quadratic set function is supermodular, which would destroy
any guarantee for the greedy selection in Stage 4. So::

    A = U U' - P P',    U in R^{K x r_U},  P in R^{K x r_P},   diag(A) = 0

with ``r_P = 0`` recovering the PSD ablation arm.

Cost: ``K(r_U + r_P) = 1600`` parameters against ``m >= 50,000`` samples, fitted
on the counterfactuals COAR already collected. **Zero additional forward passes
through the target model** -- the load-bearing feasibility claim of the project.

Identifiability
---------------
``U`` and ``P`` are identified only up to an orthogonal rotation *within* each
block, so individual embedding rows mean nothing; only the inner products
``A_ij`` are interpretable. The API enforces that: the parameters are private
and :meth:`InteractionModel.interaction_matrix` is the only way out.

Efficiency
----------
Examples are fitted in **groups** with a batched parameter tensor, because each
individual problem (K=100 features) is far too small to saturate a GPU -- the
cost is kernel-launch overhead, and grouping amortizes it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .masks import MaskStore

__all__ = ["IndefiniteFM", "InteractionModel", "fit_interaction", "quad_naive"]


class IndefiniteFM(nn.Module):
    """Batched indefinite factorization machine over ``n_models`` examples.

    ``_U`` / ``_P`` are private by contract: see the identifiability note above.
    """

    def __init__(
        self,
        n_models: int,
        K: int,
        r_U: int = 8,
        r_P: int = 8,
        *,
        init_std: float = 0.01,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if r_U < 1:
            raise ValueError("r_U must be >= 1")
        if r_P < 0:
            raise ValueError("r_P must be >= 0 (0 = PSD ablation)")
        self.n_models, self.K, self.r_U, self.r_P = n_models, K, r_U, r_P

        def _init(r: int) -> nn.Parameter:
            t = torch.empty(n_models, K, r)
            with torch.no_grad():
                t.normal_(0.0, init_std, generator=generator)
            return nn.Parameter(t)

        self._U = _init(r_U)
        self._P = _init(r_P) if r_P > 0 else None

    # ------------------------------------------------------------------
    @staticmethod
    def _half_fm(V: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """``0.5 * (||sum_i v_i m_i||^2 - sum_i v_i ||m_i||^2)`` -- the FM identity.

        ``V``: ``(B, G, K)`` binary. ``M``: ``(G, K, r)``. Returns ``(B, G)``.

        The second term is what removes the diagonal. It is not an approximation:
        since ``v_i in {0,1}`` we have ``v_i^2 = v_i``, so diagonal terms are
        absorbable into the linear term and must not be double-counted.
        """
        z = torch.einsum("bgk,gkr->bgr", V, M)
        sq_norms = (M * M).sum(dim=-1)  # (G, K)
        return 0.5 * ((z * z).sum(dim=-1) - torch.einsum("bgk,gk->bg", V, sq_norms))

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """``V`` of shape ``(B, G, K)`` -> quadratic term ``(B, G)``."""
        out = self._half_fm(V, self._U)
        if self._P is not None:
            out = out - self._half_fm(V, self._P)
        return out

    def penalty(self) -> torch.Tensor:
        pen = (self._U * self._U).sum()
        if self._P is not None:
            pen = pen + (self._P * self._P).sum()
        return pen

    # ------------------------------------------------------------------
    @torch.no_grad()
    def interaction_matrix(self, g: int | None = None) -> torch.Tensor:
        """``A = UU' - PP'`` with ``diag(A) = 0``.

        The **only** sanctioned view of the fitted parameters.
        """
        U = self._U if g is None else self._U[g : g + 1]
        A = U @ U.transpose(-1, -2)
        if self._P is not None:
            P = self._P if g is None else self._P[g : g + 1]
            A = A - P @ P.transpose(-1, -2)
        A = A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
        return A[0] if g is not None else A


def quad_naive(V: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Reference ``O(K^2)`` double sum ``sum_{i<j} A_ij v_i v_j``, for test 6."""
    if V.ndim != 2:
        raise ValueError("expected V of shape (B, K)")
    K = V.shape[1]
    iu = torch.triu_indices(K, K, offset=1, device=V.device)
    return (A[iu[0], iu[1]] * V[:, iu[0]] * V[:, iu[1]]).sum(dim=1)


# --------------------------------------------------------------------------
@dataclass
class InteractionModel:
    """Fitted Stage-3 model for every evaluation example."""

    A: np.ndarray  # (E, K, K) interaction matrices, diag 0
    pool: np.ndarray  # (E, K) component indices, or (K,) if shared
    w_pool: np.ndarray  # (E, K) linear coefficients on the pool
    b: np.ndarray  # (E,) intercepts
    shared_pool: bool
    freeze_linear: bool
    info: dict = field(default_factory=dict)

    @property
    def E(self) -> int:
        return int(self.A.shape[0])

    @property
    def K(self) -> int:
        return int(self.A.shape[1])

    def pool_for(self, e: int) -> np.ndarray:
        return self.pool if self.shared_pool else self.pool[e]

    def interaction_matrix(self, e: int) -> np.ndarray:
        return self.A[e]

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            A=self.A,
            pool=self.pool,
            w_pool=self.w_pool,
            b=self.b,
            shared_pool=np.array(self.shared_pool),
            freeze_linear=np.array(self.freeze_linear),
        )

    @classmethod
    def load(cls, path) -> InteractionModel:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                A=z["A"],
                pool=z["pool"],
                w_pool=z["w_pool"],
                b=z["b"],
                shared_pool=bool(z["shared_pool"]),
                freeze_linear=bool(z["freeze_linear"]),
            )

    # ------------------------------------------------------------------
    def predict(
        self,
        masks: MaskStore,
        lin_pred: np.ndarray,
        *,
        device: str = "cuda",
        group_size: int = 32,
        row_chunk: int = 8192,
    ) -> np.ndarray:
        """``lin_pred + quadratic term``, shape ``(m, E)``.

        ``lin_pred`` is the Stage-1 prediction on the same masks; passing it in
        keeps this a pure add-on and makes the "zero extra forward passes"
        property visible in the call signature.
        """
        m, E = len(masks), self.E
        if lin_pred.shape != (m, E):
            raise ValueError(f"lin_pred must be {(m, E)}, got {lin_pred.shape}")
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        out = np.asarray(lin_pred, dtype=np.float32).copy()

        A_t = torch.from_numpy(self.A).to(dev, torch.float32)
        for g0 in range(0, E, group_size):
            g1 = min(g0 + group_size, E)
            pool_g = self._pool_block(g0, g1)
            pool_t = torch.from_numpy(pool_g).to(dev, torch.long)
            Ag = A_t[g0:g1]
            for lo in range(0, m, row_chunk):
                hi = min(lo + row_chunk, m)
                Xc = torch.from_numpy(masks.unpack(lo, hi)).to(dev, torch.float32)
                V = Xc[:, pool_t]  # (B, G, K)
                # sum_{i<j} A_ij v_i v_j  ==  0.5 * v' A v   when diag(A) = 0
                quad = 0.5 * torch.einsum("bgk,gkl,bgl->bg", V, Ag, V)
                out[lo:hi, g0:g1] += quad.cpu().numpy()
                del Xc, V, quad
        return out

    def _pool_block(self, g0: int, g1: int) -> np.ndarray:
        if self.shared_pool:
            return np.broadcast_to(self.pool, (g1 - g0, self.K)).copy()
        return self.pool[g0:g1]


# --------------------------------------------------------------------------
def fit_interaction(
    masks: MaskStore,
    Y: np.ndarray,
    lin_pred: np.ndarray,
    pool: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    *,
    r_U: int = 8,
    r_P: int = 8,
    lambda_fm: float = 1e-4,
    freeze_linear: bool = True,
    lr: float = 1e-2,
    epochs: int = 20,
    batch_size: int = 1024,
    init_std: float = 0.01,
    group_size: int = 32,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> InteractionModel:
    """Fit ``A`` per example on the counterfactuals COAR already collected.

    With ``freeze_linear=True`` the linear part is held at its Stage-1 value, so
    this is exactly "fit the quadratic to the Stage-1 residual". With
    ``freeze_linear=False`` the pool's linear coefficients and the intercept are
    refit jointly; the off-pool linear contribution stays frozen either way,
    since components outside the pool have no interaction terms to trade against.
    """
    m, E = len(masks), int(Y.shape[1])
    shared = pool.ndim == 1
    K = int(pool.shape[-1] if not shared else pool.shape[0])
    if lin_pred.shape != (m, E):
        raise ValueError(f"lin_pred must be {(m, E)}, got {lin_pred.shape}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Batch order must be seeded too. torch.randperm on CUDA draws from the
    # global CUDA RNG unless given a generator, which made repeat fits of the
    # same data differ by ~0.005 in held-out R^2 -- enough to make a threshold
    # test flaky and to make a reported number irreproducible.
    perm_gen = torch.Generator(device=dev).manual_seed(seed + 1_000_003)

    A_out = np.zeros((E, K, K), dtype=np.float32)
    w_out = np.zeros((E, K), dtype=np.float32)
    b_out = np.zeros(E, dtype=np.float32)

    # Full mask matrix as uint8 on the GPU: 50,000 x 3,904 = 195 MB for Setup A.
    # Falls back to CPU-side streaming if that would not fit the VRAM budget.
    bytes_needed = m * masks.N
    preload = dev.type == "cuda" and bytes_needed < 1.2 * 1024**3
    if preload:
        X_all = torch.from_numpy(masks.unpack()).to(dev)  # uint8
        if verbose:
            print(f"[naca] masks preloaded to GPU as uint8 ({bytes_needed / 1024**2:.0f} MB)")
    else:
        X_all = torch.from_numpy(masks.unpack())  # uint8, CPU
        if verbose:
            print(f"[naca] masks kept on CPU ({bytes_needed / 1024**2:.0f} MB), streaming")

    steps_per_epoch = max(1, m // batch_size)
    t0 = time.time()
    history: list[dict] = []

    for g0 in range(0, E, group_size):
        g1 = min(g0 + group_size, E)
        G = g1 - g0
        pool_g = (
            np.broadcast_to(pool, (G, K)).copy() if shared else pool[g0:g1]
        )  # (G, K)
        pool_t = torch.from_numpy(np.ascontiguousarray(pool_g)).to(X_all.device, torch.long)

        # (m, G, K) uint8 gather -- 160 MB at G=32, K=100, m=50k
        Vg_all = X_all[:, pool_t]
        if Vg_all.device != dev:
            Vg_all = Vg_all.to(dev)

        y_g = torch.from_numpy(np.ascontiguousarray(Y[:, g0:g1])).to(dev, torch.float32)
        lin_g = torch.from_numpy(np.ascontiguousarray(lin_pred[:, g0:g1])).to(
            dev, torch.float32
        )

        # Stage-1 linear coefficients restricted to each example's pool
        w_init = np.take_along_axis(W[:, g0:g1].T, pool_g, axis=1)  # (G, K)
        w_init_t = torch.from_numpy(np.ascontiguousarray(w_init)).to(dev, torch.float32)
        b_init_t = torch.from_numpy(np.ascontiguousarray(b[g0:g1])).to(dev, torch.float32)

        model = IndefiniteFM(
            G, K, r_U, r_P, init_std=init_std, generator=gen
        ).to(dev)
        params: list[nn.Parameter] = list(model.parameters())

        if freeze_linear:
            w_par = None
            b_par = None
        else:
            w_par = nn.Parameter(w_init_t.clone())
            b_par = nn.Parameter(b_init_t.clone())
            params += [w_par, b_par]

        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs * steps_per_epoch)
        )

        for epoch in range(epochs):
            perm = torch.randperm(m, device=dev, generator=perm_gen)
            running = 0.0
            for s in range(steps_per_epoch):
                idx = perm[s * batch_size : (s + 1) * batch_size]
                V = Vg_all[idx].to(torch.float32)  # (B, G, K)
                target = y_g[idx]
                base = lin_g[idx]

                pred = model(V)
                if freeze_linear:
                    pred = pred + base
                else:
                    # base already contains the Stage-1 pool contribution; strip
                    # it out so the trainable pool terms replace rather than
                    # double-count it.
                    off_pool = base - torch.einsum("bgk,gk->bg", V, w_init_t) - b_init_t
                    pred = pred + off_pool + torch.einsum("bgk,gk->bg", V, w_par) + b_par

                loss = ((pred - target) ** 2).mean() + lambda_fm * model.penalty()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                running += float(loss.detach())
            if verbose and (epoch == epochs - 1):
                history.append(
                    {
                        "group": [g0, g1],
                        "final_epoch_loss": running / steps_per_epoch,
                    }
                )

        A_out[g0:g1] = model.interaction_matrix().cpu().numpy()
        if freeze_linear:
            w_out[g0:g1] = w_init
            b_out[g0:g1] = b[g0:g1]
        else:
            w_out[g0:g1] = w_par.detach().cpu().numpy()
            b_out[g0:g1] = b_par.detach().cpu().numpy()

        del Vg_all, y_g, lin_g, model, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        if verbose:
            done = g1
            rate = done / max(time.time() - t0, 1e-9)
            print(
                f"[naca] fitted {done:>5d}/{E} examples  "
                f"({rate:5.1f} ex/s, ETA {(E - done) / max(rate, 1e-9) / 60:4.1f}m)"
            )

    del X_all
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    info = {
        "K": K,
        "r_U": r_U,
        "r_P": r_P,
        "psd_only": r_P == 0,
        "lambda_fm": lambda_fm,
        "freeze_linear": freeze_linear,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_params_per_example": K * (r_U + r_P),
        "m": m,
        "samples_per_param": m / max(1, K * (r_U + r_P)),
        "seconds": time.time() - t0,
        "history": history,
    }
    if verbose:
        print(
            f"[naca] Stage 3 done in {info['seconds'] / 60:.1f} min "
            f"({info['n_params_per_example']} params/example vs m={m}, "
            f"{info['samples_per_param']:.1f} samples/param)"
        )
    return InteractionModel(
        A=A_out,
        pool=pool,
        w_pool=w_out,
        b=b_out,
        shared_pool=shared,
        freeze_linear=freeze_linear,
        info=info,
    )


def expected_co_ablations(m: int, alpha: float) -> float:
    """Expected number of masks in which a given pair is *jointly ablated*.

    The identifiability calculation from the proposal: ~``m * alpha^2``, which
    depends on ``m`` and ``alpha`` but not on ``N``.
    """
    return m * alpha * alpha


def identifiability_report(m: int, alpha: float, K: int, r_U: int, r_P: int) -> dict:
    n_pairs = K * (K - 1) // 2
    n_params = K * (r_U + r_P)
    return {
        "m": m,
        "alpha": alpha,
        "expected_co_ablations_per_pair": expected_co_ablations(m, alpha),
        "n_pairs_in_pool": n_pairs,
        "n_free_parameters": n_params,
        "samples_per_parameter": m / max(1, n_params),
        "pairs_per_parameter": n_pairs / max(1, n_params),
    }
