"""Stage 4 -- COAR-EDIT baseline and interaction-aware NACA-EDIT.

COAR-EDIT scores every component independently with a two-sample t-statistic and
takes top-/bottom-``k`` (base paper Eq. 8). By construction it cannot see
redundancy (two components that do the same job, so ablating both buys little)
or complementarity (two that only matter together). NACA-EDIT replaces the
ranking with greedy forward search under the fitted surrogate, which can.

The greedy step is ``O(K)``, not a re-evaluation of ``g`` (CLAUDE.md 2.5). With
``diag(A) = 0``::

    g(v) = b + w'v + (1/2) v'Av

Flipping ``v_i : 1 -> 0`` removes exactly the terms containing ``i``, which sum
to ``v_i (Av)_i``. So with ``a = Av`` carried along::

    Delta(c_i | C_edit) = -( w_i + a_i )

and committing ``c*`` is ``v[c*] <- 0``, ``a <- a - A[:, c*]``. Test 9 pins this
against full re-evaluation.

Selection needs a **shared** component pool across all examples, otherwise
``Delta_T`` averages quantities indexed differently per example and the argmax is
meaningless. That is why `screening.shared_pool` exists.

Nothing here reports a result: surrogate-predicted edit effects are search
signal only. Every number in the paper comes from :func:`verify_edit`, which
performs the real ablation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .metrics import benjamini_hochberg, two_sample_t

__all__ = [
    "EditSelection",
    "coar_edit_select",
    "naca_edit_select",
    "fdr_candidate_filter",
    "surrogate_delta",
    "verify_edit",
]

Direction = Literal["increase", "suppress"]


@dataclass
class EditSelection:
    """A chosen edit set plus the trace that produced it."""

    components: np.ndarray  # global component indices, in selection order
    method: str
    direction: Direction
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trace: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.components.size)

    def to_dict(self) -> dict:
        return {
            "components": self.components.tolist(),
            "method": self.method,
            "direction": self.direction,
            "n_edit": len(self),
            "trace": self.trace,
            "info": self.info,
        }


# ------------------------------------------------------------- COAR-EDIT
def coar_edit_select(
    W: np.ndarray,
    target_idx: np.ndarray,
    ref_idx: np.ndarray,
    k: int,
    *,
    direction: Direction = "increase",
    candidates: np.ndarray | None = None,
) -> EditSelection:
    """Base-paper COAR-EDIT: rank by the two-sample t-statistic, take ``k``.

    ``W`` is ``(N, E)`` Stage-1 attributions. ``tau(c_i)`` contrasts component
    ``i``'s attribution on the target examples against the reference examples.

    Ablating ``c_i`` changes the margin by ``-w_i``. So to **increase** the target
    margin we ablate the most *negatively* contributing components (bottom-k by
    tau); to **suppress** it we ablate the most positively contributing ones
    (top-k by tau).
    """
    tau = two_sample_t(W[:, target_idx].T, W[:, ref_idx].T)
    pool = np.arange(W.shape[0]) if candidates is None else np.asarray(candidates)
    sub = tau[pool]
    order = np.argsort(sub) if direction == "increase" else np.argsort(-sub)
    chosen = pool[order[:k]]
    return EditSelection(
        components=np.asarray(chosen, dtype=np.int64),
        method="coar-edit",
        direction=direction,
        scores=tau,
        info={
            "n_candidates": int(pool.size),
            "n_target": int(len(target_idx)),
            "n_reference": int(len(ref_idx)),
        },
    )


def fdr_candidate_filter(
    W: np.ndarray,
    target_idx: np.ndarray,
    ref_idx: np.ndarray,
    q: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the candidate pool to components passing BH-FDR at level ``q``.

    Returns ``(candidate_indices, tau)``. Cheap, and it keeps the search from
    spending its budget on components whose target/reference difference is noise.
    """
    from scipy.stats import t as student_t

    a = W[:, target_idx].T
    b = W[:, ref_idx].T
    tau = two_sample_t(a, b)
    na, nb = a.shape[0], b.shape[0]
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    # Welch-Satterthwaite degrees of freedom
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        df = np.where(den > 0, num / den, float(na + nb - 2))
    df = np.clip(np.nan_to_num(df, nan=na + nb - 2.0), 1.0, None)
    p = 2.0 * student_t.sf(np.abs(tau), df)
    keep = benjamini_hochberg(p, q=q)
    idx = np.nonzero(keep)[0].astype(np.int64)
    print(
        f"[edit] FDR filter q={q}: {idx.size}/{W.shape[0]} components pass "
        f"({idx.size / W.shape[0]:.1%})"
    )
    return idx, tau


# ------------------------------------------------------------- NACA-EDIT
def surrogate_delta(
    w_pool: np.ndarray, A: np.ndarray, v: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """``Delta(c_i | C_edit) = -(w_i + a_i)`` per example, shape ``(E, K)``.

    Already-ablated positions are returned as ``nan`` so they cannot be chosen
    twice.
    """
    d = -(w_pool + a)
    return np.where(v > 0, d, np.nan)


def naca_edit_select(
    w_pool: np.ndarray,
    A: np.ndarray,
    pool: np.ndarray,
    target_idx: np.ndarray,
    ref_idx: np.ndarray,
    k: int,
    *,
    direction: Direction = "increase",
    lambda_edit: float = 1.0,
    gain_tol: float = 1e-6,
    candidates: np.ndarray | None = None,
) -> EditSelection:
    """Greedy forward selection under the interaction-aware surrogate.

    Parameters
    ----------
    w_pool, A
        ``(E, K)`` linear coefficients and ``(E, K, K)`` interaction matrices on
        a **shared** pool.
    pool
        ``(K,)`` global component indices the pool refers to.
    candidates
        Optional restriction (global indices), e.g. the FDR-filtered set.
    """
    E, K = w_pool.shape
    if A.shape != (E, K, K):
        raise ValueError(f"A must be {(E, K, K)}, got {A.shape}")
    if pool.ndim != 1 or pool.shape[0] != K:
        raise ValueError("NACA-EDIT needs a shared pool of shape (K,)")

    allowed = np.ones(K, dtype=bool)
    if candidates is not None:
        allowed = np.isin(pool, np.asarray(candidates))
        if not allowed.any():
            raise ValueError("no pool component survived the candidate filter")

    v = np.ones((E, K), dtype=np.float32)
    a = np.zeros((E, K), dtype=np.float32)
    # a = A v with v all-ones is the row-sum of A
    for e in range(E):
        a[e] = A[e].sum(axis=1)

    sign = 1.0 if direction == "increase" else -1.0
    chosen: list[int] = []
    trace: list[dict] = []

    for step in range(k):
        # v is identical across examples (the same set is ablated for all), so
        # availability is one mask, and averaging over only the available
        # columns keeps this exact and free of empty-slice warnings.
        avail = v[0] > 0
        cols = np.nonzero(avail)[0]
        d = surrogate_delta(w_pool, A, v, a)  # (E, K), nan where ablated
        dT = np.full(K, np.nan, dtype=np.float64)
        dR = np.full(K, np.nan, dtype=np.float64)
        dT[cols] = d[np.ix_(target_idx, cols)].mean(axis=0)
        dR[cols] = d[np.ix_(ref_idx, cols)].mean(axis=0)
        obj = sign * dT - lambda_edit * np.abs(dR)
        obj = np.where(allowed & avail, obj, -np.inf)

        if not np.isfinite(obj).any():
            break
        j = int(np.argmax(obj))
        gain = float(obj[j])
        if step > 0 and gain < gain_tol and not np.isfinite(gain):
            break

        chosen.append(int(pool[j]))
        trace.append(
            {
                "step": step,
                "pool_index": j,
                "component": int(pool[j]),
                "objective": gain,
                "delta_target": float(dT[j]),
                "delta_reference": float(dR[j]),
            }
        )
        # commit: v_j <- 0 and a <- a - A[:, :, j]
        v[:, j] = 0.0
        a -= A[:, :, j]

    return EditSelection(
        components=np.asarray(chosen, dtype=np.int64),
        method="naca-edit",
        direction=direction,
        trace=trace,
        info={
            "K": K,
            "lambda_edit": lambda_edit,
            "n_candidates": int(allowed.sum()),
            "n_target": int(len(target_idx)),
            "n_reference": int(len(ref_idx)),
        },
    )


def greedy_reference_slow(
    w_pool: np.ndarray,
    A: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Full ``g`` evaluation (minus the constant ``b``), for test 9.

    ``g(v) - b = w'v + (1/2) v'Av``, computed directly rather than incrementally.
    """
    lin = np.einsum("ek,ek->e", w_pool, v)
    quad = 0.5 * np.einsum("ek,ekl,el->e", v, A, v)
    return lin + quad


# ------------------------------------------------------------ verification
def verify_edit(
    selection: EditSelection,
    measure: Callable[[np.ndarray | None], tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Measure the true ``(epsilon, delta)`` by really ablating the model.

    ``measure(components)`` must return ``(target_margins, reference_margins)``;
    ``measure(None)`` returns the unedited baselines. Surrogate predictions never
    appear in a reported result (CLAUDE.md 2.5, 8).
    """
    from .metrics import edit_epsilon_delta

    tb, rb = measure(None)
    ta, ra = measure(selection.components)
    eff = edit_epsilon_delta(tb, ta, rb, ra, n_edit=len(selection))
    out = eff.to_dict()
    out["method"] = selection.method
    out["direction"] = selection.direction
    return out
