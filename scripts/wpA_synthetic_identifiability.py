"""A synthetic demonstration of the proposition itself, with no network involved.

The existing positive control plants *pairwise* structure and shows the estimator
finds it. It says nothing about the proposition, which is a statement about the
design matrix. This script isolates that statement.

Data-generating process, exactly the form in Corollary 1:

    y(v) = b + w'v + beta * f(n) + eps,     n = sum(v)

with ``beta`` KNOWN and nonzero, and ``f`` one of the two orders the corollary
names:

    f(n) = n                 the linear case    (the coefficient-shift ambiguity)
    f(n) = n(n-1)/2          the quadratic case (the aggregate interaction term)

``w``, ``b`` and the noise scale are set to representative values so the demo
sits in a realistic regime rather than a convenient one.

Three sampling protocols, one estimator throughout (ridge with an unpenalised
intercept -- i.e. fitted on the centered system, as everywhere in this paper):

    A  fixed cardinality      every mask ablates exactly k
    B  two cardinalities      half at k1, half at k2
    C  Bernoulli(alpha)       each component kept independently

Predictions, which are simply the proposition's content:

    A  beta UNRECOVERABLE: q = f(n) is constant, so it centers to exactly zero and
       is exactly collinear with the intercept. The fitted value is whatever the
       regulariser picks -- for ridge, zero.
    B  beta RECOVERED. One extra cardinality buys one rank (Corollary 2).
    C  nominally identifiable, poorly conditioned: n fluctuates only
       O(sqrt(N p (1-p))), a relative spread O(1/sqrt(N)).

beta is swept over three magnitudes to show A fails at every one -- non-recovery
is structural, not a small-effect problem.

Efficiency note: the design does not depend on beta, so the Gram matrix, its
spectrum, and the Cholesky factor are computed ONCE per (protocol, form) and
reused across the sweep. Vc'Vc is further shared across the two forms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

# ---- representative scales for the synthetic regime --------------------------
N_COMPONENTS = 3904
W_SD = 0.109387          # per-example sd of the fitted attributions
B_MEAN = 0.2795          # intercept
NOISE_SD = 1.199388      # residual sd after the linear fit
ALPHA_TRAIN = 0.10
LAMBDA = 100.0

FORMS = {
    "linear": ("n", lambda n: n.astype(np.float64)),
    "quadratic": ("n(n-1)/2", lambda n: n.astype(np.float64) * (n.astype(np.float64) - 1.0) / 2.0),
}


# ----------------------------------------------------------------- sampling
def masks_fixed(m, N, k, rng):
    """Every row ablates exactly k components. Vectorised via argpartition."""
    V = np.ones((m, N), dtype=np.float64)
    idx = np.argpartition(rng.random((m, N)), k, axis=1)[:, :k]
    np.put_along_axis(V, idx, 0.0, axis=1)
    return V


def masks_two_cardinalities(m, N, k1, k2, rng):
    h = m // 2
    return np.vstack([masks_fixed(h, N, k1, rng), masks_fixed(m - h, N, k2, rng)])


def masks_bernoulli(m, N, p_keep, rng):
    return (rng.random((m, N)) < p_keep).astype(np.float64)


# ----------------------------------------------------------------- machinery
def ood_error(w_hat, beta_hat, b_hat, w, beta, ffn, N, k_list, rng, n_eval=400):
    """Transfer error to held-out cardinalities: mean |pred - truth|."""
    errs = []
    for k in k_list:
        V = masks_fixed(n_eval, N, k, rng)
        n = V.sum(axis=1)
        truth = B_MEAN + V @ w + beta * ffn(n)
        pred = b_hat + V @ w_hat + beta_hat * ffn(n)
        errs.append(float(np.abs(pred - truth).mean()))
    return float(np.mean(errs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=N_COMPONENTS)
    ap.add_argument("--m", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N, m = args.N, args.m
    rng = np.random.default_rng(args.seed)
    w = rng.normal(0.0, W_SD, size=N)

    k = int(round(N * ALPHA_TRAIN))
    k1, k2 = int(round(N * 0.05)), int(round(N * 0.20))
    p_keep = 1.0 - ALPHA_TRAIN
    ood_ks = [int(N * a) for a in (0.03, 0.075, 0.15)]

    print("=" * 104)
    print("SYNTHETIC IDENTIFIABILITY DEMONSTRATION -- no network, no ablation: the design alone")
    print("=" * 104)
    print(f"  N = {N}   m = {m}   lambda = {LAMBDA:g}   ridge with UNPENALISED intercept")
    print(f"  synthetic scales: sd(w) = {W_SD}, noise sd = {NOISE_SD}")
    print(f"  A: fixed k = {k}    B: k in {{{k1}, {k2}}}    C: Bernoulli(keep = {p_keep})")
    print(f"  OOD transfer evaluated at k = {ood_ks}")
    print()

    protocols = [
        ("A fixed cardinality", lambda: masks_fixed(m, N, k, rng)),
        ("B two cardinalities", lambda: masks_two_cardinalities(m, N, k1, k2, rng)),
        ("C Bernoulli", lambda: masks_bernoulli(m, N, p_keep, rng)),
        # Added after protocol B came back non-identifiable for the QUADRATIC
        # form: with only two distinct n, any f(n) lies in span{1, n}, and n is
        # itself V @ 1. Three cardinalities is the first design that can separate
        # a quadratic aggregate. Reported alongside B, not instead of it.
        ("B3 three cardinalities",
         lambda: np.vstack([masks_fixed(m // 3, N, k1, rng),
                            masks_fixed(m // 3, N, k, rng),
                            masks_fixed(m - 2 * (m // 3), N, k2, rng)])),
    ]

    rows = []
    for pname, make in protocols:
        t0 = time.time()
        V = make()
        n = V.sum(axis=1)
        Vc = V - V.mean(axis=0, keepdims=True)
        Gv = Vc.T @ Vc                      # shared across both forms
        print(f"  [{pname}] cardinalities: {len(np.unique(n))} distinct, "
              f"rel spread {n.std() / n.mean():.2e}   (Gram in {time.time() - t0:.0f}s)")

        for fname, (flabel, ffn) in FORMS.items():
            q = ffn(n)
            qc = q - q.mean()
            cross = Vc.T @ qc
            G = np.empty((N + 1, N + 1), dtype=np.float64)
            G[:N, :N] = Gv
            G[:N, N] = cross
            G[N, :N] = cross
            G[N, N] = float(qc @ qc)

            # THE identifiability diagnostic, and it is a projection, not a rank
            # count. Rank via an eigenvalue cutoff is unreliable here: the Gram's
            # largest eigenvalue is ~1e16, so a "numerically nonzero" smallest
            # eigenvalue can still be exact collinearity. Instead ask directly how
            # much of the aggregate column survives projecting out the design:
            #
            #     resid_frac = || q_c - P_{V_c} q_c || / || q_c ||
            #
            # resid_frac = 0  <=>  q is an exact linear combination of the design
            #                      columns, hence NOT separately identifiable.
            # This is stable, scale-free, and it is what the proposition asserts.
            evv, Uv = np.linalg.eigh(Gv)
            cut = max(evv.max(), 0.0) * len(evv) * np.finfo(float).eps
            keep = evv > cut
            proj = Uv[:, keep] @ ((Uv[:, keep].T @ cross) / evv[keep])
            resid_sq = float(qc @ qc) - float(cross @ proj)
            q_norm = float(np.sqrt(max(float(qc @ qc), 0.0)))
            resid_frac = float(np.sqrt(max(resid_sq, 0.0)) / q_norm) if q_norm > 0 else 0.0
            rank = int(keep.sum()) + (1 if resid_frac > 1e-6 else 0)
            cond = float(evv.max() / evv[keep].min()) if keep.any() else float("inf")

            cf = cho_factor(G + LAMBDA * np.eye(N + 1), lower=True)
            cov_nn = float(cho_solve(cf, np.eye(N + 1)[:, N])[N])
            se = float(np.sqrt(max(cov_nn, 0.0)) * NOISE_SD)

            scale = 1.0 / (N if fname == "linear" else N * N / 2.0)
            betas = [1e-2 * scale, 1.0 * scale, 1e2 * scale]

            for beta in betas:
                y = B_MEAN + V @ w + beta * q + rng.normal(0.0, NOISE_SD, size=m)
                yc = y - y.mean()
                rhs = np.concatenate([Vc.T @ yc, [qc @ yc]])
                coef = cho_solve(cf, rhs)
                w_hat, beta_hat = coef[:N], float(coef[N])
                b_hat = y.mean() - (np.concatenate([V.mean(axis=0), [q.mean()]]) @ coef)
                rel = abs(beta_hat - beta) / abs(beta)
                ood = ood_error(w_hat, beta_hat, b_hat, w, beta, ffn, N, ood_ks, rng)
                rows.append({
                    "form": fname, "f": flabel, "protocol": pname,
                    "beta_true": float(beta), "beta_hat": beta_hat,
                    "rel_error": float(rel), "se_beta": se,
                    "rank": rank, "full_rank": int(N + 1),
                    "rank_deficient": bool(rank < N + 1),
                    "cond": cond, "q_centered_norm": q_norm, "resid_frac": resid_frac,
                    # 1e-6, not 1e-8: with ||q_c|| ~ 1e8 a residual fraction of 1e-8 is
                    # machine epsilon, i.e. exact collinearity, not signal.
                    "identifiable": bool(resid_frac > 1e-6),
                    "ood_transfer_err": ood,
                    "n_distinct_cardinalities": int(len(np.unique(n))),
                    "cardinality_rel_spread": float(n.std() / n.mean()),
                })
        del V, Vc, Gv

    # ------------------------------------------------------------------ report
    for fname, (flabel, _) in FORMS.items():
        print()
        print("#" * 104)
        print(f"  f(n) = {flabel}      ({fname} case of Corollary 1)")
        print("#" * 104)
        print(f"  {'beta':>11} {'protocol':<24} {'beta_hat':>13} {'rel err':>11} "
              f"{'se(beta)':>10} {'resid_frac':>12} {'ident?':>8} {'cond':>10} {'OOD err':>9}")
        sub = [r for r in rows if r["form"] == fname]
        last_beta = None
        for r in sub:
            if last_beta is not None and r["beta_true"] != last_beta:
                print()
            last_beta = r["beta_true"]
            rk = f"{r['rank']}/{r['full_rank']}"
            flag = "*" if r["rank_deficient"] else " "
            print(f"  {r['beta_true']:>11.3e} {r['protocol']:<21} {r['beta_hat']:>13.3e} "
                  f"{r['rel_error']:>10.1%} {r['se_beta']:>10.2e} {rk:>10}{flag} "
                  f"{r['q_centered_norm']:>10.2e} {r['cond']:>10.2e} {r['ood_transfer_err']:>9.3f}")
        print("\n  * = rank deficient: the aggregate column adds no rank over [V | 1]")

    out = ROOT / "results" / "wpA_synthetic_identifiability.json"
    out.write_text(json.dumps({
        "config": {"N": N, "m": m, "lambda": LAMBDA, "k_fixed": k, "k_two": [k1, k2],
                   "p_keep": p_keep, "w_sd": W_SD, "noise_sd": NOISE_SD,
                   "seed": args.seed, "ood_ks": ood_ks},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
