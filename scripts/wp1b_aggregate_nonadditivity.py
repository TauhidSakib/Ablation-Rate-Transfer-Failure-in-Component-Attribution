"""WP1b (reframed for branch B) -- the non-additivity is aggregate, not pairwise.

WP1a established that the out-of-distribution collapse is not a gauge artifact
(the required constant is 23x larger than the one present) but a *level* error:
the additive model's predicted mean moves 0.10 across the whole alpha grid while
the truth moves 9.14, and the per-component marginal effect falls 6x as ablation
deepens.

That saturation is exactly what a **uniform pairwise interaction** produces. If
``A_ij = a`` for every pair, then

    sum_{i<j} a v_i v_j = a * n_kept (n_kept - 1) / 2

which is quadratic in the number of kept components -- the observed curvature.
So the hypothesis is that the network's non-additivity is real and large, but
*global*: spread thinly over all ~7.6M pairs rather than concentrated in a few.
A K-restricted low-rank model cannot see it, because it only ever models pairs
inside a 100-component pool.

There is a sharp corollary, and it is the reason COAR cannot learn this term:

    **At a single alpha, n_kept is CONSTANT, so the aggregate term is constant on
    the training data and therefore structurally unidentifiable.**

No estimator fitted at one ablation rate can recover it, however flexible. It
becomes identifiable only when alpha varies. This script tests all of that by
comparing three models on the same held-out grid:

  A  linear, fitted at alpha = 0.10 only            (what COAR does)
  B  linear, fitted on pooled alpha in {0.05, 0.10, 0.20}
  C  linear + one global term  q = n_kept(n_kept-1)/2, pooled

Model C adds exactly **one scalar parameter** over model B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from naca.coar import CoarModel  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.masks import MaskStore  # noqa: E402
from naca.metrics import pearson_r, r2_offset_corrected, r2_score  # noqa: E402


def atag(a: float) -> str:
    return f"a{a:g}".replace(".", "p")


def q_of(n_kept: float) -> float:
    """The aggregate term a uniform pairwise interaction would contribute."""
    return n_kept * (n_kept - 1.0) / 2.0


def fit_pooled(
    stores: list[MaskStore],
    Ys: list[np.ndarray],
    lam: float,
    *,
    with_q: bool,
    device: str = "cuda",
    chunk: int = 8192,
):
    """Ridge on pooled masks, optionally with one extra global column.

    Accumulates the centered normal equations across several mask stores. The
    intercept stays unpenalized (recovered analytically), and ``q`` -- when
    present -- is also left unpenalized: it is a single global parameter and
    shrinking it would defeat the purpose of the test.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    N = stores[0].N
    E = int(Ys[0].shape[1])
    P = N + (1 if with_q else 0)
    m_tot = sum(len(s) for s in stores)

    # first pass: means
    xsum = torch.zeros(P, dtype=torch.float64, device=dev)
    ysum = torch.zeros(E, dtype=torch.float64, device=dev)
    for s, Y in zip(stores, Ys):
        for lo in range(0, len(s), chunk):
            hi = min(lo + chunk, len(s))
            Xc = torch.from_numpy(s.unpack(lo, hi)).to(dev, torch.float64)
            xsum[:N] += Xc.sum(dim=0)
            if with_q:
                xsum[N] += q_of(float(N - s.spec.k_ablated)) * (hi - lo)
            ysum += torch.from_numpy(np.ascontiguousarray(Y[lo:hi])).to(dev, torch.float64).sum(0)
            del Xc
    xbar = xsum / m_tot
    ybar = ysum / m_tot

    # second pass: centered Gram and rhs
    G = torch.zeros((P, P), dtype=torch.float64, device=dev)
    rhs = torch.zeros((P, E), dtype=torch.float64, device=dev)
    for s, Y in zip(stores, Ys):
        qv = q_of(float(N - s.spec.k_ablated))
        for lo in range(0, len(s), chunk):
            hi = min(lo + chunk, len(s))
            Xc = torch.from_numpy(s.unpack(lo, hi)).to(dev, torch.float64)
            if with_q:
                col = torch.full((hi - lo, 1), qv, dtype=torch.float64, device=dev)
                Xc = torch.cat([Xc, col], dim=1)
            Xc -= xbar.unsqueeze(0)
            Yc = torch.from_numpy(np.ascontiguousarray(Y[lo:hi])).to(dev, torch.float64) - ybar
            G += Xc.T @ Xc
            rhs += Xc.T @ Yc
            del Xc, Yc

    A = G.clone()
    idx = torch.arange(N, device=dev)
    A[idx, idx] += lam  # q column deliberately unpenalized
    L = torch.linalg.cholesky(A)
    W = torch.cholesky_solve(rhs, L)
    b = ybar - xbar @ W
    out = {
        "W": W[:N].to(torch.float32).cpu().numpy(),
        "b": b.to(torch.float32).cpu().numpy(),
        "a_q": float(W[N].mean()) if with_q else 0.0,
        "a_q_vec": W[N].to(torch.float32).cpu().numpy() if with_q else None,
        "xbar_q": float(xbar[N]) if with_q else 0.0,
    }
    del G, rhs, A, L, W
    torch.cuda.empty_cache()
    return out


def predict(model, store: MaskStore, *, device="cuda", chunk=8192, with_q=False):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    N = store.N
    Wt = torch.from_numpy(model["W"]).to(dev, torch.float32)
    bt = torch.from_numpy(model["b"]).to(dev, torch.float32)
    out = np.empty((len(store), model["W"].shape[1]), dtype=np.float32)
    qv = q_of(float(N - store.spec.k_ablated))
    aq = (
        torch.from_numpy(model["a_q_vec"]).to(dev, torch.float32)
        if (with_q and model["a_q_vec"] is not None)
        else None
    )
    for lo in range(0, len(store), chunk):
        hi = min(lo + chunk, len(store))
        Xc = torch.from_numpy(store.unpack(lo, hi)).to(dev, torch.float32)
        p = Xc @ Wt + bt.unsqueeze(0)
        if aq is not None:
            p = p + qv * aq.unsqueeze(0)
        out[lo:hi] = p.cpu().numpy()
        del Xc, p
    return out


def score(ty, pred):
    return {
        "rho": float(np.nanmean(pearson_r(ty, pred))),
        "r2": float(np.nanmean(r2_score(ty, pred))),
        "r2_oc": float(np.nanmean(r2_offset_corrected(ty, pred))),
        "pred_mean": float(pred.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="WP1b -- aggregate non-additivity")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--lam", type=float, default=100.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = args.collect_dir or sorted(cfg.paths.runs.glob("*_collect"))[-1]
    print(f"[wp1b] collect dir: {d}\n")

    # ---- assemble the pooled training set ---------------------------------
    stores, Ys, alphas = [], [], []
    stores.append(MaskStore.load(d / "masks_train.npz"))
    Ys.append(np.asarray(np.load(d / "Y_train.npy", mmap_mode="r")))
    alphas.append(cfg.alpha_train)
    for mp in sorted(d.glob("masks_train_a*.npz")):
        tag = mp.stem.replace("masks_train_", "")
        yp = d / f"Y_train_{tag}.npy"
        if yp.exists():
            s = MaskStore.load(mp)
            stores.append(s)
            Ys.append(np.asarray(np.load(yp, mmap_mode="r")))
            alphas.append(s.spec.alpha)
    print(f"[wp1b] pooled alpha_train values: {alphas}")
    print(f"[wp1b] pooled m = {sum(len(s) for s in stores)}")
    N = stores[0].N
    for s in stores:
        print(f"         alpha={s.spec.alpha:<6.3f} m={len(s):>6d} n_kept={N - s.spec.k_ablated}")
    print()

    # ---- three models -------------------------------------------------------
    single = CoarModel.load(d / "W.npz")
    model_a = {"W": single.W, "b": single.b, "a_q": 0.0, "a_q_vec": None}
    print("[wp1b] fitting B: linear, pooled multi-alpha")
    model_b = fit_pooled(stores, Ys, args.lam, with_q=False, device=cfg.device)
    print("[wp1b] fitting C: linear + one global aggregate term, pooled multi-alpha")
    model_c = fit_pooled(stores, Ys, args.lam, with_q=True, device=cfg.device)

    kap_a = float(model_a["W"].mean(axis=0).mean())
    kap_b = float(model_b["W"].mean(axis=0).mean())
    print()
    print("  mean(w) -- the direction single-alpha ridge is forced to zero:")
    print(f"      A single-alpha : {kap_a:+.6e}")
    print(f"      B pooled       : {kap_b:+.6e}   (free once alpha varies)")
    print(f"  global term coefficient a (model C): {model_c['a_q']:+.6e}")
    print()

    # ---- evaluate on the held-out grid --------------------------------------
    rows = []
    print(f"    {'alpha':>7} | {'A rho':>7} {'A R2oc':>8} | {'B rho':>7} {'B R2oc':>8} "
          f"| {'C rho':>7} {'C R2oc':>8} | {'true mean':>9}")
    for a in cfg.alpha_test_grid:
        t = atag(a)
        mp, yp = d / f"masks_test_{t}.npz", d / f"Y_test_{t}.npy"
        if not (mp.exists() and yp.exists()):
            continue
        tm = MaskStore.load(mp)
        ty = np.asarray(np.load(yp, mmap_mode="r"))
        sa = score(ty, predict(model_a, tm, device=cfg.device))
        sb = score(ty, predict(model_b, tm, device=cfg.device))
        sc = score(ty, predict(model_c, tm, device=cfg.device, with_q=True))
        rows.append(
            {
                "alpha_test": a,
                "true_mean": float(ty.mean()),
                **{f"A_{k}": v for k, v in sa.items()},
                **{f"B_{k}": v for k, v in sb.items()},
                **{f"C_{k}": v for k, v in sc.items()},
            }
        )
        print(
            f"    {a:7.3f} | {sa['rho']:7.4f} {sa['r2_oc']:8.4f} | "
            f"{sb['rho']:7.4f} {sb['r2_oc']:8.4f} | "
            f"{sc['rho']:7.4f} {sc['r2_oc']:8.4f} | {ty.mean():9.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(d / "wp1b_aggregate.csv", index=False)

    print()
    print("  predicted mean vs truth (does the model track the LEVEL?)")
    print(f"    {'alpha':>7} {'true':>9} {'A':>9} {'B':>9} {'C':>9}")
    for r in rows:
        print(
            f"    {r['alpha_test']:7.3f} {r['true_mean']:9.3f} {r['A_pred_mean']:9.3f} "
            f"{r['B_pred_mean']:9.3f} {r['C_pred_mean']:9.3f}"
        )
    spread = {
        k: max(r[f"{k}_pred_mean"] for r in rows) - min(r[f"{k}_pred_mean"] for r in rows)
        for k in "ABC"
    }
    true_spread = max(r["true_mean"] for r in rows) - min(r["true_mean"] for r in rows)
    print(f"    spread: true {true_spread:.3f} | A {spread['A']:.3f} "
          f"| B {spread['B']:.3f} | C {spread['C']:.3f}")

    # ---- raw R2 on the OOD cells, the headline comparison -------------------
    ood = [r for r in rows if r["alpha_test"] != cfg.alpha_train]
    summary = {
        "alphas_pooled": alphas,
        "kappa_single_alpha": kap_a,
        "kappa_pooled": kap_b,
        "global_term_coefficient": model_c["a_q"],
        "mean_ood_r2_raw": {k: float(np.mean([r[f"{k}_r2"] for r in ood])) for k in "ABC"},
        "mean_ood_r2_oc": {k: float(np.mean([r[f"{k}_r2_oc"] for r in ood])) for k in "ABC"},
        "mean_ood_rho": {k: float(np.mean([r[f"{k}_rho"] for r in ood])) for k in "ABC"},
        "pred_mean_spread": spread,
        "true_mean_spread": float(true_spread),
        "rows": rows,
    }
    print()
    print("  mean over OOD alphas:")
    for metric in ("r2_raw", "r2_oc", "rho"):
        key = f"mean_ood_{metric}"
        v = summary[key]
        print(f"    {metric:>7}:  A {v['A']:+.4f}   B {v['B']:+.4f}   C {v['C']:+.4f}")

    (d / "wp1b_aggregate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[wp1b] wrote {d / 'wp1b_aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
