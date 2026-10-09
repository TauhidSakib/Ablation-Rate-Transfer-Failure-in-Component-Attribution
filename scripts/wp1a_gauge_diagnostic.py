"""WP1a -- is the gauge degeneracy *live* in the fitted attributions, or latent?

CLAUDE-PHASE2 section 2.2 is right to insist this be measured rather than
assumed. The algebra says what to expect.

The solver works on the centered system ``(Xc'Xc + lambda I) w = Xc' yc``. Every
mask ablates exactly ``k`` components, so every row of ``X`` sums to ``N - k``
and centering gives ``Xc 1 = 0``. Left-multiply the normal equations by ``1'``::

    1' Xc'Xc w + lambda 1'w = 1' Xc' yc
    (Xc 1)' Xc w + lambda 1'w = (Xc 1)' yc
    0 + lambda 1'w = 0        =>        1'w = 0

**For any lambda > 0, ridge with an unpenalized intercept drives mean(w) to
zero.** The gauge fixes itself. So the interesting question is not "is kappa
nonzero" -- in floating point it always will be -- but:

    Is kappa large enough to CAUSE the observed out-of-distribution collapse?

That is the branch test implemented here. A gauge shift ``c`` moves the
prediction on an alpha-mask by exactly ``c * (N - k_alpha)``, so the ``c`` needed
to produce the level error we observe is computable, and can be compared against
the kappa actually present. A threshold on kappa alone decides nothing.

Predictions, stated before measuring:

  P1. kappa = mean(w) is numerically negligible relative to the coefficient scale.
  P2. Every mask is exactly k-hot (zero variance in row sums).
  P3. The intercept is unpenalized (solved analytically, never fitted).
  P4. The predicted mean is nearly *constant* across alpha while the true mean
      margin swings widely -- the OOD failure is a pure LEVEL error. A model
      pinned at mean(w)=0 cannot express "keeping more components raises the
      margin", because that is exactly the direction ridge zeroed out.
  P5. If that level error were a gauge artifact it would be linear in
      (N - k_alpha), so ONE constant per example would fix every alpha at once.
      If a single constant does *not* fix it, the aggregate response of the
      margin to ablation fraction is genuinely nonlinear and no reparameterization
      can express it.
  P6. Direct test of the alternative: is the true mean margin a saturating
      (concave) function of the number of kept components, rather than linear?
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
import torch  # noqa: E402

from naca.coar import CoarModel, predict_chunked  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.masks import MaskStore  # noqa: E402
from naca.metrics import r2_offset_corrected, r2_score  # noqa: E402
from naca.solver import fit_ridge, normal_equations  # noqa: E402


def atag(a: float) -> str:
    return f"a{a:g}".replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser(description="WP1a -- gauge diagnostic")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--lam", type=float, default=100.0, help="lambda W.npz was fit with")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = args.collect_dir or sorted(cfg.paths.runs.glob("*_collect"))[-1]
    print(f"[wp1a] collect dir: {d}")
    print(f"[wp1a] attributions were fitted at lambda = {args.lam:g}")
    print()

    coar = CoarModel.load(d / "W.npz")
    masks = MaskStore.load(d / "masks_train.npz")
    N, E = coar.N, coar.E
    k_train = masks.spec.k_ablated
    out: dict = {"N": N, "E": E, "k_train": k_train, "lambda_fitted": args.lam}

    # ---------------------------------------------------------------- P1
    kappa = coar.W.mean(axis=0)
    w_scale = float(np.abs(coar.W).mean())
    print("P1  kappa^(z) = mean_i w_i^(z)")
    print(f"      mean {kappa.mean():+.6e}   SD {kappa.std():.3e}   max|.| {np.abs(kappa).max():.3e}")
    print(f"      coefficient scale mean|w_i| = {w_scale:.3e}")
    print()
    out["kappa"] = {
        "mean": float(kappa.mean()),
        "sd": float(kappa.std()),
        "max_abs": float(np.abs(kappa).max()),
        "mean_abs_w": w_scale,
    }

    # ---------------------------------------------------------------- P2
    rs = masks.unpack(0, min(20000, len(masks))).sum(axis=1)
    khot = bool(rs.var() == 0)
    print(f"P2  mask row sums unique={np.unique(rs)[:4]}  var={rs.var():.3e}  -> k-hot: {khot}")
    print()
    out["masks_exactly_k_hot"] = khot

    # ---------------------------------------------------------------- P3
    Y = np.asarray(np.load(d / "Y_train.npy", mmap_mode="r"))
    big = fit_ridge(masks, Y[:, :16], 1e10, method="cholesky", device=cfg.device)
    b_err = float(np.abs(big.b - Y[:, :16].mean(axis=0)).max())
    print("P3  intercept penalization (lambda = 1e10)")
    print(f"      max|w| -> {np.abs(big.W).max():.3e}   max|b - mean(y)| -> {b_err:.3e}")
    print(f"      -> intercept unpenalized: {b_err < 1e-2}")
    print()
    out["intercept_unpenalized"] = bool(b_err < 1e-2)
    del big

    # ---------------------------------------------------------------- P4
    ne = normal_equations(masks, Y[:, :1], device=cfg.device, chunk=8192)
    ev = torch.linalg.eigvalsh(ne.Gxx)
    print("P4  centered Gram spectrum")
    print(f"      eig_min {float(ev[0]):+.3e}   eig_max {float(ev[-1]):.3e}")
    print(f"      #eigs below lambda={args.lam:g}: {int((ev < args.lam).sum())} of {len(ev)}")
    print()
    out["gram_spectrum"] = {
        "eig_min": float(ev[0]),
        "eig_max": float(ev[-1]),
        "n_below_lambda": int((ev < args.lam).sum()),
    }
    del ne, ev
    torch.cuda.empty_cache()

    print("P4b level error across alpha")
    hdr = f"    {'alpha':>7} {'k':>5} {'n_kept':>7} {'true mean':>10} {'pred mean':>10}"
    print(hdr + f" {'level err':>10} {'R2 raw':>9} {'R2 oc':>8}")
    rows = []
    for a in cfg.alpha_test_grid:
        t = atag(a)
        mp, yp = d / f"masks_test_{t}.npz", d / f"Y_test_{t}.npy"
        if not (mp.exists() and yp.exists()):
            continue
        tm = MaskStore.load(mp)
        ty = np.asarray(np.load(yp, mmap_mode="r"))
        pred = predict_chunked(coar.W, coar.b, tm, device=cfg.device)
        k_a = tm.spec.k_ablated
        lev = ty.mean(axis=0) - pred.mean(axis=0)
        rows.append(
            {
                "alpha": a,
                "k": k_a,
                "n_kept": N - k_a,
                "true_mean": float(ty.mean()),
                "pred_mean": float(pred.mean()),
                "level_err_mean": float(lev.mean()),
                "level_err": lev,
                "r2_raw": float(np.nanmean(r2_score(ty, pred))),
                "r2_oc": float(np.nanmean(r2_offset_corrected(ty, pred))),
            }
        )
        r = rows[-1]
        print(
            f"    {a:7.3f} {k_a:5d} {r['n_kept']:7d} {r['true_mean']:10.3f} "
            f"{r['pred_mean']:10.3f} {r['level_err_mean']:10.3f} "
            f"{r['r2_raw']:9.3f} {r['r2_oc']:8.3f}"
        )
    pred_spread = max(r["pred_mean"] for r in rows) - min(r["pred_mean"] for r in rows)
    true_spread = max(r["true_mean"] for r in rows) - min(r["true_mean"] for r in rows)
    print(f"    predicted mean spans {pred_spread:.3f}; true mean spans {true_spread:.3f}")
    print()
    out["pred_mean_spread"] = float(pred_spread)
    out["true_mean_spread"] = float(true_spread)

    # ---------------------------------------------------------------- P5
    nk = np.array([r["n_kept"] for r in rows], dtype=np.float64)
    L = np.stack([r["level_err"] for r in rows])
    c_star = (nk[:, None] * L).sum(axis=0) / (nk**2).sum()
    resid = L - np.outer(nk, c_star)
    frac = 1.0 - float((resid**2).sum()) / max(float((L**2).sum()), 1e-30)
    print("P5  can ONE gauge constant per example fix every alpha at once?")
    print(f"      c* mean {c_star.mean():+.3e}  sd {c_star.std():.3e}")
    print(f"      fraction of level error explained: {frac:.4f}")
    for i, r in enumerate(rows):
        print(
            f"        alpha={r['alpha']:6.3f}  before {r['level_err_mean']:+8.3f}"
            f"   after {resid[i].mean():+8.3f}"
        )
    print()
    out["single_constant"] = {
        "c_star_mean": float(c_star.mean()),
        "c_star_sd": float(c_star.std()),
        "fraction_of_level_error_explained": float(frac),
    }

    # ---------------------------------------------------------------- P6
    # Is the true mean margin linear in the number of kept components, or
    # saturating? A gauge constant can only produce a linear term, so curvature
    # here is structure no reparameterization can reach.
    tm_ = np.array([r["true_mean"] for r in rows])
    slopes = np.diff(tm_) / np.diff(nk)
    lin = np.polyfit(nk, tm_, 1)
    quad = np.polyfit(nk, tm_, 2)
    r2_lin_fit = 1 - np.sum((tm_ - np.polyval(lin, nk)) ** 2) / np.sum((tm_ - tm_.mean()) ** 2)
    r2_quad_fit = 1 - np.sum((tm_ - np.polyval(quad, nk)) ** 2) / np.sum((tm_ - tm_.mean()) ** 2)
    print("P6  is the true mean margin linear in n_kept, or saturating?")
    print("      local slopes (d margin / d n_kept), coarse -> fine ablation:")
    print("        " + "  ".join(f"{s:.4f}" for s in slopes))
    print(f"      straight-line fit R^2 = {r2_lin_fit:.4f}")
    print(f"      quadratic    fit R^2 = {r2_quad_fit:.4f}   (curvature {quad[0]:+.3e})")
    print()
    out["saturation"] = {
        "local_slopes": [float(s) for s in slopes],
        "linear_fit_r2": float(r2_lin_fit),
        "quadratic_fit_r2": float(r2_quad_fit),
        "quadratic_curvature": float(quad[0]),
    }

    # ------------------------------------------------------------- verdict
    need = [
        abs(r["level_err_mean"] / ((N - r["k"]) - (N - k_train)))
        for r in rows
        if r["k"] != k_train
    ]
    c_needed = float(np.mean(need))
    c_present = float(np.abs(kappa).max())
    shortfall = c_needed / max(c_present, 1e-30)
    print("VERDICT")
    print(f"      gauge constant REQUIRED to explain the level error : {c_needed:.3e}")
    print(f"      gauge constant actually PRESENT (max |kappa|)      : {c_present:.3e}")
    print(f"      shortfall                                          : {shortfall:.0f}x")
    print()
    branch = "A_live" if c_present >= 0.5 * c_needed else "B_not_the_cause"
    out["verdict"] = {
        "c_needed": c_needed,
        "c_present": c_present,
        "shortfall_factor": shortfall,
        "single_constant_explains": float(frac),
        "branch": branch,
    }
    if branch == "A_live":
        print("[wp1a] BRANCH A -- the gauge is live and big enough to matter.")
        print("       Proceed to the WP1b corruption test as specified.")
    else:
        print("[wp1a] BRANCH B -- ridge gauge-fixes to mean(w) ~ 0, and the residual")
        print(f"       kappa is {shortfall:.0f}x too small to cause the OOD collapse.")
        print("       What the collapse actually is: a LEVEL error. The predicted mean")
        print(f"       moves only {pred_spread:.2f} across the whole alpha grid while the true")
        print(f"       mean moves {true_spread:.2f}. And one gauge constant explains just")
        print(f"       {frac:.1%} of that, so it is not a reparameterization artifact --")
        print("       the margin's aggregate response to ablation fraction is nonlinear,")
        print("       which no choice of gauge can express.")
    (d / "wp1a_gauge_diagnostic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wp1a] wrote {d / 'wp1a_gauge_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
