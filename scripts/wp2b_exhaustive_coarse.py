"""WP2b -- the exhaustive counterfactual surface at coarse granularity.

At coarse granularity the additivity question can be answered with **no estimator
at all**. With G units there are only 2^G distinct ablation subsets; at the block
level of a ResNet-18 that is 2^9 = 512 masks, about a minute of compute. Measuring
all of them gives the complete surface and its exact ANOVA (Moebius) decomposition

    f(m) = sum_{S subset of units} beta_S * prod_{i in S} m_i    (m_i = 1 if ABLATED)

where |S| = 1 are individual effects, |S| = 2 exact pairwise interactions, and so
on up to |S| = G. Every objection that applies to a fitted low-rank estimator is
removed at once: no sampling error, no candidate pool, no low-rank assumption, and
no identifiability problem -- enumerating *every* cardinality is precisely what
breaks the fixed-cardinality degeneracy proved in wp1c.

Four experiments:

  1  FULL SPECTRUM. Variance explained by truncating at each order 1..G, and the
     coefficient mass at each order. The definitive answer to "how non-additive is
     this network", computable here and essentially nowhere else.

  2  IS ORDER-2 UNIFORM? If the aggregate term found at filter granularity is a
     genuine uniform pairwise effect with coefficient ``a``, then a block pair
     (i,j) spanning n_i * n_j filter pairs must carry beta_ij ~ a * n_i * n_j.
     A falsifiable prediction, tested against model C's fitted ``a`` by regressing
     beta_ij on n_i*n_j through the origin.

  3  THE PROTOCOL ON TRIAL. Fit additive and full-pairwise models on
     *fixed-cardinality* subsets -- exactly what COAR does -- and score them
     against the exhaustive truth. A positive control at the level of the
     **method**, not the estimator: the right answer is known, so what the
     sampling protocol costs can be measured rather than argued.

  4  IS THE SATURATION A GRANULARITY ARTIFACT? The ~6x decline in marginal effect
     measured at filter granularity, recomputed exactly here. Surviving means it is
     a property of the network; vanishing means it is specific to filter
     granularity. Either answer is sharp.

Also collects all singletons and pairs at the layer level (1 + 17 + 136 = 154
masks), giving exact pairwise interactions there without enumerating 2^17.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from naca.ablation import Ablator  # noqa: E402
from naca.collect import _forward_margins, preload_eval  # noqa: E402
from naca.components import build_component_index  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.datasets import load_dataset, num_classes_for  # noqa: E402
from naca.granularity import build_unit_map  # noqa: E402
from naca.models import build_model, load_checkpoint  # noqa: E402


# ------------------------------------------------------------------ masks
def enumerate_all(G: int) -> np.ndarray:
    """All 2^G ablation indicators, shape (2^G, G). Row r has bit i set iff unit i ablated."""
    idx = np.arange(1 << G, dtype=np.uint32)
    return ((idx[:, None] >> np.arange(G)[None, :]) & 1).astype(np.uint8)


def enumerate_upto_pairs(G: int) -> np.ndarray:
    rows = [np.zeros(G, dtype=np.uint8)]
    for i in range(G):
        r = np.zeros(G, dtype=np.uint8)
        r[i] = 1
        rows.append(r)
    for i, j in itertools.combinations(range(G), 2):
        r = np.zeros(G, dtype=np.uint8)
        r[i] = r[j] = 1
        rows.append(r)
    return np.stack(rows)


@torch.inference_mode()
def collect_unit_masks(model, index, umap, M, x_eval, y_eval, sub_batch, log_every=128, amp_dtype=torch.float32):
    ablator = Ablator(model, index)
    buf = torch.empty(int(x_eval.shape[0]), dtype=torch.float32, device=x_eval.device)
    Y = np.empty((M.shape[0], int(x_eval.shape[0])), dtype=np.float32)
    t0 = time.time()
    try:
        for t in range(M.shape[0]):
            fm = (1 - M[t]).astype(np.float32)[umap.unit_of]
            ablator.apply(torch.from_numpy(fm).to(x_eval.device))
            _forward_margins(model, x_eval, y_eval, buf, sub_batch, amp_dtype)
            Y[t] = buf.cpu().numpy()
            if log_every and (t + 1) % log_every == 0:
                rate = (t + 1) / max(time.time() - t0, 1e-9)
                print(f"      {t + 1}/{M.shape[0]}  {rate:.1f} masks/s")
    finally:
        ablator.restore()
    return Y


def mobius(Y: np.ndarray, G: int) -> np.ndarray:
    """Fast zeta/Moebius transform. Checked against brute force in tests/test_granularity.py."""
    beta = Y.astype(np.float64).copy()
    ar = np.arange(beta.shape[0])
    for i in range(G):
        bit = 1 << i
        idx = ar[(ar & bit) != 0]
        beta[idx] -= beta[idx - bit]
    return beta


def walsh(Y: np.ndarray, G: int) -> np.ndarray:
    """Orthogonal functional-ANOVA coefficients over the uniform measure on masks.

    The Moebius basis ``prod_{i in S} m_i`` is NOT orthogonal, so truncating a
    Moebius expansion is not the best order-<=k model and its R^2 is not even
    monotone in k -- measured directly: -29.8, -156.5, -211.7 for k = 0, 1, 2.
    Moebius coefficients are the right object for reading off an individual
    interaction; they are the wrong object for apportioning variance.

    The Walsh basis ``chi_S(m) = prod_{i in S} (-1)^{m_i}`` IS orthonormal under
    the uniform measure over all 2^G masks, so

        Var(f) = sum_{S != empty} fhat(S)^2

    exactly, the share of variance at order k is well defined, shares sum to 1,
    and the best order-<=k approximation in L2 *is* the truncation. That makes
    cumulative share monotone by construction.
    """
    F = Y.astype(np.float64).copy()
    n = F.shape[0]
    h = 1
    while h < n:
        for lo in range(0, n, h * 2):
            a = F[lo : lo + h].copy()
            b = F[lo + h : lo + 2 * h].copy()
            F[lo : lo + h] = a + b
            F[lo + h : lo + 2 * h] = a - b
        h *= 2
    return F / n


def reconstruct_by_order(beta, M, order, G):
    """Yield ``(kmax, Yhat)``, accumulating orders incrementally."""
    Yhat = np.zeros_like(beta)
    for kmax in range(G + 1):
        for s in np.nonzero(order == kmax)[0]:
            bits = M[s].astype(bool)
            if not bits.any():
                Yhat += beta[s][None, :]
            else:
                Yhat[np.all(M[:, bits] == 1, axis=1)] += beta[s][None, :]
        yield kmax, Yhat


def fit_ls(design: np.ndarray, Y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    A = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(A, design.T @ Y)


def design_matrix(M: np.ndarray, order: int) -> np.ndarray:
    """``[1 | m_i]`` for order 1; adds every ``m_i m_j`` for order 2."""
    cols = [np.ones((M.shape[0], 1)), M.astype(np.float64)]
    if order >= 2:
        G = M.shape[1]
        cols.append(
            np.stack([M[:, i] * M[:, j] for i, j in itertools.combinations(range(G), 2)], axis=1)
        )
    return np.concatenate(cols, axis=1)


def r2_of(Y, Yhat):
    ss = ((Y - Yhat) ** 2).mean(axis=0)
    return float((1.0 - ss / np.maximum(Y.var(axis=0), 1e-12)).mean())


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="WP2b -- exhaustive coarse surface")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--skip-layer", action="store_true")
    ap.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="forward precision. fp32 by default: the exhaustive surface is small "
             "(512 masks at 9 units, 4,096 at 12) so precision is free here, "
        "so precision costs nothing here, and it keeps the cleanest experiment in "
        "the project free of the bf16 noise floor that limits the filter-level "
        "NAF measurement.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = args.collect_dir or sorted(cfg.paths.runs.glob("*_collect"))[-1]
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    amp = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    print(f"[wp2b] forward precision: {args.dtype}")
    print(f"[wp2b] collect dir: {d}\n")

    model = build_model(cfg.arch, num_classes=num_classes_for(cfg.dataset), device=device)
    load_checkpoint(model, cfg.paths.models / f"{cfg.setup}_{cfg.arch}.pt")
    model.eval()
    index = build_component_index(model, arch=cfg.arch, expected_n=cfg.n_components, verbose=False)
    test_ds = load_dataset(cfg.dataset, cfg.paths.data, train=False, device=device, download=False)
    eval_ds = test_ds.subset(np.load(d / "eval_indices.npy"))
    x_eval, y_eval = preload_eval(eval_ds.images, eval_ds.labels, eval_ds.mean, eval_ds.std)
    clean = np.load(d / "clean_margins.npy")
    out: dict = {"forward_dtype": args.dtype}

    umap = build_unit_map(index, "block")
    G = umap.n_units
    sizes = umap.sizes()
    print(f"[wp2b] BLOCK level: {G} units -> {1 << G} masks (EXHAUSTIVE)")
    for nm, sz in zip(umap.names, sizes):
        print(f"       {nm:<10} {sz:>5} filters")
    print()

    M = enumerate_all(G)
    path = d / f"wp2b_block_Y_{args.dtype}.npy"
    if path.exists():
        Y = np.load(path)
        print("       reusing cached surface")
    else:
        Y = collect_unit_masks(model, index, umap, M, x_eval, y_eval, cfg.eval_sub_batch, amp_dtype=amp)
        np.save(path, Y)
    np.save(d / "wp2b_block_M.npy", M)
    assert np.isfinite(Y).all(), "non-finite margins in the exhaustive surface"
    order = M.sum(axis=1)
    print(f"       surface {Y.shape}, margins [{Y.min():.2f}, {Y.max():.2f}]")
    print(f"       sanity: |empty-mask margin - clean| = {np.abs(Y[0] - clean).max():.4f}")
    print()

    beta = mobius(Y, G)

    # ========================================== 1. FULL INTERACTION SPECTRUM
    print("=" * 74)
    print("1. FULL SPECTRUM -- orthogonal variance decomposition by interaction order")
    print("=" * 74)
    fhat = walsh(Y, G)
    popcount = np.array([bin(i).count("1") for i in range(1 << G)])
    # Var(f) = sum over non-empty S of fhat(S)^2, exactly (Parseval)
    energy = (fhat**2).sum(axis=0) - fhat[0] ** 2
    var_direct = Y.var(axis=0)
    print(f"    Parseval check: max relative error between sum fhat^2 and Var(f) = "
          f"{np.abs(energy - var_direct).max() / max(var_direct.max(), 1e-12):.3e}")
    print()
    rows, cum = [], 0.0
    print(f"    {'order':>6} {'#terms':>8} {'share of variance':>19} {'cumulative R^2':>16}")
    for k in range(G + 1):
        sel = popcount == k
        if k == 0:
            continue
        share = float((fhat[sel] ** 2).sum(axis=0).sum() / max(energy.sum(), 1e-30))
        cum += share
        rows.append({"order": k, "n_terms": int(sel.sum()),
                     "variance_share": share, "cumulative_r2": cum})
        print(f"    {k:>6} {int(sel.sum()):>8} {share:>19.6f} {cum:>16.6f}")
    lin = rows[0]["cumulative_r2"]
    quad = rows[1]["cumulative_r2"]
    naf_exact = (quad - lin) / max(1.0 - lin, 1e-12)
    print()
    print(f"    additive (order 1 only)          : R^2 = {lin:.6f}")
    print(f"    + exact pairwise (orders 1-2)    : R^2 = {quad:.6f}")
    print(f"    EXACT NAF at block granularity   : {naf_exact:+.6f}")
    print(f"    left to orders >= 3              : {1 - quad:.6f}")

    # Moebius coefficients remain the right object for individual interactions
    tot_abs = np.abs(beta).sum()
    print()
    print("    (Moebius view -- exact interaction magnitudes, not variance shares)")
    print(f"    {'order':>6} {'#terms':>8} {'mean|beta|':>13}")
    share_by_order = {}
    for k in range(G + 1):
        sel = order == k
        if not sel.any():
            continue
        share_by_order[int(k)] = float(np.abs(beta[sel]).sum() / max(tot_abs, 1e-30))
        print(f"    {k:>6} {int(sel.sum()):>8} {np.abs(beta[sel]).mean():>13.5f}")
    pd.DataFrame(rows).to_csv(d / "wp2b_block_orders.csv", index=False)

    # ================================ 1b. THE SEVERITY AXIS (pre-registered)
    print()
    print("=" * 74)
    print("1b. NON-ADDITIVITY vs SEVERITY  (pre-registered test, PAPER.md 5.5)")
    print("=" * 74)
    print("    Order and severity are different axes. This stratifies model quality by")
    print("    mask cardinality and reports the measured damage of each stratum, so the")
    print("    claim 'apparent non-additivity tracks ablation severity' can be falsified")
    print("    rather than asserted.")
    print()
    D1 = design_matrix(M, 1)
    D2 = design_matrix(M, 2)
    c1 = fit_ls(D1, Y.astype(np.float64))
    c2 = fit_ls(D2, Y.astype(np.float64))
    P1, P2 = D1 @ c1, D2 @ c2
    clean_mean = float(Y[0].mean())
    print(f"    {'k ablated':>10} {'#masks':>7} {'severity':>10} {'add R^2':>9} "
          f"{'pair R^2':>9} {'add resid':>10}")
    sev_rows = []
    for k in range(1, G + 1):
        sel = order == k
        Ys = Y[sel]
        if Ys.shape[0] < 2:
            continue
        var_s = Ys.var(axis=0)
        ok = var_s > 1e-9
        r2a = float((1 - ((Ys - P1[sel]) ** 2).mean(axis=0)[ok] / var_s[ok]).mean()) if ok.any() else float("nan")
        r2b = float((1 - ((Ys - P2[sel]) ** 2).mean(axis=0)[ok] / var_s[ok]).mean()) if ok.any() else float("nan")
        severity = float(np.abs(Ys - Y[0][None, :]).mean())
        resid = float(np.sqrt(((Ys - P1[sel]) ** 2).mean()))
        sev_rows.append({"k": int(k), "n_masks": int(sel.sum()), "severity": severity,
                         "additive_r2": r2a, "pairwise_r2": r2b, "additive_rmse": resid})
        print(f"    {k:>10} {int(sel.sum()):>7} {severity:>10.4f} {r2a:>9.4f} "
              f"{r2b:>9.4f} {resid:>10.4f}")
    pd.DataFrame(sev_rows).to_csv(d / "wp2b_severity_axis.csv", index=False)

    sv = np.array([r["severity"] for r in sev_rows])
    qa = np.array([r["additive_r2"] for r in sev_rows])
    fin = np.isfinite(sv) & np.isfinite(qa)
    if fin.sum() >= 3:
        corr = float(np.corrcoef(sv[fin], qa[fin])[0, 1])
        print()
        print(f"    correlation(severity, additive R^2) = {corr:+.4f}")
        if corr < -0.5:
            v = ("SUPPORTED -- additive quality degrades as severity rises, so the "
                 "non-additivity tracks damage rather than component type")
        elif abs(corr) <= 0.5:
            v = ("NOT SUPPORTED -- additive quality is roughly flat in severity; the "
                 "non-additivity is a property of the component type, not the damage")
        else:
            v = "INVERTED -- additive quality improves with severity; the account is wrong"
        print(f"    -> pre-registered claim {v}")
        out["severity_axis"] = {"rows": sev_rows, "corr_severity_vs_additive_r2": corr,
                                "verdict": v, "clean_mean_margin": clean_mean}

    # ========================================== 2. IS ORDER-2 UNIFORM?
    print()
    print("=" * 74)
    print("2. IS THE ORDER-2 STRUCTURE UNIFORM? (the aggregate term, seen from above)")
    print("=" * 74)
    pairs = list(itertools.combinations(range(G), 2))
    b2 = np.array([beta[(1 << i) | (1 << j)].mean() for i, j in pairs])
    npair = np.array([float(sizes[i]) * float(sizes[j]) for i, j in pairs])
    cv = float(np.std(b2) / max(abs(np.mean(b2)), 1e-30))
    print(f"    {len(pairs)} block pairs; mean beta_ij = {b2.mean():+.5f}, sd = {b2.std():.5f}")
    print(f"    coefficient of variation = {cv:.3f}   (0 would be perfectly uniform)")
    a_hat = float((npair @ b2) / (npair @ npair))
    r2_pred = 1.0 - float(((b2 - a_hat * npair) ** 2).sum()) / max(
        float(((b2 - b2.mean()) ** 2).sum()), 1e-30)
    print()
    print("    prediction: a uniform FILTER-level pairwise coefficient 'a' implies")
    print("                beta_ij = a * n_i * n_j  for blocks of n_i, n_j filters")
    print(f"      best-fit a from the exhaustive surface : {a_hat:+.6e}")
    a_model_c = None
    wp1b = d / "wp1b_aggregate.json"
    if wp1b.exists():
        a_model_c = json.loads(wp1b.read_text(encoding="utf-8")).get("global_term_coefficient")
        if a_model_c:
            print(f"      a fitted by model C at FILTER level    : {a_model_c:+.6e}")
            print(f"      ratio (exhaustive / model C)          : {a_hat / a_model_c:.3f}")
    print(f"      R^2 of  beta_ij ~ a * n_i * n_j        : {r2_pred:.4f}")
    verdict = ("CONFIRMED -- order-2 structure is the uniform aggregate term"
               if r2_pred > 0.8 else
               "REFUTED -- order-2 structure is NOT uniform; it carries real per-pair detail")
    print(f"      -> {verdict}")
    out["order2_uniformity"] = {
        "n_pairs": len(pairs), "mean_beta": float(b2.mean()), "sd_beta": float(b2.std()),
        "cv": cv, "a_from_exhaustive": a_hat, "a_from_model_c": a_model_c,
        "r2_of_uniform_prediction": float(r2_pred), "verdict": verdict,
    }

    # ========================================== 3. THE PROTOCOL ON TRIAL
    print()
    print("=" * 74)
    print("3. THE PROTOCOL ON TRIAL -- fit at fixed cardinality, score against truth")
    print("=" * 74)
    print("    Each model is fitted ONLY on masks of one cardinality (what COAR does),")
    print(f"    then scored on the entire {1 << G}-mask surface and on its own stratum.")
    print()
    print(f"    {'fit on k':>9} {'#masks':>7} | {'additive full':>14} {'pairwise full':>14}"
          f" | {'additive @k':>12} {'pairwise @k':>12}")
    prot = []
    for k in range(1, G):
        sel = np.nonzero(order == k)[0]
        if len(sel) < 4:
            continue
        row = {"k": int(k), "n_masks": int(len(sel))}
        for name, o in (("additive", 1), ("pairwise", 2)):
            D = design_matrix(M[sel], o)
            if D.shape[0] <= D.shape[1]:
                row[f"{name}_full"] = float("nan")
                row[f"{name}_at_k"] = float("nan")
                continue
            coef = fit_ls(D, Y[sel].astype(np.float64))
            row[f"{name}_full"] = r2_of(Y, design_matrix(M, o) @ coef)
            row[f"{name}_at_k"] = r2_of(Y[sel], D @ coef)
        prot.append(row)
        print(f"    {k:>9} {len(sel):>7} | {row['additive_full']:>14.4f} "
              f"{row['pairwise_full']:>14.4f} | {row['additive_at_k']:>12.4f} "
              f"{row['pairwise_at_k']:>12.4f}")
    print()
    for name, o in (("additive", 1), ("pairwise", 2)):
        D = design_matrix(M, o)
        coef = fit_ls(D, Y.astype(np.float64))
        print(f"    ORACLE {name:>9} fitted on all {len(M)} masks: R^2 = {r2_of(Y, D @ coef):.4f}")
    pd.DataFrame(prot).to_csv(d / "wp2b_protocol_trial.csv", index=False)
    out["protocol_trial"] = prot

    # ========================================== 4. SATURATION AT BLOCK LEVEL
    print()
    print("=" * 74)
    print("4. IS THE SATURATION A GRANULARITY ARTIFACT?")
    print("=" * 74)
    mean_by_k = np.array([Y[order == k].mean() for k in range(G + 1)])
    units_kept = np.arange(G, -1, -1).astype(float)
    slopes = np.diff(mean_by_k) / -1.0  # per additional ablated unit
    print(f"    {'units ablated':>14} {'mean margin':>13}")
    for k in range(G + 1):
        print(f"    {k:>14} {mean_by_k[k]:>13.4f}")
    print()
    print("    marginal effect per additional ablated unit, shallow -> deep:")
    print("      " + "  ".join(f"{s:+.4f}" for s in slopes))
    ratio = abs(slopes[0]) / max(abs(slopes[-1]), 1e-12)
    print(f"    first/last magnitude ratio at BLOCK level : {ratio:.2f}x")
    print("    (filter level measured 6.0x -- FINDING_aggregate_nonadditivity.md)")
    r2l = 1 - np.sum((mean_by_k - np.polyval(np.polyfit(units_kept, mean_by_k, 1), units_kept)) ** 2) \
        / np.sum((mean_by_k - mean_by_k.mean()) ** 2)
    r2q = 1 - np.sum((mean_by_k - np.polyval(np.polyfit(units_kept, mean_by_k, 2), units_kept)) ** 2) \
        / np.sum((mean_by_k - mean_by_k.mean()) ** 2)
    print(f"    straight-line fit R^2 = {r2l:.4f}   quadratic fit R^2 = {r2q:.4f}")
    print(f"    -> saturation {'SURVIVES' if ratio > 2.0 else 'DOES NOT survive'} at block granularity")
    out["saturation_block"] = {
        "mean_margin_by_units_ablated": mean_by_k.tolist(),
        "slopes": slopes.tolist(), "first_last_ratio": float(ratio),
        "linear_fit_r2": float(r2l), "quadratic_fit_r2": float(r2q),
        "filter_level_ratio": 6.0,
    }
    out["block"] = {
        "n_units": G, "unit_names": umap.names, "unit_sizes": sizes.tolist(),
        "n_masks": int(M.shape[0]), "r2_by_max_order": rows,
        "exact_naf": float(naf_exact), "abs_beta_share_by_order": share_by_order,
    }

    # =============================================================== LAYER
    if not args.skip_layer:
        umapL = build_unit_map(index, "layer")
        GL = umapL.n_units
        ML = enumerate_upto_pairs(GL)
        print()
        print(f"[wp2b] LAYER level: {GL} units, {ML.shape[0]} masks "
              f"(empty + singletons + pairs -> exact pairwise without 2^{GL})")
        pathL = d / f"wp2b_layer_Y_{args.dtype}.npy"
        if pathL.exists():
            YL = np.load(pathL)
            print("       reusing cached surface")
        else:
            YL = collect_unit_masks(model, index, umapL, ML, x_eval, y_eval,
                                    cfg.eval_sub_batch, log_every=50, amp_dtype=amp)
            np.save(pathL, YL)
        f0 = YL[0]
        singles = {int(np.nonzero(ML[t])[0][0]): YL[t] for t in range(1, 1 + GL)}
        n_eval = int(x_eval.shape[0])
        inter = np.zeros((GL, GL, n_eval), dtype=np.float64)
        for t in range(1 + GL, ML.shape[0]):
            i, j = np.nonzero(ML[t])[0]
            inter[i, j] = inter[j, i] = YL[t] - singles[i] - singles[j] + f0
        iu = np.triu_indices(GL, 1)
        pair_mag = float(np.abs(inter[iu[0], iu[1]]).mean())
        main_mag = float(np.abs(np.stack([singles[i] - f0 for i in range(GL)])).mean())
        szL = umapL.sizes()
        npairL = np.array([float(szL[i]) * float(szL[j]) for i, j in zip(*iu)])
        b2L = np.array([inter[i, j].mean() for i, j in zip(*iu)])
        aL = float((npairL @ b2L) / (npairL @ npairL))
        r2L = 1.0 - float(((b2L - aL * npairL) ** 2).sum()) / max(
            float(((b2L - b2L.mean()) ** 2).sum()), 1e-30)
        print(f"       mean |main effect|       : {main_mag:.5f}")
        print(f"       mean |pairwise interact| : {pair_mag:.5f}")
        print(f"       interaction / main       : {pair_mag / max(main_mag, 1e-12):.4f}")
        print(f"       uniform prediction R^2   : {r2L:.4f}   (a = {aL:+.6e})")
        out["layer"] = {
            "n_units": GL, "unit_names": umapL.names,
            "mean_abs_main": main_mag, "mean_abs_pairwise": pair_mag,
            "ratio": pair_mag / max(main_mag, 1e-12),
            "a_from_exhaustive": aL, "r2_of_uniform_prediction": float(r2L),
        }
        np.save(d / "wp2b_layer_interactions.npy", inter.mean(axis=2))

    (d / "wp2b_exhaustive.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[wp2b] wrote {d / 'wp2b_exhaustive.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
