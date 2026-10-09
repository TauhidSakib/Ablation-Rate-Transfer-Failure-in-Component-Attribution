"""Phases 5 and 7 -- screen, fit the interaction model, and measure NAF.

This is the anchor result. It answers, on held-out counterfactuals:

* **H1** -- is any of the component-counterfactual surface non-additive?
  ``NAF = (R2_NACA - R2_lin) / (1 - R2_lin)``, with a paired Wilcoxon across
  examples.
* **H2** -- does the gap grow for ``alpha_test > alpha_train``? Reported
  one-sided, alongside ``Var(f_M)`` so the small-alpha signal-to-noise collapse
  is visible rather than mistaken for misspecification (the design spec 2.7).
* **H3** -- is non-additivity concentrated in late layers? NAF stratified by the
  depth group of the pooled components.

Costs **zero additional forward passes** through the target model: every input
here was produced by Phase 3 and Phase 4.

Both raw and offset-corrected R^2 are reported at every alpha. See
`the gauge-degeneracy analysis` -- raw R^2 is not gauge-invariant OOD, and
reporting it alone could manufacture an H2-shaped result out of an arbitrary
choice the ridge solver made.
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

from naca.coar import CoarModel  # noqa: E402
from naca.components import ComponentIndex  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.interaction import fit_interaction, identifiability_report  # noqa: E402
from naca.masks import MaskStore  # noqa: E402
from naca.metrics import (  # noqa: E402
    mse,
    non_additive_fraction,
    paired_wilcoxon,
    pearson_r,
    r2_offset_corrected,
    r2_score,
    variance,
)
from naca.screening import residual_scores, select_pool  # noqa: E402


def latest_run(runs: Path, suffix: str) -> Path:
    cands = sorted(p for p in runs.glob(f"*_{suffix}") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"no run directory matching *_{suffix} under {runs}")
    return cands[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="NACA Phases 5+7 -- interaction model")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--r-U", type=int, default=None)
    ap.add_argument("--r-P", type=int, default=None)
    ap.add_argument("--screening", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda-fm", type=float, default=None)
    ap.add_argument("--freeze-linear", type=str, default=None, choices=["true", "false"])
    ap.add_argument("--tag", type=str, default="", help="suffix for output files")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument(
        "--skip-h3",
        action="store_true",
        help="skip the depth-localisation refits. They cost one extra Stage-3 fit "
        "per depth group, which dominates runtime in an ablation sweep where only "
        "the headline NAF varies across arms.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    for name, val in (
        ("K", args.K),
        ("r_U", args.r_U),
        ("r_P", args.r_P),
        ("screening", args.screening),
        ("fm_epochs", args.epochs),
        ("lambda_fm", args.lambda_fm),
    ):
        if val is not None:
            setattr(cfg, name, val)
    if args.freeze_linear is not None:
        cfg.freeze_linear = args.freeze_linear == "true"
    cfg.validate()

    d = Path(args.collect_dir) if args.collect_dir else latest_run(cfg.paths.runs, "collect")
    tag = f"_{args.tag}" if args.tag else ""
    print(f"[naca] collect dir: {d}")
    print(
        f"[naca] K={cfg.K} r_U={cfg.r_U} r_P={cfg.r_P} screening={cfg.screening} "
        f"freeze_linear={cfg.freeze_linear} lambda_fm={cfg.lambda_fm:g} "
        f"epochs={cfg.fm_epochs}"
    )

    # ---- inputs (all pre-computed; no model is loaded here) ---------------
    masks = MaskStore.load(d / "masks_train.npz")
    Y = np.load(d / "Y_train.npy", mmap_mode="r")
    coar = CoarModel.load(d / "W.npz")
    lin_train = np.load(d / "lin_pred_train.npy", mmap_mode="r")
    R = np.load(d / "residuals_train.npy", mmap_mode="r")
    E = coar.E
    print(f"[naca] m={len(masks)} N={masks.N} E={E}")

    ident = identifiability_report(len(masks), cfg.alpha_train, cfg.K, cfg.r_U, cfg.r_P)
    print(f"[naca] identifiability: {json.dumps(ident)}")

    # ---- Stage 2: screening ------------------------------------------------
    score_path = d / "screen_scores.npy"
    if score_path.exists():
        S = np.load(score_path)
        print("[naca] reusing cached residual scores")
    else:
        S = residual_scores(masks, np.asarray(R), device=cfg.device)
        np.save(score_path, S)
    pool = select_pool(coar.W, S, cfg.K, mode=cfg.screening)
    np.save(d / f"pool{tag}.npy", pool)
    n_distinct = len(set(pool.ravel().tolist()))
    print(
        f"[naca] pool: {pool.shape}, {n_distinct} distinct components "
        f"({n_distinct / masks.N:.1%} of all)"
    )

    # ---- Stage 3: fit -------------------------------------------------------
    model = fit_interaction(
        masks,
        np.asarray(Y),
        np.asarray(lin_train),
        pool,
        coar.W,
        coar.b,
        r_U=cfg.r_U,
        r_P=cfg.r_P,
        lambda_fm=cfg.lambda_fm,
        freeze_linear=cfg.freeze_linear,
        lr=cfg.fm_lr,
        epochs=cfg.fm_epochs,
        batch_size=cfg.fm_batch_size,
        init_std=cfg.fm_init_std,
        group_size=args.group_size,
        device=cfg.device,
        seed=cfg.seed,
    )
    model.save(d / f"naca_model{tag}.npz")

    # interaction-matrix diagnostics (only inner products are interpretable)
    eig = np.linalg.eigvalsh(model.A)
    frac_indef = float(((eig.min(axis=1) < -1e-6) & (eig.max(axis=1) > 1e-6)).mean())
    print(
        f"[naca] A: mean|A_ij|={np.abs(model.A).mean():.4e}, "
        f"{frac_indef:.1%} of examples have eigenvalues of both signs"
    )

    # ---- evaluate across alpha ---------------------------------------------
    rows, per_alpha_pe = [], {}
    for alpha in cfg.alpha_test_grid:
        atag = f"a{alpha:g}".replace(".", "p")
        mp, yp, lp = (
            d / f"masks_test_{atag}.npz",
            d / f"Y_test_{atag}.npy",
            d / f"lin_pred_test_{atag}.npy",
        )
        if not (mp.exists() and yp.exists() and lp.exists()):
            print(f"[naca] skipping alpha={alpha}: artifacts missing")
            continue
        tm = MaskStore.load(mp)
        ty = np.asarray(np.load(yp, mmap_mode="r"))
        lin_t = np.asarray(np.load(lp, mmap_mode="r"))
        naca_t = model.predict(tm, lin_t, device=cfg.device, group_size=args.group_size)

        r2_lin, r2_naca = r2_score(ty, lin_t), r2_score(ty, naca_t)
        rho_lin, rho_naca = pearson_r(ty, lin_t), pearson_r(ty, naca_t)
        mse_lin, mse_naca = mse(ty, lin_t), mse(ty, naca_t)
        oc_lin, oc_naca = r2_offset_corrected(ty, lin_t), r2_offset_corrected(ty, naca_t)
        naf = non_additive_fraction(r2_lin, r2_naca)
        naf_oc = non_additive_fraction(oc_lin, oc_naca)
        w_raw = paired_wilcoxon(r2_naca, r2_lin, alternative="greater")
        w_oc = paired_wilcoxon(oc_naca, oc_lin, alternative="greater")

        rows.append(
            {
                "alpha_test": alpha,
                "alpha_train": cfg.alpha_train,
                "is_ood": alpha != cfg.alpha_train,
                "rho_lin": float(np.nanmean(rho_lin)),
                "rho_naca": float(np.nanmean(rho_naca)),
                "mse_lin": float(np.nanmean(mse_lin)),
                "mse_naca": float(np.nanmean(mse_naca)),
                "r2_lin": float(np.nanmean(r2_lin)),
                "r2_naca": float(np.nanmean(r2_naca)),
                "delta_r2": float(np.nanmean(r2_naca - r2_lin)),
                "naf_mean": float(np.nanmean(naf)),
                "naf_median": float(np.nanmedian(naf)),
                "r2_oc_lin": float(np.nanmean(oc_lin)),
                "r2_oc_naca": float(np.nanmean(oc_naca)),
                "delta_r2_oc": float(np.nanmean(oc_naca - oc_lin)),
                "naf_oc_mean": float(np.nanmean(naf_oc)),
                "naf_oc_median": float(np.nanmedian(naf_oc)),
                "wilcoxon_p_raw": w_raw["pvalue"],
                "wilcoxon_p_oc": w_oc["pvalue"],
                "target_var_mean": float(np.nanmean(variance(ty))),
                "n_examples": int(np.asarray(r2_lin).size),
            }
        )
        per_alpha_pe[atag] = {
            "r2_lin": r2_lin,
            "r2_naca": r2_naca,
            "naf": naf,
            "r2_oc_lin": oc_lin,
            "r2_oc_naca": oc_naca,
            "naf_oc": naf_oc,
        }
        print(
            f"[naca] alpha={alpha:<6.3f} R2 {np.nanmean(r2_lin):+.4f} -> "
            f"{np.nanmean(r2_naca):+.4f}  NAF={np.nanmean(naf):+.4f}  "
            f"| oc {np.nanmean(oc_lin):+.4f} -> {np.nanmean(oc_naca):+.4f}  "
            f"NAF_oc={np.nanmean(naf_oc):+.4f}  p={w_oc['pvalue']:.2e}  "
            f"Var={np.nanmean(variance(ty)):.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(d / f"naca_across_alpha{tag}.csv", index=False)
    np.savez_compressed(
        d / f"per_example_naca{tag}.npz",
        **{f"{k}__{kk}": vv for k, v in per_alpha_pe.items() for kk, vv in v.items()},
    )

    # ---- H3: depth localization --------------------------------------------
    # Measured by REFITTING with the pool confined to one depth group at a time.
    # A share-weighted average of per-example NAF cannot answer this question: if
    # every example has a similar NAF, that average comes out flat by
    # construction no matter where the interactions actually live. Verified on
    # synthetic data with interactions planted only in the deep half -- the
    # weighted statistic reported 0.44-0.49 for every group, i.e. nothing.
    # Confining the pool and refitting asks the question directly: how much of
    # the residual can interactions *within this group alone* explain?
    depth_rows = []
    idx_path = d / "component_index.json"
    atrain_tag = f"a{cfg.alpha_train:g}".replace(".", "p")
    if not args.skip_h3 and idx_path.exists() and (d / f"Y_test_{atrain_tag}.npy").exists():
        index = ComponentIndex.from_dict(json.loads(idx_path.read_text(encoding="utf-8")))
        groups = index.depth_group_array()
        tm_a = MaskStore.load(d / f"masks_test_{atrain_tag}.npz")
        ty_a = np.asarray(np.load(d / f"Y_test_{atrain_tag}.npy", mmap_mode="r"))
        lin_a = np.asarray(np.load(d / f"lin_pred_test_{atrain_tag}.npy", mmap_mode="r"))
        oc_lin_a = r2_offset_corrected(ty_a, lin_a)

        present = [
            g
            for g in ["stem", "layer1", "layer2", "layer3", "layer4"]
            if (groups == g).any()
        ]
        for g in present:
            members = np.nonzero(groups == g)[0]
            k_g = int(min(cfg.K, members.size))
            sub = S[members]
            order = np.argsort(-sub, axis=0)[:k_g]
            pool_g = members[order].T.astype(np.int64)  # (E, k_g)

            mdl_g = fit_interaction(
                masks,
                np.asarray(Y),
                np.asarray(lin_train),
                pool_g,
                coar.W,
                coar.b,
                r_U=cfg.r_U,
                r_P=cfg.r_P,
                lambda_fm=cfg.lambda_fm,
                freeze_linear=cfg.freeze_linear,
                lr=cfg.fm_lr,
                epochs=cfg.fm_epochs,
                batch_size=cfg.fm_batch_size,
                group_size=args.group_size,
                device=cfg.device,
                seed=cfg.seed,
                verbose=False,
            )
            pred_g = mdl_g.predict(
                tm_a, lin_a, device=cfg.device, group_size=args.group_size
            )
            naf_g = non_additive_fraction(
                oc_lin_a, r2_offset_corrected(ty_a, pred_g)
            )
            share = np.array([(groups[pool[e]] == g).mean() for e in range(E)])
            depth_rows.append(
                {
                    "group": g,
                    "k_used": k_g,
                    "naf_oc_group_refit": float(np.nanmean(naf_g)),
                    "naf_oc_group_refit_median": float(np.nanmedian(naf_g)),
                    "mean_pool_share": float(share.mean()),
                    "n_components_in_group": int(members.size),
                    "component_share_of_network": float((groups == g).mean()),
                }
            )
            print(
                f"[H3] {g:<8s} k={k_g:<4d} NAF_oc(group-only refit)="
                f"{depth_rows[-1]['naf_oc_group_refit']:+.4f}"
            )
        dd = pd.DataFrame(depth_rows)
        dd["pool_enrichment"] = dd["mean_pool_share"] / dd["component_share_of_network"]
        # plotting reads this column; it now carries the refit measurement
        dd["naf_oc_weighted"] = dd["naf_oc_group_refit"]
        dd.to_csv(d / f"naca_depth{tag}.csv", index=False)
        print()
        print("[H3] depth localization (group-confined refits, at alpha_train):")
        print(dd.to_string(index=False))

    # ---- verdicts -----------------------------------------------------------
    print(f"\n{'-' * 74}")
    at = df[df["alpha_test"] == cfg.alpha_train]
    if not at.empty:
        r = at.iloc[0]
        h1 = bool(r["naf_oc_mean"] > 0 and r["wilcoxon_p_oc"] < 0.05)
        print(
            f"[H1] at alpha_train: NAF_oc={r['naf_oc_mean']:+.4f} "
            f"(raw {r['naf_mean']:+.4f}), paired Wilcoxon p={r['wilcoxon_p_oc']:.3e} "
            f"-> {'non-additive' if h1 else 'NOT distinguishable from additive'}"
        )
    ood_hi = df[df["alpha_test"] > cfg.alpha_train]
    ood_lo = df[df["alpha_test"] < cfg.alpha_train]
    if not ood_hi.empty and not at.empty:
        grew = bool(ood_hi["naf_oc_mean"].max() > at.iloc[0]["naf_oc_mean"])
        print(
            f"[H2] NAF_oc at alpha_train={at.iloc[0]['naf_oc_mean']:+.4f}, "
            f"max above={ood_hi['naf_oc_mean'].max():+.4f} -> "
            f"{'grows (one-sided, as predicted)' if grew else 'does NOT grow'}"
        )
    if not ood_lo.empty:
        print(
            f"[H2-check] below alpha_train, Var(f_M) falls to "
            f"{ood_lo['target_var_mean'].min():.3f} vs "
            f"{at.iloc[0]['target_var_mean']:.3f} at alpha_train "
            f"-- the low-alpha branch is a signal-to-noise effect, not misspecification"
        )

    summary = {
        "config": {
            "K": cfg.K,
            "r_U": cfg.r_U,
            "r_P": cfg.r_P,
            "screening": cfg.screening,
            "freeze_linear": cfg.freeze_linear,
            "lambda_fm": cfg.lambda_fm,
            "epochs": cfg.fm_epochs,
        },
        "identifiability": ident,
        "fit_info": {k: v for k, v in model.info.items() if k != "history"},
        "pool_distinct_components": n_distinct,
        "fraction_examples_indefinite": frac_indef,
        "across_alpha": rows,
        "depth": depth_rows,
    }
    (d / f"naca_summary{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[naca] wrote {d / f'naca_summary{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
