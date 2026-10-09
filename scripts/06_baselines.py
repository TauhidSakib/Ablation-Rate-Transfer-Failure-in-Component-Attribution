"""Phase 6 (and Phase 10) -- competing estimators.

The headline job is the **danger baseline**: multi-alpha linear COAR. If fitting
the additive model at several ablation rates and selecting or interpolating by
``alpha_test`` closes the out-of-distribution gap, then the gap was calibration,
not misspecification, and the interaction story is undercut. That verdict has to
be reached *before* Phase 7's numbers are interpreted, which is why this script
prints an explicit verdict and writes it to disk.

Also runs, when asked: the K-restricted MLP and gradient-boosted-trees arms
(``--mlp``, ``--gbt``) and leave-one-out (``--loo``). The PSD ablation is not
here -- it is `04_fit_naca.py --r-P 0`, the same code path with the negative
block removed, which is what makes it a clean comparison.
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

from naca.baselines import (  # noqa: E402
    fit_gbt_baseline,
    fit_mlp_baseline,
    fit_multi_alpha,
    score_predictions,
)
from naca.coar import CoarModel, predict_chunked  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.masks import MaskStore  # noqa: E402
from naca.metrics import (  # noqa: E402
    non_additive_fraction,
    r2_score,
)


def latest_run(runs: Path, suffix: str) -> Path:
    cands = sorted(p for p in runs.glob(f"*_{suffix}") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"no run directory matching *_{suffix} under {runs}")
    return cands[-1]


def atag(alpha: float) -> str:
    return f"a{alpha:g}".replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser(description="NACA Phase 6/10 -- baselines")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--gbt", action="store_true")
    ap.add_argument("--max-examples", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = Path(args.collect_dir) if args.collect_dir else latest_run(cfg.paths.runs, "collect")
    print(f"[base] collect dir: {d}")

    # ---- assemble the alpha_train bank ------------------------------------
    train_sets: dict[float, tuple[MaskStore, np.ndarray]] = {
        cfg.alpha_train: (
            MaskStore.load(d / "masks_train.npz"),
            np.asarray(np.load(d / "Y_train.npy", mmap_mode="r")),
        )
    }
    for mp in sorted(d.glob("masks_train_a*.npz")):
        tag = mp.stem.replace("masks_train_", "")
        yp = d / f"Y_train_{tag}.npy"
        if not yp.exists():
            continue
        store = MaskStore.load(mp)
        train_sets[store.spec.alpha] = (store, np.asarray(np.load(yp, mmap_mode="r")))
    print(f"[base] alpha_train values available: {sorted(train_sets)}")

    if len(train_sets) < 2:
        print(
            "\n[base] WARNING: only one alpha_train available, so the multi-alpha "
            "danger baseline cannot be run. Collect more with:\n"
            "  python scripts/02_collect.py --run-dir <dir> --only-extra "
            "--extra-train-alphas 0.05,0.20 --extra-train-m 20000"
        )

    bank = fit_multi_alpha(
        train_sets, lam=cfg.lambda_ridge, device=cfg.device, method=cfg.solver
    )
    single = CoarModel.load(d / "W.npz")

    # ---- evaluate everything across the alpha grid -------------------------
    rows = []
    for alpha in cfg.alpha_test_grid:
        t = atag(alpha)
        mp, yp = d / f"masks_test_{t}.npz", d / f"Y_test_{t}.npy"
        if not (mp.exists() and yp.exists()):
            continue
        tm = MaskStore.load(mp)
        ty = np.asarray(np.load(yp, mmap_mode="r"))

        p_single = predict_chunked(single.W, single.b, tm, device=cfg.device)
        p_near, used = bank.predict_nearest(tm, alpha, device=cfg.device)
        p_interp = bank.predict_interpolated(tm, alpha, device=cfg.device)

        row = {"alpha_test": alpha, "nearest_alpha_train_used": used}
        for name, pred in (
            ("single_alpha", p_single),
            ("multi_alpha_nearest", p_near),
            ("multi_alpha_interp", p_interp),
        ):
            sc = score_predictions(ty, pred)
            for k, v in sc.items():
                row[f"{name}__{k}"] = v
        rows.append(row)
        print(
            f"[base] alpha={alpha:<6.3f} R2_oc  single={row['single_alpha__r2_oc_mean']:+.4f}  "
            f"multi-near={row['multi_alpha_nearest__r2_oc_mean']:+.4f}  "
            f"multi-interp={row['multi_alpha_interp__r2_oc_mean']:+.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(d / "baselines_multi_alpha.csv", index=False)

    # ---- the verdict --------------------------------------------------------
    verdict: dict = {"n_alpha_train": len(train_sets), "alphas": sorted(train_sets)}
    if len(train_sets) >= 2 and not df.empty:
        ood = df[df["alpha_test"] != cfg.alpha_train]
        gain_near = float(
            (ood["multi_alpha_nearest__r2_oc_mean"] - ood["single_alpha__r2_oc_mean"]).mean()
        )
        gain_int = float(
            (ood["multi_alpha_interp__r2_oc_mean"] - ood["single_alpha__r2_oc_mean"]).mean()
        )
        best = max(gain_near, gain_int)
        verdict.update(
            {
                "mean_ood_r2_oc_gain_nearest": gain_near,
                "mean_ood_r2_oc_gain_interp": gain_int,
                "closes_the_gap": bool(best > 0.05),
            }
        )
        print(f"\n{'-' * 74}")
        print(
            f"[DANGER BASELINE] mean OOD R2_oc gain from multi-alpha: "
            f"nearest {gain_near:+.4f}, interpolated {gain_int:+.4f}"
        )
        if best > 0.05:
            print(
                "[DANGER BASELINE] *** multi-alpha linear COAR substantially closes the "
                "OOD gap. Per the design spec Phase 6, STOP and re-scope: the interaction "
                "story for *prediction* is undercut. Pivot to editing (H4) plus the "
                "SNR/misspecification diagnostic. ***"
            )
        else:
            print(
                "[DANGER BASELINE] multi-alpha does NOT close the OOD gap "
                "(< 0.05 mean R2_oc). The interaction hypothesis survives its "
                "cheapest competitor; proceed to Phase 7."
            )

        # If NACA has already been fitted, put all three on one line.
        naca_csv = d / "naca_across_alpha.csv"
        if naca_csv.exists():
            nd = pd.read_csv(naca_csv)
            merged = df.merge(nd[["alpha_test", "r2_oc_naca"]], on="alpha_test", how="inner")
            merged["naca_minus_best_multi"] = merged["r2_oc_naca"] - merged[
                ["multi_alpha_nearest__r2_oc_mean", "multi_alpha_interp__r2_oc_mean"]
            ].max(axis=1)
            merged.to_csv(d / "baselines_vs_naca.csv", index=False)
            print("\n[base] NACA vs best multi-alpha, by alpha_test (R2_oc):")
            print(
                merged[
                    [
                        "alpha_test",
                        "single_alpha__r2_oc_mean",
                        "multi_alpha_interp__r2_oc_mean",
                        "r2_oc_naca",
                        "naca_minus_best_multi",
                    ]
                ].to_string(index=False)
            )

    # ---- optional non-linear arms -------------------------------------------
    extra: dict = {}
    if args.mlp or args.gbt:
        pool = np.load(d / "pool.npy")
        masks = train_sets[cfg.alpha_train][0]
        Y = train_sets[cfg.alpha_train][1]
        lin = np.asarray(np.load(d / "lin_pred_train.npy", mmap_mode="r"))
        n = args.max_examples
        r2_lin_tr = r2_score(Y[:, :n], lin[:, :n])

        if args.mlp:
            out = fit_mlp_baseline(
                masks, Y, lin, pool, device=cfg.device, max_examples=n, seed=cfg.seed
            )
            pred = lin[:, :n] + out["residual_pred"]
            r2 = r2_score(Y[:, :n], pred)
            extra["mlp"] = {
                "r2_mean": float(np.nanmean(r2)),
                "naf_mean": float(np.nanmean(non_additive_fraction(r2_lin_tr, r2))),
                "n_examples": n,
                "note": "in-sample on the training masks",
            }
            print(f"[base] MLP  in-sample R2={extra['mlp']['r2_mean']:.4f}")
        if args.gbt:
            out = fit_gbt_baseline(masks, Y, lin, pool, max_examples=n, seed=cfg.seed)
            pred = lin[:, :n] + out["residual_pred"]
            r2 = r2_score(Y[:, :n], pred)
            extra["gbt"] = {
                "r2_mean": float(np.nanmean(r2)),
                "naf_mean": float(np.nanmean(non_additive_fraction(r2_lin_tr, r2))),
                "n_examples": n,
                "note": "in-sample on the training masks",
            }
            print(f"[base] GBT  in-sample R2={extra['gbt']['r2_mean']:.4f}")

    (d / "baselines_summary.json").write_text(
        json.dumps({"verdict": verdict, "rows": rows, "extra": extra}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[base] wrote {d / 'baselines_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
