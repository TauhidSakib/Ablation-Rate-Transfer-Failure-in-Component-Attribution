"""WP1d -- does the aggregate term have structure, and does it explain Fig. 13?

Two experiments sharing one fitting routine.

EXPERIMENT 1 -- structure in the aggregate term.
wp1c corollary 3 showed that per-*group* kept-counts still vary at fixed alpha
(the total is pinned, the split is not), so of the G(G+1)/2 block-aggregate
features exactly one combination is unidentifiable and the rest are learnable
**from single-alpha data COAR already collects**. We fit:

    A  linear only                            (COAR)
    C  linear + 1 global term                 (needs multi-alpha)
    D  linear + 15 block-aggregate terms      single-alpha
    E  linear + 15 block-aggregate terms      pooled multi-alpha

If D beats A, aggregate structure is recoverable without collecting anything new,
which is the practically useful version of the finding.

EXPERIMENT 2 -- what actually causes the Fig. 13 correlation decay.
The base paper reports rho decaying on both sides of alpha_train and reads it as
possible misspecification. Two competing explanations:

  (i)  TRANSFER   -- the model is fitted at alpha_train and mis-transfers.
  (ii) INTRINSIC  -- an additive model fitted *at that alpha* would do no better,
                     because the counterfactual signal there is simply harder.

These are separable with an **oracle-at-alpha** control: fit a fresh linear model
using training masks drawn at alpha_test itself, and evaluate at alpha_test. That
removes transfer entirely. If the oracle also decays, the cause is intrinsic and
no amount of interaction modelling recovers it. Training sets exist at
alpha in {0.05, 0.10, 0.20}, giving one point on each side of alpha_train.

Note in advance: rho is invariant to the level error that dominates the aggregate
effect, so the aggregate term is *a priori* unlikely to explain a rho decay. The
experiment is run to find out, and the answer is reported either way.
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
from naca.components import ComponentIndex  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.masks import MaskStore  # noqa: E402
from naca.metrics import pearson_r, r2_offset_corrected, r2_score  # noqa: E402

CHUNK = 8192


def atag(a: float) -> str:
    return f"a{a:g}".replace(".", "p")


# ------------------------------------------------------------------ features
class FeatureSpec:
    """Builds aggregate feature columns from a block of masks."""

    def __init__(self, kind: str, groups: np.ndarray | None = None):
        self.kind = kind
        self.groups = groups
        if kind == "none":
            self.names: list[str] = []
        elif kind == "global":
            self.names = ["q_total"]
        elif kind == "block":
            # Derive the depth groups from the index rather than hardcoding
            # ResNet names: ViT (WP9) has block0..block11, and a hardcoded list
            # silently produced ZERO features and an empty model rather than
            # failing loudly. Preserve first-appearance (depth) order.
            seen: list[str] = []
            for g in np.asarray(groups).tolist():
                if g not in seen:
                    seen.append(g)
            gs = seen
            self.gs = gs
            self.masks_by_group = [(groups == g) for g in gs]
            self.names = []
            for i, g in enumerate(gs):
                self.names.append(f"{g}x{g}")
                for h in gs[i + 1 :]:
                    self.names.append(f"{g}x{h}")
        else:
            raise ValueError(kind)

    @property
    def F(self) -> int:
        return len(self.names)

    def build(self, Xblock: np.ndarray) -> np.ndarray:
        """``Xblock`` is (rows, N) of 0/1 -> (rows, F) float64 features."""
        if self.kind == "none":
            return np.zeros((Xblock.shape[0], 0), dtype=np.float64)
        n = Xblock.sum(axis=1, dtype=np.float64)
        if self.kind == "global":
            return (n * (n - 1.0) / 2.0).reshape(-1, 1)
        counts = [Xblock[:, m].sum(axis=1, dtype=np.float64) for m in self.masks_by_group]
        cols = []
        for i, ci in enumerate(counts):
            cols.append(ci * (ci - 1.0) / 2.0)
            for cj in counts[i + 1 :]:
                cols.append(ci * cj)
        return np.stack(cols, axis=1)


# -------------------------------------------------------------------- fitting
def fit_augmented(stores, Ys, lam, spec: FeatureSpec, *, device="cuda"):
    """Ridge on [X | F], intercept and F unpenalized, centered."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    N = stores[0].N
    E = int(Ys[0].shape[1])
    P = N + spec.F
    m_tot = sum(len(s) for s in stores)

    xsum = torch.zeros(P, dtype=torch.float64, device=dev)
    ysum = torch.zeros(E, dtype=torch.float64, device=dev)
    for s, Y in zip(stores, Ys):
        for lo in range(0, len(s), CHUNK):
            hi = min(lo + CHUNK, len(s))
            Xb = s.unpack(lo, hi)
            Xt = torch.from_numpy(Xb).to(dev, torch.float64)
            xsum[:N] += Xt.sum(dim=0)
            if spec.F:
                Ft = torch.from_numpy(spec.build(Xb)).to(dev, torch.float64)
                xsum[N:] += Ft.sum(dim=0)
                del Ft
            ysum += torch.from_numpy(np.ascontiguousarray(Y[lo:hi])).to(dev, torch.float64).sum(0)
            del Xt
    xbar, ybar = xsum / m_tot, ysum / m_tot

    # Standardize the aggregate columns. They carry magnitudes ~1e6 (pair counts)
    # against 0/1 mask columns, so an absolute ridge means nothing on them and the
    # Gram is numerically hopeless. A second pass gets their SD; the coefficients
    # are then reported in standardized units, which is also what makes them
    # comparable across blocks of very different size.
    fsd = torch.ones(max(spec.F, 1), dtype=torch.float64, device=dev)
    if spec.F:
        acc = torch.zeros(spec.F, dtype=torch.float64, device=dev)
        for s, _ in zip(stores, Ys):
            for lo in range(0, len(s), CHUNK):
                hi = min(lo + CHUNK, len(s))
                Ft = torch.from_numpy(spec.build(s.unpack(lo, hi))).to(dev, torch.float64)
                acc += ((Ft - xbar[N:].unsqueeze(0)) ** 2).sum(dim=0)
                del Ft
        fsd = (acc / m_tot).sqrt().clamp_min(1e-12)

    G = torch.zeros((P, P), dtype=torch.float64, device=dev)
    rhs = torch.zeros((P, E), dtype=torch.float64, device=dev)
    for s, Y in zip(stores, Ys):
        for lo in range(0, len(s), CHUNK):
            hi = min(lo + CHUNK, len(s))
            Xb = s.unpack(lo, hi)
            parts = [torch.from_numpy(Xb).to(dev, torch.float64)]
            if spec.F:
                parts.append(torch.from_numpy(spec.build(Xb)).to(dev, torch.float64))
            Z = torch.cat(parts, dim=1) - xbar.unsqueeze(0)
            if spec.F:
                Z[:, N:] /= fsd.unsqueeze(0)
            Yc = torch.from_numpy(np.ascontiguousarray(Y[lo:hi])).to(dev, torch.float64) - ybar
            G += Z.T @ Z
            rhs += Z.T @ Yc
            del Z, Yc, parts

    A = G.clone()
    idx = torch.arange(N, device=dev)
    A[idx, idx] += lam
    n_singular = 0
    if spec.F:
        # Scale-aware ridge on the standardized aggregates. At fixed alpha exactly
        # one combination of the block features is unidentifiable (wp1c Corollary 3),
        # so this block IS singular by construction -- the ridge selects the
        # minimum-norm representative of that direction rather than papering over
        # a numerical problem. It is 1e-6 relative, so identifiable directions are
        # untouched.
        idf = torch.arange(N, P, device=dev)
        ev = torch.linalg.eigvalsh(G[N:, N:])
        n_singular = int((ev < 1e-6 * float(ev.max())).sum())
        A[idf, idf] += 1e-6 * m_tot
    B = torch.cholesky_solve(rhs, torch.linalg.cholesky(A))
    if spec.F:
        B = B.clone()
        B[N:] /= fsd.unsqueeze(1)  # back to natural units
    b = ybar - xbar @ B
    model = {
        "W": B[:N].to(torch.float32).cpu().numpy(),
        "b": b.to(torch.float32).cpu().numpy(),
        "A_feat": B[N:].to(torch.float32).cpu().numpy() if spec.F else None,
        "spec": spec,
        "feature_sd": fsd.cpu().numpy() if spec.F else None,
        "n_singular_feature_dirs": n_singular,
    }
    del G, rhs, A, B
    torch.cuda.empty_cache()
    return model


def predict(model, store, *, device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    spec: FeatureSpec = model["spec"]
    Wt = torch.from_numpy(model["W"]).to(dev, torch.float32)
    bt = torch.from_numpy(model["b"]).to(dev, torch.float32)
    Af = (
        torch.from_numpy(model["A_feat"]).to(dev, torch.float32)
        if model["A_feat"] is not None
        else None
    )
    out = np.empty((len(store), model["W"].shape[1]), dtype=np.float32)
    for lo in range(0, len(store), CHUNK):
        hi = min(lo + CHUNK, len(store))
        Xb = store.unpack(lo, hi)
        p = torch.from_numpy(Xb).to(dev, torch.float32) @ Wt + bt.unsqueeze(0)
        if Af is not None:
            Ft = torch.from_numpy(spec.build(Xb)).to(dev, torch.float32)
            p = p + Ft @ Af
            del Ft
        out[lo:hi] = p.cpu().numpy()
        del p
    return out


def score(ty, pred):
    return {
        "rho": float(np.nanmean(pearson_r(ty, pred))),
        "r2": float(np.nanmean(r2_score(ty, pred))),
        "r2_oc": float(np.nanmean(r2_offset_corrected(ty, pred))),
    }


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="WP1d -- aggregate structure + Fig.13")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--lam", type=float, default=100.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = args.collect_dir or sorted(cfg.paths.runs.glob("*_collect"))[-1]
    print(f"[wp1d] collect dir: {d}\n")

    index = ComponentIndex.from_dict(
        json.loads((d / "component_index.json").read_text(encoding="utf-8"))
    )
    groups = index.depth_group_array()

    main_store = MaskStore.load(d / "masks_train.npz")
    main_Y = np.asarray(np.load(d / "Y_train.npy", mmap_mode="r"))
    extra = []
    for mp in sorted(d.glob("masks_train_a*.npz")):
        tag = mp.stem.replace("masks_train_", "")
        yp = d / f"Y_train_{tag}.npy"
        if yp.exists():
            extra.append((MaskStore.load(mp), np.asarray(np.load(yp, mmap_mode="r"))))
    pooled_stores = [main_store] + [s for s, _ in extra]
    pooled_Ys = [main_Y] + [y for _, y in extra]

    sp_none = FeatureSpec("none")
    sp_glob = FeatureSpec("global")
    sp_blk = FeatureSpec("block", groups)
    print(f"[wp1d] block features ({sp_blk.F}): {', '.join(sp_blk.names)}\n")

    # ---------------- EXPERIMENT 1 -----------------------------------------
    print("=" * 74)
    print("EXPERIMENT 1  does the aggregate term have structure?")
    print("=" * 74)
    single = CoarModel.load(d / "W.npz")
    models = {
        "A_linear_single": {"W": single.W, "b": single.b, "A_feat": None, "spec": sp_none},
    }
    print("[wp1d] fitting D: linear + block aggregates, SINGLE alpha")
    models["D_block_single"] = fit_augmented([main_store], [main_Y], args.lam, sp_blk, device=cfg.device)
    print("[wp1d] fitting C: linear + global term, pooled")
    models["C_global_pooled"] = fit_augmented(pooled_stores, pooled_Ys, args.lam, sp_glob, device=cfg.device)
    print("[wp1d] fitting E: linear + block aggregates, pooled")
    models["E_block_pooled"] = fit_augmented(pooled_stores, pooled_Ys, args.lam, sp_blk, device=cfg.device)
    print()

    # report the fitted block coefficients (mean across examples)
    fE = models["E_block_pooled"].get("A_feat")
    fD = models["D_block_single"].get("A_feat")
    if fE is None or fD is None:
        print(f"    [wp1d] no block-aggregate features for this architecture "
              f"({sp_blk.F} built) -- skipping the block coefficient table. "
              f"Models A and C, which carry the transfer result, are unaffected.")
    else:
        coef = fE.mean(axis=1)
        coef_s = fD.mean(axis=1)
        print(f"    {'feature':<16} {'pooled (E)':>13} {'single-alpha (D)':>18}")
        for nm, cE, cD in zip(sp_blk.names, coef, coef_s):
            print(f"    {nm:<16} {cE:>13.3e} {cD:>18.3e}")
    print()

    rows = []
    print(f"    {'alpha':>7} | " + " | ".join(f"{k.split('_')[0]:>6} rho  R2oc" for k in models))
    for a in cfg.alpha_test_grid:
        t = atag(a)
        mp, yp = d / f"masks_test_{t}.npz", d / f"Y_test_{t}.npy"
        if not (mp.exists() and yp.exists()):
            continue
        tm = MaskStore.load(mp)
        ty = np.asarray(np.load(yp, mmap_mode="r"))
        row = {"alpha_test": a}
        line = f"    {a:7.3f} | "
        for key, mdl in models.items():
            s = score(ty, predict(mdl, tm, device=cfg.device))
            row.update({f"{key}__{k}": v for k, v in s.items()})
            line += f"{s['rho']:6.4f} {s['r2_oc']:6.3f} | "
        rows.append(row)
        print(line)
    df = pd.DataFrame(rows)
    df.to_csv(d / "wp1d_structure.csv", index=False)

    pooled_alphas = {s.spec.alpha for s in pooled_stores}
    strict = df[~df.alpha_test.isin(pooled_alphas)]
    print()
    print(f"    mean over alphas held out from ALL models {sorted(strict.alpha_test)}:")
    for key in models:
        print(
            f"      {key:<18} rho {strict[f'{key}__rho'].mean():.4f}   "
            f"R2 {strict[f'{key}__r2'].mean():+8.3f}   "
            f"R2_oc {strict[f'{key}__r2_oc'].mean():+.4f}"
        )
    print()

    # ---------------- EXPERIMENT 2 -----------------------------------------
    print("=" * 74)
    print("EXPERIMENT 2  oracle-at-alpha: is the Fig.13 decay transfer or intrinsic?")
    print("=" * 74)
    oracle_rows = []
    print(f"    {'alpha':>7} {'m_train':>8} | {'transfer rho':>13} {'ORACLE rho':>11} "
          f"| {'transfer R2oc':>14} {'ORACLE R2oc':>12} {'Var(f_M)':>9}")
    for s, Y in [(main_store, main_Y)] + extra:
        a = s.spec.alpha
        t = atag(a)
        mp, yp = d / f"masks_test_{t}.npz", d / f"Y_test_{t}.npy"
        if not (mp.exists() and yp.exists()):
            continue
        tm = MaskStore.load(mp)
        ty = np.asarray(np.load(yp, mmap_mode="r"))
        # oracle: fit fresh at this alpha, evaluate at this alpha
        orc = fit_augmented([s], [Y], args.lam, sp_none, device=cfg.device)
        s_or = score(ty, predict(orc, tm, device=cfg.device))
        s_tr = score(ty, predict(models["A_linear_single"], tm, device=cfg.device))
        v = float(np.nanmean(ty.var(axis=0, ddof=1)))
        oracle_rows.append(
            {
                "alpha": a,
                "m_train": len(s),
                "transfer_rho": s_tr["rho"],
                "oracle_rho": s_or["rho"],
                "transfer_r2_oc": s_tr["r2_oc"],
                "oracle_r2_oc": s_or["r2_oc"],
                "target_var": v,
            }
        )
        print(f"    {a:7.3f} {len(s):8d} | {s_tr['rho']:13.4f} {s_or['rho']:11.4f} "
              f"| {s_tr['r2_oc']:14.4f} {s_or['r2_oc']:12.4f} {v:9.3f}")
        del orc

    od = pd.DataFrame(oracle_rows)
    od.to_csv(d / "wp1d_oracle.csv", index=False)
    print()
    at = od[od.alpha == cfg.alpha_train]
    if not at.empty and len(od) > 1:
        base = float(at.oracle_rho.iloc[0])
        print(f"    oracle rho at alpha_train = {base:.4f}")
        for _, r in od[od.alpha != cfg.alpha_train].iterrows():
            drop_tr = base - r.transfer_rho
            drop_or = base - r.oracle_rho
            share = drop_or / drop_tr if drop_tr > 1e-9 else float("nan")
            print(
                f"      alpha={r.alpha:6.3f}: total rho drop {drop_tr:+.4f}, "
                f"drop that survives the oracle {drop_or:+.4f}  "
                f"-> {share:.0%} INTRINSIC, {1 - share:.0%} transfer"
            )

    (d / "wp1d_summary.json").write_text(
        json.dumps(
            {
                "block_feature_names": sp_blk.names,
                "block_coef_pooled": [float(c) for c in coef],
                "block_coef_single": [float(c) for c in coef_s],
                "structure_rows": rows,
                "oracle_rows": oracle_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[wp1d] wrote {d / 'wp1d_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
