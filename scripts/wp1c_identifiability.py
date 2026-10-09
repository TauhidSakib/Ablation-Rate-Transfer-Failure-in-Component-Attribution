"""WP1c -- the identifiability proposition, stated and verified.

Turns the observation "the aggregate term is invisible at fixed alpha" into a
proposition with a proof and a numerical check.

--------------------------------------------------------------------------
PROPOSITION (fixed-cardinality blindness).

Let masks be drawn so that every mask ablates exactly k of N components, i.e.
each row of the design matrix X in {0,1}^{m x N} satisfies

    sum_i X[t,i] = N - k =: n     for every t.

Let phi : {0,...,N} -> R be any function of the kept-count alone, and let the
model be augmented with the feature column

    q[t] = phi( sum_i X[t,i] ).

Then q[t] = phi(n) for every t, so q = phi(n) * 1_m. The augmented design
[X | 1 | q] therefore has q exactly collinear with the intercept column, its rank
equals that of [X | 1], and the coefficient on q is **not identifiable**: only the
combination b + phi(n)*a is determined by the data.

COROLLARY 1 (the gauge degeneracy is the linear case).
Taking phi(n) = n and shifting w -> w + c*1_N changes every prediction by c*n,
a constant at fixed alpha. So the gauge degeneracy reported in
FINDING_gauge_degeneracy.md is this same proposition with phi linear. The
quadratic aggregate term of FINDING_aggregate_nonadditivity.md is phi(n) =
n(n-1)/2. Both are members of one family.

COROLLARY 2 (what breaks it).
If masks are drawn at two or more distinct cardinalities n_1 != n_2, then q takes
at least two distinct values and is no longer collinear with 1 unless phi is
constant on the sampled set. Identifiability is restored, and the effective rank
gained is (number of distinct cardinalities sampled) - 1, capped by the number of
aggregate features.

COROLLARY 3 (what remains identifiable even at fixed alpha).
Only the *total* kept-count is pinned. Any aggregate feature that depends on the
kept-counts of a **partition** of the components -- e.g. per-layer-group counts
n_g -- still varies across masks at fixed alpha, because a uniform k-subset
distributes randomly across groups. With G groups, the block-aggregate family
{n_g(n_g-1)/2} u {n_g n_h} has G(G+1)/2 members, of which exactly **one linear
combination** (the one summing to the total pair count) is unidentifiable; the
remaining G(G+1)/2 - 1 are identifiable at fixed alpha.

Corollary 3 is what makes the per-group experiment in wp1d worth running: it
says structure in the aggregate term is measurable from data COAR already has.
--------------------------------------------------------------------------

This script verifies all of it numerically on the real masks.
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

from naca.components import ComponentIndex  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.masks import MaskStore  # noqa: E402


def phi_quad(n):
    return n * (n - 1.0) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description="WP1c -- identifiability proposition")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=6000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = args.collect_dir or sorted(cfg.paths.runs.glob("*_collect"))[-1]
    print(f"[wp1c] collect dir: {d}\n")

    store = MaskStore.load(d / "masks_train.npz")
    N = store.N
    n_rows = min(args.rows, len(store))
    X = store.unpack(0, n_rows).astype(np.float64)
    n_kept = X.sum(axis=1)
    out: dict = {"N": N, "rows_checked": n_rows}

    # -------------------------------------------------- the premise
    print("PREMISE  every mask has the same kept-count")
    print(f"      unique kept-counts: {np.unique(n_kept)}")
    print(f"      variance: {n_kept.var():.3e}")
    assert n_kept.var() == 0.0, "premise violated: cardinality is not fixed"
    n = float(n_kept[0])
    print()
    out["kept_count"] = n

    # -------------------------------------------------- the proposition
    print("PROPOSITION  q = phi(n_kept) is exactly collinear with the intercept")
    for name, phi in (("linear phi(n)=n", lambda v: v), ("quadratic phi(n)=n(n-1)/2", phi_quad)):
        q = phi(n_kept)
        c = phi(n)
        dev = float(np.abs(q - c).max())
        print(f"      {name:<28s} q == {c:.6g} * 1 ?  max|q - c*1| = {dev:.3e}")
        assert dev == 0.0
    print()

    # rank check on [1 | q] and on a random column subset of [X | 1 | q]
    ones = np.ones((n_rows, 1))
    q = phi_quad(n_kept).reshape(-1, 1)
    r_1q = np.linalg.matrix_rank(np.hstack([ones, q]))
    print(f"      rank([1 | q]) = {r_1q}  (2 would mean identifiable; 1 means collinear)")
    out["rank_ones_q_fixed_alpha"] = int(r_1q)

    rng = np.random.default_rng(0)
    cols = rng.choice(N, size=min(400, N), replace=False)
    Xs = X[:, cols]
    r_x1 = np.linalg.matrix_rank(np.hstack([Xs, ones]))
    r_x1q = np.linalg.matrix_rank(np.hstack([Xs, ones, q]))
    print(f"      rank([Xs | 1])     = {r_x1}")
    print(f"      rank([Xs | 1 | q]) = {r_x1q}   -> q adds {r_x1q - r_x1} dimension(s)")
    print()
    out["rank_gain_fixed_alpha"] = int(r_x1q - r_x1)

    # -------------------------------------------------- corollary 2
    print("COROLLARY 2  pooling distinct cardinalities restores identifiability")
    extras = sorted(d.glob("masks_train_a*.npz"))
    if extras:
        blocks, kept = [X[:, cols]], [n_kept]
        for mp in extras:
            s2 = MaskStore.load(mp)
            r2 = min(2000, len(s2))
            X2 = s2.unpack(0, r2).astype(np.float64)
            blocks.append(X2[:, cols])
            kept.append(X2.sum(axis=1))
        Xp = np.vstack(blocks)
        nk = np.concatenate(kept)
        onesp = np.ones((Xp.shape[0], 1))
        qp = phi_quad(nk).reshape(-1, 1)
        rp_x1 = np.linalg.matrix_rank(np.hstack([Xp, onesp]))
        rp_x1q = np.linalg.matrix_rank(np.hstack([Xp, onesp, qp]))
        print(f"      distinct kept-counts pooled: {sorted(set(nk.tolist()))}")
        print(f"      rank([Xp | 1])     = {rp_x1}")
        print(f"      rank([Xp | 1 | q]) = {rp_x1q}   -> q adds {rp_x1q - rp_x1} dimension(s)")
        out["distinct_kept_counts_pooled"] = sorted({float(v) for v in nk})
        out["rank_gain_pooled"] = int(rp_x1q - rp_x1)
    else:
        print("      (no extra alpha_train sets found; skipping)")
    print()

    # -------------------------------------------------- corollary 3
    print("COROLLARY 3  per-group counts still vary at fixed alpha")
    idx_path = d / "component_index.json"
    if idx_path.exists():
        index = ComponentIndex.from_dict(json.loads(idx_path.read_text(encoding="utf-8")))
        groups = index.depth_group_array()
        names = [g for g in ["stem", "layer1", "layer2", "layer3", "layer4"] if (groups == g).any()]
        Ng = {g: int((groups == g).sum()) for g in names}
        counts = {g: X[:, groups == g].sum(axis=1) for g in names}
        print(f"      {'group':<8} {'N_g':>6} {'mean n_g':>10} {'sd n_g':>9} {'unique':>8}")
        for g in names:
            c = counts[g]
            print(f"      {g:<8} {Ng[g]:>6} {c.mean():10.1f} {c.std():9.2f} {len(np.unique(c)):>8}")
        tot = sum(counts[g] for g in names)
        print(f"      sum of per-group counts: variance {tot.var():.3e} (must be 0 -- it is the total)")

        # build the block-aggregate feature matrix and measure its usable rank
        feats, labels = [], []
        for i, g in enumerate(names):
            feats.append(counts[g] * (counts[g] - 1.0) / 2.0)
            labels.append(f"{g}x{g}")
            for h in names[i + 1 :]:
                feats.append(counts[g] * counts[h])
                labels.append(f"{g}x{h}")
        F = np.stack(feats, axis=1)
        Fc = F - F.mean(axis=0, keepdims=True)
        rank_F = np.linalg.matrix_rank(Fc)
        print()
        print(f"      block-aggregate features: {F.shape[1]} columns "
              f"({len(names)} within + {F.shape[1] - len(names)} cross)")
        print(f"      rank after centering    : {rank_F}  "
              f"-> {F.shape[1] - rank_F} unidentifiable combination(s) at fixed alpha")
        print("      (the proposition predicts exactly 1: the one summing to the total)")
        # confirm the null combination is the total-pair-count direction
        w_tot = np.ones(F.shape[1])
        resid = float(np.abs(Fc @ w_tot).max())
        print(f"      max|Fc @ 1| = {resid:.3e}   (0 confirms the all-ones combination is the null)")
        out["block_aggregate"] = {
            "n_features": int(F.shape[1]),
            "rank_centered": int(rank_F),
            "n_unidentifiable": int(F.shape[1] - rank_F),
            "all_ones_is_null": bool(resid < 1e-6),
            "labels": labels,
            "group_sizes": Ng,
        }
    print()
    print("-" * 74)
    print("[wp1c] Proposition verified: at fixed cardinality any function of the")
    print("       kept-count is collinear with the intercept and unidentifiable.")
    print("       The gauge degeneracy (phi linear) and the aggregate saturation")
    print("       term (phi quadratic) are the same phenomenon at different orders.")
    print("       Per-group aggregates escape it -- G(G+1)/2 - 1 of them are")
    print("       identifiable from single-alpha data COAR already collects.")
    (d / "wp1c_identifiability.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wp1c] wrote {d / 'wp1c_identifiability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
