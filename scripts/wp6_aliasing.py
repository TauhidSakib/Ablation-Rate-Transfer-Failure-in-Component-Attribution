"""WP6 -- numerical checks for the coarsening-aliasing theorem.

Theorem in results/THEOREM_aliasing.md. Two checks, both on data already collected:

CHECK 1 -- exact validation of the aggregation map.
    Coarsen the 9 blocks into 3 supergroups. Predict the supergroup Walsh
    spectrum from the block spectrum using  ghat(S) = sum_{T : S(T)=S} fhat(T),
    and compare against the spectrum computed directly from the restricted
    surface. These must agree to machine precision -- it is a unit test for the
    theorem, not an approximation.

CHECK 2 -- the granularity reconciliation, tested rather than argued.
    Corollary 2: pairwise structure confined inside one block is invisible at
    block level (it aliases to the constant term). Block-level NAF is large, so
    the structure-carrying filters should be SPREAD across blocks. We compute the
    block membership of the top-25 residual-screened filters and compare against
    the uniform expectation given block sizes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from naca.components import ComponentIndex  # noqa: E402
from naca.granularity import build_unit_map  # noqa: E402
from wp2b_exhaustive_coarse import walsh  # noqa: E402


def S_of_T(T_bits: int, block_of: list[int], G_out: int) -> int:
    """S(T) = { g : |T ∩ B_g| odd }, as a bitmask over supergroups."""
    parity = 0
    for i, g in enumerate(block_of):
        if (T_bits >> i) & 1:
            parity ^= 1 << g
    return parity


def main() -> int:
    ap = argparse.ArgumentParser(description="WP6 -- aliasing theorem checks")
    ap.add_argument("--collect-dir", type=Path, default=None)
    args = ap.parse_args()
    d = args.collect_dir or sorted((ROOT / "runs").glob("*_A_collect"))[-1]
    print(f"[wp6] collect dir: {d}\n")

    Y = np.load(d / "wp2b_block_Y_fp32.npy").astype(np.float64)
    M = np.load(d / "wp2b_block_M.npy")
    G = M.shape[1]
    out: dict = {"G": G}

    # ---------------------------------------------------------------- CHECK 1
    print("=" * 74)
    print("CHECK 1 -- does the aggregation map hold exactly?")
    print("=" * 74)
    # 9 blocks -> 3 supergroups: stem+layer1 | layer2+layer3 | layer4
    block_of = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    G_out = 3
    print(f"    block -> supergroup: {block_of}")

    fhat = walsh(Y, G)  # (2^9, E)

    # predicted supergroup spectrum, by summing block coefficients over pre-images
    pred = np.zeros((1 << G_out, Y.shape[1]), dtype=np.float64)
    for T in range(1 << G):
        pred[S_of_T(T, block_of, G_out)] += fhat[T]

    # direct: restrict the surface to masks constant within each supergroup
    rows = []
    for s in range(1 << G_out):
        bits = [(s >> block_of[g]) & 1 for g in range(G)]
        idx = int(sum(b << g for g, b in enumerate(bits)))
        rows.append(idx)
    Y_restricted = Y[rows]
    direct = walsh(Y_restricted, G_out)

    err = np.abs(pred - direct).max()
    scale = max(float(np.abs(direct).max()), 1e-300)
    print(f"    predicted vs direct supergroup spectrum: max abs diff {err:.3e}")
    print(f"    relative to spectrum scale             : {err / scale:.3e}")
    ok = err / scale < 1e-10
    print(f"    -> theorem {'VERIFIED to machine precision' if ok else 'FAILED'}")
    out["check1"] = {"max_abs_diff": float(err), "relative": float(err / scale), "verified": bool(ok)}

    # corollaries, checked on the same map
    print()
    sing = all(bin(S_of_T(1 << i, block_of, G_out)).count("1") == 1 for i in range(G))
    print(f"    Corollary 1 (singletons stay singletons): {sing}")
    within = [
        (i, j) for i, j in itertools.combinations(range(G), 2) if block_of[i] == block_of[j]
    ]
    inv = all(S_of_T((1 << i) | (1 << j), block_of, G_out) == 0 for i, j in within)
    print(f"    Corollary 2 (within-supergroup pairs -> empty set): {inv}  "
          f"({len(within)} such pairs)")
    mono = all(
        bin(S_of_T(T, block_of, G_out)).count("1") <= bin(T).count("1") for T in range(1 << G)
    )
    print(f"    Corollary 3 (|S(T)| <= |T| always): {mono}")
    # cancellation: does any pre-image contain coefficients of both signs?
    cancel = 0
    for S in range(1 << G_out):
        pre = [fhat[T].mean() for T in range(1 << G) if S_of_T(T, block_of, G_out) == S]
        if pre and min(pre) < 0 < max(pre):
            cancel += 1
    print(f"    Corollary 4 (signed cancellation possible): {cancel}/{1 << G_out} "
          f"pre-images contain both signs")
    out["corollaries"] = {"c1_singletons": sing, "c2_within_invisible": inv,
                          "c3_order_nonincreasing": mono, "c4_mixed_sign_preimages": cancel}

    # ---------------------------------------------------------------- CHECK 2
    print()
    print("=" * 74)
    print("CHECK 2 -- are the structure-carrying filters spread across blocks?")
    print("=" * 74)
    idx = ComponentIndex.from_dict(
        json.loads((d / "component_index.json").read_text(encoding="utf-8"))
    )
    umap = build_unit_map(idx, "block")
    S = np.load(d / "screen_scores.npy")  # (N, E) residual screening scores
    agg = S.sum(axis=1)
    top25 = np.argsort(-agg)[:25]
    sizes = umap.sizes()
    obs = np.bincount(umap.unit_of[top25], minlength=umap.n_units)
    exp = 25 * sizes / sizes.sum()

    print(f"    {'block':<10} {'filters':>8} {'expected':>9} {'observed':>9}")
    for g, name in enumerate(umap.names):
        print(f"    {name:<10} {sizes[g]:>8} {exp[g]:>9.2f} {obs[g]:>9d}")
    n_blocks_hit = int((obs > 0).sum())
    max_share = float(obs.max() / 25)
    print()
    print(f"    blocks containing >=1 of the top-25 : {n_blocks_hit} of {umap.n_units}")
    print(f"    largest share in any single block   : {max_share:.0%}")
    within_pairs = int(sum(c * (c - 1) // 2 for c in obs))
    total_pairs = 25 * 24 // 2
    print(f"    top-25 pairs that are WITHIN a block: {within_pairs}/{total_pairs} "
          f"({within_pairs / total_pairs:.0%}) -- invisible at block level by Corollary 2")

    spread = n_blocks_hit >= 4 and max_share < 0.6
    print()
    if spread:
        print("    -> SPREAD. Sections 4 and 5 are consistent: sparse filter-level structure")
        print("       distributed across blocks is visible at block level, exactly as the")
        print("       theorem requires for NAF_block to be large.")
    else:
        print("    -> CONCENTRATED. By Corollary 2 most of this structure would be INVISIBLE")
        print("       at block level, so the large NAF_block must come from elsewhere. The two")
        print("       results are then in tension and this must be reported as such.")
    out["check2"] = {
        "top25_block_counts": obs.tolist(), "expected": exp.tolist(),
        "blocks_hit": n_blocks_hit, "max_share": max_share,
        "within_block_pairs": within_pairs, "total_pairs": total_pairs,
        "spread": bool(spread),
    }

    (d / "wp6_aliasing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[wp6] wrote {d / 'wp6_aliasing.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
