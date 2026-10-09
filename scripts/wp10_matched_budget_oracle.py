"""Is the alpha=0.05 -> alpha=0.10 oracle step confounded by training budget?

Seed 4 breaks strict monotonicity by -0.000132 at exactly that step. The step is
also the only one where the two oracles are fitted on unequal data: alpha=0.10 is
the main pool (m = 25,000, or 50,000 on seeds 1-2) while alpha=0.05 is an extra
pool at m = 10,000. More data raises rho, so the confound pushes in precisely the
direction of the observed inversion.

This refits the alpha_train oracle on a random 10,000-mask SUBSAMPLE, matching the
alpha=0.05 budget exactly, and recomputes the comparison. Everything else -- the
estimator, lambda, the evaluation set -- is unchanged.

This removes the confound rather than caveating it. Whatever it shows is reported.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from naca.config import load_config  # noqa: E402
from naca.masks import MaskStore  # noqa: E402
from wp1d_structure_and_fig13 import (  # noqa: E402
    FeatureSpec, atag, fit_augmented, predict, score,
)

SEEDS = [
    ("seed 1", "YOUR_SEED1_COLLECT_DIR", "setup_a.yaml"),
    ("seed 2", "YOUR_SEED2_COLLECT_DIR", "setup_a_seed1_full.yaml"),
    ("seed 3", "*_A-s2_collect", "setup_a_seed2.yaml"),
    ("seed 4", "*_A-s3_collect", "setup_a_seed3.yaml"),
    ("seed 5", "*_A-s4_collect", "setup_a_seed4.yaml"),
]
M_MATCH = 10000


def find(pat):
    if "*" in pat:
        c = sorted((ROOT / "runs").glob(pat))
        return c[-1] if c else None
    p = ROOT / "runs" / pat
    return p if p.exists() else None


def main() -> int:
    sp_none = FeatureSpec("none")
    rng = np.random.default_rng(0)
    print("=" * 92)
    print("MATCHED-BUDGET ORACLE: alpha=0.05 vs alpha=0.10, both fitted on m = 10,000")
    print("=" * 92)
    print(f"  {'seed':<8}{'m(0.10) orig':>14}{'rho@.05':>10}{'rho@.10 orig':>14}"
          f"{'rho@.10 @10k':>14}{'gap orig':>11}{'gap matched':>13}")
    out = []
    for label, pat, cfgname in SEEDS:
        d = find(pat)
        if d is None:
            continue
        cfg = load_config(ROOT / "configs" / cfgname)
        a_tr = cfg.alpha_train

        # the alpha=0.05 oracle, as computed by wp1d
        wp1d = d / "wp1d_summary.json"
        if not wp1d.exists():
            print(f"  {label:<8}  (no wp1d_summary.json yet; skipped)")
            continue
        w = json.loads(wp1d.read_text(encoding="utf-8"))
        orc = {round(float(r["alpha"]), 4): r for r in w["oracle_rows"]}
        if 0.05 not in orc or round(a_tr, 4) not in orc:
            continue
        rho05 = orc[0.05]["oracle_rho"]
        rho10_orig = orc[round(a_tr, 4)]["oracle_rho"]

        store = MaskStore.load(d / "masks_train.npz")
        Y = np.asarray(np.load(d / "Y_train.npy", mmap_mode="r"))
        m_orig = len(store)
        if m_orig <= M_MATCH:
            rho10_match = rho10_orig
        else:
            idx = np.sort(rng.choice(m_orig, size=M_MATCH, replace=False))
            # MaskStore has no subset(); rows of the bit-packed array slice directly.
            sub = MaskStore(np.ascontiguousarray(store.packed[idx]),
                            replace(store.spec, m=M_MATCH))
            mdl = fit_augmented([sub], [Y[idx]], 100.0, sp_none, device=cfg.device)
            t = atag(a_tr)
            tm = MaskStore.load(d / f"masks_test_{t}.npz")
            ty = np.asarray(np.load(d / f"Y_test_{t}.npy", mmap_mode="r"))
            rho10_match = score(ty, predict(mdl, tm, device=cfg.device))["rho"]

        g0, g1 = rho05 - rho10_orig, rho05 - rho10_match
        print(f"  {label:<8}{m_orig:>14}{rho05:>10.6f}{rho10_orig:>14.6f}"
              f"{rho10_match:>14.6f}{g0:>+11.6f}{g1:>+13.6f}")
        out.append({"seed": label, "m_orig": m_orig, "rho05": rho05,
                    "rho10_orig": rho10_orig, "rho10_matched": rho10_match,
                    "gap_orig": g0, "gap_matched": g1})

    print()
    print("  gap = rho(0.05) - rho(0.10). POSITIVE means monotonicity holds at this step.")
    n_ok0 = sum(1 for r in out if r["gap_orig"] > 0)
    n_ok1 = sum(1 for r in out if r["gap_matched"] > 0)
    worst = min(out, key=lambda r: r["gap_orig"]) if out else None

    # Reporting order is fixed here so it cannot drift. The AS-COLLECTED result is
    # primary: it is what the pre-registered criterion was scored on. The matched-
    # budget refit is a secondary diagnostic explaining the one break -- it does
    # not replace the primary count, and must never be quoted in its place.
    print()
    print("  " + "=" * 86)
    print("  PRIMARY (as collected):   "
          f"{n_ok0}/{len(out)} seeds monotone at this step" +
          (f"; {worst['seed']} breaks by {worst['gap_orig']:+.6f}"
           if worst and worst["gap_orig"] <= 0 else ""))
    print("  SECONDARY (all seeds refit at matched m = 10,000):   "
          f"{n_ok1}/{len(out)} seeds monotone")
    print("  " + "-" * 86)
    print("  The secondary result explains the primary break; it does not overturn it.")
    print("  Quote the primary count first, always.")
    print("  " + "=" * 86)

    p = ROOT / "results" / "wp10_matched_budget_oracle.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
