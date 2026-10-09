"""Score WP7 against its pre-registration and emit the final two-seed table.

Pre-registration: results/PREREG_wp7_seed2_full.md, written in advance.
Reading committed in advance:

  NAF(K=25) rises toward ~0.029  -> the seed-2 shortfall was screening noise from
                                    half the masks; concentration REPLICATES.
  NAF(K=25) stays near ~0.0048   -> the magnitude of concentrated structure is
                                    seed-dependent while its existence is not.
  anything in between            -> report the value and decline to call it either.

Nothing here is tuned. The thresholds below are the ones written down beforehand.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
S1 = ROOT / "runs" / "YOUR_SEED1_COLLECT_DIR"


def naf_at(run: Path, tag: str, alpha: float = 0.10):
    p = run / f"naca_summary_{tag}.json"
    if not p.exists():
        return None
    b = json.loads(p.read_text(encoding="utf-8"))
    at = [r for r in b["across_alpha"] if r["alpha_test"] == alpha]
    if not at:
        return None
    return {
        "naf": at[0]["naf_oc_mean"],
        "r2_oc_naca": at[0]["r2_oc_naca"],
        "params": b["fit_info"]["n_params_per_example"],
        "K": b["config"]["K"],
        "rank": b["config"]["r_U"] + b["config"]["r_P"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed2-25k", type=Path, default=None, help="the m=25,000 seed-2 run")
    ap.add_argument("--seed2-50k", type=Path, default=None, help="the m=50,000 seed-2 run")
    args = ap.parse_args()

    runs = sorted((ROOT / "runs").glob("*_A-s1_collect"))
    s2_25 = args.seed2_25k or (runs[0] if runs else None)
    s2_50 = args.seed2_50k or (runs[-1] if len(runs) > 1 else None)
    print(f"seed1 (m=50k): {S1.name}")
    print(f"seed2 (m=25k): {s2_25.name if s2_25 else '-'}")
    print(f"seed2 (m=50k): {s2_50.name if s2_50 else '-'}")
    print()

    Ks = [25, 50, 100, 200]
    print("=" * 78)
    print("CAPACITY-MATCHED K-SWEEP (rank/K ~ 0.6), NAF at alpha_train")
    print("=" * 78)
    print(f"  {'K':>5} {'params':>8} | {'seed1 m=50k':>13} {'seed2 m=25k':>13} {'seed2 m=50k':>13}")
    table = {}
    for K in Ks:
        row = []
        for run in (S1, s2_25, s2_50):
            v = naf_at(run, f"Kcm{K}") if run else None
            row.append(v["naf"] if v else float("nan"))
        params = next((naf_at(r, f"Kcm{K}")["params"] for r in (S1, s2_50) if r and naf_at(r, f"Kcm{K}")), None)
        table[K] = row
        print(f"  {K:>5} {str(params):>8} | {row[0]:>13.5f} {row[1]:>13.5f} {row[2]:>13.5f}")

    print()
    print("=" * 78)
    print("PRE-REGISTERED SCORING")
    print("=" * 78)
    s1_25, s2a_25, s2b_25 = table[25]
    s1_100, _, s2b_100 = table[100]
    print(f"  seed1 NAF(K=25), m=50k        : {s1_25:.5f}")
    print(f"  seed2 NAF(K=25), m=25k        : {s2a_25:.5f}   <- the ambiguous number")
    print(f"  seed2 NAF(K=25), m=50k        : {s2b_25:.5f}   <- the test")
    print()

    # thresholds as pre-registered: "toward ~0.029" vs "stays near ~0.0048"
    hi, lo = 0.029, 0.0048

    if s2b_25 >= 0.6 * hi:
        verdict = ("BUDGET. The seed-2 shortfall was screening noise from half the masks. "
                   "Concentration REPLICATES; quote the matched-m numbers for both seeds.")
    elif s2b_25 <= 1.6 * lo:
        verdict = ("SEED-DEPENDENT MAGNITUDE. The existence of concentrated structure replicates "
                   "(K=25 >> K=100 on both seeds); its size does not. Report as a range.")
    else:
        verdict = (f"INTERMEDIATE ({s2b_25:.5f}, between {lo} and {hi}). Per the pre-registration, "
                   f"report the value and decline to call it either way.")
    print(f"  -> {verdict}")

    ratio_1 = s1_25 / s1_100 if s1_100 else float("nan")
    ratio_2 = s2b_25 / s2b_100 if s2b_100 else float("nan")
    print()
    print(f"  concentration ratio K=25 / K=100:  seed1 {ratio_1:.1f}x   seed2(50k) {ratio_2:.1f}x")

    pf = (s2_50 / "precision_floor.json") if s2_50 else None
    if pf and pf.exists():
        j = json.loads(pf.read_text(encoding="utf-8"))
        floor = j["floor_fraction_of_residual"]
        # The floor is noise_var / residual_var: a share of the residual, so it
        # must lie in [0, 1). A value >= 1 means the "repeat measurement" did not
        # measure the same model -- which is exactly what happened when this stage
        # was invoked without --config and loaded the seed-1 checkpoint against
        # seed-2 margins (floor came out 8.69). Refuse to report it as a floor.
        if not (0.0 <= floor < 1.0):
            print(f"  seed2 precision floor          : INVALID ({floor:.4f})")
            print(f"    A floor is noise/residual and must be < 1. This run compared")
            print(f"    against a different model. Re-run wp0_precision_floor.py with")
            print(f"    the matching --config; the checkpoint guard there will confirm.")
        else:
            print(f"  seed2 precision floor          : {floor:.6f}")
            print(f"  NAF(K=25) above floor by       : {s2b_25 / floor:.0f}x")

    out = {"table": {str(k): v for k, v in table.items()}, "verdict": verdict,
           "ratio_seed1": ratio_1, "ratio_seed2_50k": ratio_2,
           "note": "thresholds fixed in PREREG_wp7_seed2_full.md in advance"}
    (ROOT / "results" / "wp7_score.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {ROOT / 'results' / 'wp7_score.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
