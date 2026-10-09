"""Print every result the project has produced, in one command.

Written so that a session which comes back cold can see the full state without
re-deriving anything:

    python scripts/print_results.py

Missing outputs are reported as PENDING rather than raising, so this is also the
right thing to run while work is still in flight.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
RUN = ROOT / "runs" / "YOUR_SEED1_COLLECT_DIR"


def rule(t=""):
    print("=" * 78)
    if t:
        print(t)
        print("=" * 78)


def load(name):
    p = RUN / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"  [!] {name} unreadable: {exc}")
        return None


def csv(name):
    p = RUN / name
    if not p.exists():
        return None
    import pandas as pd

    try:
        return pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001 - want the diagnostic, not a crash
        print(f"  [!] {name} unreadable: {exc}")
        return None


def main() -> int:
    rule("NACA — full results")
    print(f"run dir: {RUN}")

    # ---------------------------------------------------------- what exists
    rule("0. PIPELINE STATE")
    wanted = [
        ("collect_metrics.json", "Phase 3 counterfactual collection"),
        ("linear_summary.json", "Phase 4 linear COAR + gate"),
        ("naca_summary.json", "Phase 5/7 interaction model, NAF"),
        ("wp1a_gauge_diagnostic.json", "WP1a gauge diagnostic"),
        ("wp1b_aggregate.json", "WP1b aggregate term"),
        ("wp1c_identifiability.json", "WP1c proposition"),
        ("wp1d_summary.json", "WP1d structure + oracle-at-alpha"),
        ("wp2a_calibration.json", "WP2a severity calibration"),
        ("wp2b_exhaustive.json", "WP2b exhaustive block surface"),
        ("wp2b_severity_axis.csv", "WP2b severity axis (pre-registered)"),
        ("precision_floor.json", "bf16 precision floor"),
        ("ablation_table.csv", "Phase 10 ablation table"),
    ]
    for f, desc in wanted:
        print(f"  [{'done' if (RUN / f).exists() else '    '}] {desc:<44} {f}")

    for lg, label in (
        ("remaining.log", "unattended runner"),
        ("chain.log", "earlier chain"),
    ):
        p = ROOT / "runs" / lg
        if p.exists():
            print(f"\n  --- {label} ({lg}) ---")
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]:
                print(f"      {line}")

    # ------------------------------------------------------------ gates
    rule("1. REPRODUCTION GATES")
    cm = load("collect_metrics.json")
    if cm:
        print(f"  model test accuracy      {cm.get('model_test_acc', float('nan')):.4f}  (base paper 0.91)")
        print(f"  N components             {cm.get('N')}")
        print(f"  collection wall time     {cm.get('total_minutes', 0):.0f} min")
    ls = load("linear_summary.json")
    if ls:
        print(f"  rho at alpha_train       {ls.get('rho_at_alpha_train', float('nan')):.4f}  "
              f"(base paper {ls.get('paper_rho')})  gate_passed={ls.get('gate_passed')}")
        print(f"  lambda_ridge             {ls.get('lambda_ridge')}")

    # ------------------------------------------------- the aggregate story
    rule("2. THE AGGREGATE TERM  (WP1a / WP1b)")
    ga = load("wp1a_gauge_diagnostic.json")
    if ga:
        v = ga.get("verdict", {})
        print(f"  gauge constant needed    {v.get('c_needed', float('nan')):.3e}")
        print(f"  gauge constant present   {v.get('c_present', float('nan')):.3e}")
        print(f"  shortfall                {v.get('shortfall_factor', float('nan')):.0f}x  -> {v.get('branch')}")
        print(f"  predicted mean spread    {ga.get('pred_mean_spread', float('nan')):.3f}")
        print(f"  true mean spread         {ga.get('true_mean_spread', float('nan')):.3f}")
        sat = ga.get("saturation", {})
        sl = sat.get("local_slopes", [])
        if sl:
            print(f"  marginal effect decay    {sl[0]:.4f} -> {sl[-1]:.4f}  "
                  f"({abs(sl[0] / sl[-1]) if sl[-1] else float('nan'):.1f}x)")
            print(f"  linear fit R2 {sat.get('linear_fit_r2', 0):.4f}   "
                  f"quadratic fit R2 {sat.get('quadratic_fit_r2', 0):.4f}")
    ab = load("wp1b_aggregate.json")
    if ab:
        print(f"\n  mean(w) single-alpha     {ab.get('kappa_single_alpha', float('nan')):+.3e}")
        print(f"  mean(w) pooled alpha     {ab.get('kappa_pooled', float('nan')):+.3e}")
        print(f"  global term coefficient  {ab.get('global_term_coefficient', float('nan')):+.3e}")
        for metric in ("mean_ood_r2_raw", "mean_ood_r2_oc", "mean_ood_rho"):
            m = ab.get(metric, {})
            if m:
                print(f"  {metric:<24} A {m.get('A', 0):+8.4f}   B {m.get('B', 0):+8.4f}   C {m.get('C', 0):+8.4f}")

    # --------------------------------------------------- the proposition
    rule("3. THE PROPOSITION  (WP1c)")
    ic = load("wp1c_identifiability.json")
    if ic:
        print(f"  kept-count at alpha_train        {ic.get('kept_count')}")
        print(f"  rank([1|q]) at fixed alpha       {ic.get('rank_ones_q_fixed_alpha')}  (1 = collinear)")
        print(f"  rank gain from q, fixed alpha    {ic.get('rank_gain_fixed_alpha')}")
        print(f"  rank gain from q, pooled         {ic.get('rank_gain_pooled')}")
        ba = ic.get("block_aggregate", {})
        if ba:
            print(f"  block features rank / count      {ba.get('rank_centered')} / {ba.get('n_features')}"
                  f"   unidentifiable={ba.get('n_unidentifiable')}")

    # ------------------------------------------------ transfer vs intrinsic
    rule("4. ORACLE-AT-ALPHA  (WP1d) — is the alpha-decay transfer or intrinsic?")
    od = csv("wp1d_oracle.csv")
    if od is not None:
        print(od.to_string(index=False))
        at = od[od.alpha == 0.10]
        if not at.empty:
            base = float(at.oracle_rho.iloc[0])
            print(f"\n  oracle rho is monotone in alpha: "
                  f"{', '.join(f'{r.oracle_rho:.4f}' for r in od.sort_values('alpha').itertuples())}")
            for r in od[od.alpha != 0.10].itertuples():
                tot = base - r.transfer_rho
                orc = base - r.oracle_rho
                if abs(tot) > 1e-9:
                    print(f"    alpha={r.alpha:.3f}: {orc / tot:6.0%} intrinsic, {1 - orc / tot:6.0%} transfer")
    sd = csv("wp1d_structure.csv")
    if sd is not None:
        keys = [c for c in sd.columns if c.endswith("__r2_oc")]
        pooled = {0.05, 0.10, 0.20}
        strict = sd[~sd.alpha_test.isin(pooled)]
        print("\n  mean over alphas held out from ALL models:")
        for k in keys:
            print(f"    {k.replace('__r2_oc',''):<20} R2_oc {strict[k].mean():+.4f}")

    # ----------------------------------------------------- the exhaustive
    rule("5. EXHAUSTIVE BLOCK SURFACE  (WP2b)")
    ex = load("wp2b_exhaustive.json")
    if ex is None:
        print("  PENDING")
    else:
        print(f"  forward precision: {ex.get('forward_dtype', 'unknown')}")
        blk = ex.get("block", {})
        print(f"  units {blk.get('n_units')}  masks {blk.get('n_masks')}  sizes {blk.get('unit_sizes')}")
        od2 = csv("wp2b_block_orders.csv")
        if od2 is not None:
            print("\n  orthogonal (Walsh) variance decomposition by order:")
            print(od2.to_string(index=False))
        print(f"\n  EXACT NAF at block granularity: {blk.get('exact_naf', float('nan')):+.6f}")
        u = ex.get("order2_uniformity", {})
        if u:
            print("\n  is order-2 uniform?")
            print(f"    mean beta_ij {u.get('mean_beta', 0):+.5f}  sd {u.get('sd_beta', 0):.5f}  cv {u.get('cv', 0):.3f}")
            print(f"    a from exhaustive {u.get('a_from_exhaustive', 0):+.3e}   "
                  f"a from model C {u.get('a_from_model_c') or float('nan'):+.3e}")
            print(f"    R2 of beta_ij ~ a*n_i*n_j: {u.get('r2_of_uniform_prediction', 0):.4f}")
            print(f"    -> {u.get('verdict')}")
        sa = ex.get("saturation_block", {})
        if sa:
            print(f"\n  saturation at block level: first/last ratio {sa.get('first_last_ratio', 0):.2f}x"
                  f"   (filter level {sa.get('filter_level_ratio')}x)")
            print(f"    linear fit R2 {sa.get('linear_fit_r2', 0):.4f}  quadratic {sa.get('quadratic_fit_r2', 0):.4f}")
        sv = ex.get("severity_axis", {})
        if sv:
            print(f"\n  PRE-REGISTERED severity test: corr(severity, additive R2) = "
                  f"{sv.get('corr_severity_vs_additive_r2', float('nan')):+.4f}")
            print(f"    -> {sv.get('verdict')}")
        lay = ex.get("layer", {})
        if lay:
            print(f"\n  layer level: |main| {lay.get('mean_abs_main', 0):.4f}  "
                  f"|pairwise| {lay.get('mean_abs_pairwise', 0):.4f}  "
                  f"ratio {lay.get('ratio', 0):.4f}  uniform-R2 {lay.get('r2_of_uniform_prediction', 0):.4f}")
    sev = csv("wp2b_severity_axis.csv")
    if sev is not None:
        print("\n  non-additivity vs severity, stratum by stratum:")
        print(sev.to_string(index=False))
    pt = csv("wp2b_protocol_trial.csv")
    if pt is not None:
        print("\n  the protocol on trial (fit at one cardinality, score on the whole surface):")
        print(pt.to_string(index=False))

    # ------------------------------------------------------ severity calib
    rule("6. SEVERITY CALIBRATION  (WP2a)")
    cal = load("wp2a_calibration.json")
    if cal:
        print(f"  target mean|delta margin| = {cal.get('target_mean_abs_delta', 0):.4f} (filter, alpha_train)")
        for lvl, c in (cal.get("chosen") or {}).items():
            print(f"    {lvl:<14} n_units {c.get('n_units'):>5}  k {c.get('k_units'):>4}  "
                  f"comp_frac {c.get('mean_component_fraction', 0):.4f}  "
                  f"severity ratio {c.get('ratio_to_target', 0):.3f}")

    # -------------------------------------------------------- precision
    rule("7. PRECISION FLOOR — does bf16 limit the NAF null?")
    pf = load("precision_floor.json")
    if pf is None:
        print("  PENDING")
    else:
        print(f"  bf16 vs fp32 discrepancy: sd {pf.get('diff_sd', 0):.6f}  max {pf.get('diff_max_abs', 0):.6f}")
        print(f"  residual variance         {pf.get('residual_var', 0):.4f}")
        print(f"  noise variance            {pf.get('noise_var', 0):.6f}")
        print(f"  FLOOR                     {pf.get('floor_fraction_of_residual', 0):.6f}")
        print(f"  measured NAF              {pf.get('naf_measured', 0):.6f}")
        print(f"  NAF / floor               {pf.get('naf_over_floor', 0):.2f}x")
        print(f"  bias-corrected NAF        {pf.get('naf_bias_corrected', 0):.6f}")
        print(f"  -> {pf.get('verdict')}")

    # --------------------------------------------------------- ablations
    rule("8. ABLATION TABLE  (Phase 10)")
    ab2 = csv("ablation_table.csv")
    if ab2 is None:
        print("  PENDING")
    else:
        cols = [c for c in ("arm", "value", "naf_oc_at_train", "r2_oc_naca",
                            "n_params_per_example", "wilcoxon_p", "failed") if c in ab2]
        print(ab2[cols].to_string(index=False))
        if "naf_oc_at_train" in ab2:
            n = ab2["naf_oc_at_train"].dropna()
            if len(n):
                print(f"\n  NAF across all arms: min {n.min():+.5f}  max {n.max():+.5f}  "
                      f"median {n.median():+.5f}")
                # NOT a null verdict. The finding is CONCENTRATION: NAF is large in a
                # small pool and falls as the pool widens. Report the spread and the
                # floor, never a single number. See PAPER.md 5.4.
                FLOOR = 0.00025  # bf16 noise floor (reference)
                lo, hi = float(n.min()), float(n.max())
                print(f"  -> spread {hi / max(lo, 1e-12):.0f}x across arms; "
                      f"floor {FLOOR:.5f} ({hi / FLOOR:.0f}x at the top arm, "
                      f"{lo / FLOOR:.1f}x at the bottom)")
                print("     Read as CONCENTRATION, not a null. Arms at or below the floor "
                      "are not measurements;")
                print("     never quote one without the floor.")

    # ------------------------------------------------------------ errors
    rule("9. FAILURES IN ANY LOG")
    found = False
    for p in sorted((ROOT / "runs").glob("*.err")):
        if p.stat().st_size > 0:
            txt = p.read_text(encoding="utf-8", errors="replace").strip()
            if txt and ("Traceback" in txt or "Error" in txt):
                found = True
                print(f"  --- {p.name} ---")
                for line in txt.splitlines()[-6:]:
                    print(f"      {line}")
    if not found:
        print("  none")

    rule()
    print("paper: PAPER.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
