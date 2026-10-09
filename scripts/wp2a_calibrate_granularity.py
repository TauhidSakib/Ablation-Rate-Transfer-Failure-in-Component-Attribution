"""WP2a -- calibrate the severity-matched protocol before collecting anything.

CLAUDE-PHASE2 section 3.2 is the important constraint: at 9 units, alpha = 0.10
means ablating 0.9 units, and one whole residual block is catastrophically more
damaging than 10% of filters. Comparing granularities at matched *fraction*
would confound granularity with ablation severity, and any apparent trend in the
resulting NAF-vs-N curve would be an artifact of that confound rather than a
property of computation.

So severity is measured first, and the collection budget is set from it. For each
granularity we sweep the number of ablated units ``k`` and record the mean
absolute change in margin over the eval set, then pick the ``k`` whose damage
matches the filter-level reference at alpha = 0.10 as closely as possible.

Outputs a calibration table and the chosen ``k`` per level, which
`wp2b_collect_granularity.py` consumes. Nothing is collected here -- this is
cheap (a few hundred forward passes) and must be inspected before committing
hours of GPU time.
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

from naca.collect import preload_eval  # noqa: E402
from naca.components import build_component_index  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.datasets import load_dataset, num_classes_for  # noqa: E402
from naca.granularity import LEVELS, build_unit_map, severity_of  # noqa: E402
from naca.models import build_model, load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="WP2a -- severity calibration")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--n-probe", type=int, default=48)
    ap.add_argument("--tolerance", type=float, default=0.10, help="severity match band")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = args.collect_dir or sorted(cfg.paths.runs.glob("*_collect"))[-1]
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[wp2a] collect dir: {d}\n")

    model = build_model(cfg.arch, num_classes=num_classes_for(cfg.dataset), device=device)
    load_checkpoint(model, cfg.paths.models / f"{cfg.setup}_{cfg.arch}.pt")
    model.eval()
    index = build_component_index(
        model, arch=cfg.arch, expected_n=cfg.n_components, verbose=False
    )
    test_ds = load_dataset(cfg.dataset, cfg.paths.data, train=False, device=device, download=False)
    eval_ds = test_ds.subset(np.load(d / "eval_indices.npy"))
    x_eval, y_eval = preload_eval(eval_ds.images, eval_ds.labels, eval_ds.mean, eval_ds.std)
    clean = np.load(d / "clean_margins.npy")

    # ---- unit maps ---------------------------------------------------------
    umaps = {}
    print(f"{'level':<14} {'n_units':>8} {'unit size min/med/max':>26}")
    for lvl in LEVELS:
        um = build_unit_map(index, lvl, group_size=args.group_size)
        umaps[lvl] = um
        s = um.sizes()
        print(f"{lvl:<14} {um.n_units:>8} {s.min():>8}/{int(np.median(s))}/{s.max():<10}")
    print()

    # ---- reference severity: filters at alpha_train -------------------------
    k_ref = int(round(cfg.alpha_train * index.N))
    ref = severity_of(
        model, index, umaps["filter"], k_ref, x_eval, y_eval, clean,
        n_probe=args.n_probe, sub_batch=cfg.eval_sub_batch, seed=cfg.seed,
    )
    target = ref["mean_abs_delta"]
    print(f"[wp2a] REFERENCE  filter level, k={k_ref} (alpha={cfg.alpha_train})")
    print(f"        mean|delta margin| = {target:.4f}   signed {ref['mean_signed_delta']:+.4f}")
    print()

    # ---- sweep k at each level ---------------------------------------------
    rows, chosen = [], {}
    for lvl in LEVELS:
        um = umaps[lvl]
        if lvl == "filter":
            ks = sorted({max(1, int(round(f * um.n_units))) for f in
                         (0.02, 0.05, 0.075, 0.10, 0.15, 0.20)})
        else:
            # coarse levels: sweep small integer counts, they saturate fast
            cap = max(2, min(um.n_units - 1, 24))
            ks = sorted({max(1, int(round(v))) for v in np.unique(
                np.concatenate([np.arange(1, min(cap, 9)),
                                np.round(np.linspace(1, cap, 6))]))})
        print(f"[wp2a] {lvl} (n_units={um.n_units})")
        best, best_gap = None, float("inf")
        for k in ks:
            if k >= um.n_units:
                continue
            s = severity_of(
                model, index, um, k, x_eval, y_eval, clean,
                n_probe=args.n_probe, sub_batch=cfg.eval_sub_batch, seed=cfg.seed,
            )
            s["ratio_to_target"] = s["mean_abs_delta"] / target
            rows.append(s)
            gap = abs(np.log(max(s["mean_abs_delta"], 1e-9) / target))
            flag = ""
            if gap < best_gap:
                best_gap, best, flag = gap, s, "  <-- closest"
            print(
                f"      k={k:<4d} comp_frac={s['mean_component_fraction']:.4f}  "
                f"mean|d|={s['mean_abs_delta']:8.4f}  "
                f"ratio={s['ratio_to_target']:6.3f}{flag}"
            )
        chosen[lvl] = best
        within = abs(best["ratio_to_target"] - 1.0) <= args.tolerance
        print(
            f"      -> chosen k={best['k_units']}  ratio={best['ratio_to_target']:.3f}  "
            f"{'WITHIN' if within else 'OUTSIDE'} the +/-{args.tolerance:.0%} band"
        )
        if not within:
            print(
                "      NOTE: severity cannot be matched at this granularity -- the unit is\n"
                "            too coarse for the target damage to be reachable at any integer k.\n"
                "            Report the achieved ratio; do not pretend it is matched."
            )
        print()

    df = pd.DataFrame(rows)
    df.to_csv(d / "wp2a_severity_sweep.csv", index=False)
    out = {
        "target_mean_abs_delta": target,
        "reference": ref,
        "group_size": args.group_size,
        "tolerance": args.tolerance,
        "chosen": {k: v for k, v in chosen.items()},
        "unit_counts": {k: int(v.n_units) for k, v in umaps.items()},
    }
    (d / "wp2a_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 74)
    print("PLAN  severity-matched collection budget")
    print(f"    {'level':<14} {'n_units':>8} {'k':>5} {'comp frac':>10} {'severity ratio':>15}")
    for lvl in LEVELS:
        c = chosen[lvl]
        print(
            f"    {lvl:<14} {c['n_units']:>8} {c['k_units']:>5} "
            f"{c['mean_component_fraction']:>10.4f} {c['ratio_to_target']:>15.3f}"
        )
    print(f"\n[wp2a] wrote {d / 'wp2a_calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
