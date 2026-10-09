"""Phase 10 -- the required ablation table (CLAUDE.md section 7).

Each row re-runs `04_fit_naca.py` with one knob changed and collects the headline
numbers into a single CSV. Rows are driven as subprocesses so every arm goes
through exactly the same tested code path as the main result -- an ablation that
runs through a private shortcut is not an ablation of the thing you shipped.

Arms:

* ``K``            {25, 50, 100, 200}   -- does NAF saturate with pool size?
* ``rank``         (r_U, r_P) in {2,4,8,16} -- how much interaction structure exists?
* ``psd``          A = UU' vs UU' - PP'  -- validates the core design choice
* ``screening``    marginal / residual / union -- validates section 2.3
* ``freeze``       freeze_linear true / false
* ``m``            {50k, 25k, 10k}      -- sample efficiency

``--only`` selects a subset; ``--dry-run`` prints the plan without running it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from naca.config import load_config  # noqa: E402

PY = sys.executable


def latest_run(runs: Path, suffix: str) -> Path:
    cands = sorted(p for p in runs.glob(f"*_{suffix}") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"no run directory matching *_{suffix} under {runs}")
    return cands[-1]


def build_plan(cfg) -> list[dict]:
    """Every arm as (name, tag, extra CLI flags)."""
    plan: list[dict] = []

    for K in (25, 50, 100, 200):
        plan.append({"arm": "K", "value": K, "tag": f"K{K}", "flags": ["--K", str(K)]})

    # Capacity-matched sweep. The fixed-rank K arm is NOT nested: with r_U+r_P
    # held at 16, a larger pool spreads the same rank-16 matrix over more
    # components, so expressiveness per pair collapses (rank/K = 64% at K=25 but
    # 8% at K=200). That arm answers "how much can you see at a fixed budget";
    # this one holds rank/K ~ 0.6 so it answers "how much structure is there".
    # Both belong in the paper -- they are different questions.
    for K in (25, 50, 100, 200):
        r = max(1, int(round(0.3 * K)))  # r_U = r_P = 0.3K  ->  (r_U+r_P)/K = 0.6
        plan.append(
            {
                "arm": "K_capacity_matched",
                "value": f"K={K}, r={r}+{r}",
                "tag": f"Kcm{K}",
                "flags": ["--K", str(K), "--r-U", str(r), "--r-P", str(r)],
            }
        )

    for r in (2, 4, 8, 16):
        plan.append(
            {
                "arm": "rank",
                "value": f"{r}+{r}",
                "tag": f"r{r}",
                "flags": ["--r-U", str(r), "--r-P", str(r)],
            }
        )

    # PSD arm: same total parameter budget, negative block removed.
    plan.append(
        {
            "arm": "psd",
            "value": "PSD (A=UU')",
            "tag": "psd",
            "flags": ["--r-U", str(cfg.r_U + cfg.r_P), "--r-P", "0"],
        }
    )
    plan.append(
        {
            "arm": "psd",
            "value": "indefinite (A=UU'-PP')",
            "tag": "indef",
            "flags": ["--r-U", str(cfg.r_U), "--r-P", str(cfg.r_P)],
        }
    )

    for mode in ("marginal", "residual", "union"):
        plan.append(
            {
                "arm": "screening",
                "value": mode,
                "tag": f"scr_{mode}",
                "flags": ["--screening", mode],
            }
        )

    for fl in ("true", "false"):
        plan.append(
            {
                "arm": "freeze",
                "value": fl,
                "tag": f"freeze_{fl}",
                "flags": ["--freeze-linear", fl],
            }
        )
    return plan


def read_result(d: Path, tag: str, alpha_train: float) -> dict:
    path = d / f"naca_summary_{tag}.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = blob.get("across_alpha", [])
    at = next((r for r in rows if r["alpha_test"] == alpha_train), None)
    ood = [r for r in rows if r["alpha_test"] > alpha_train]
    out = {
        "n_params_per_example": blob.get("fit_info", {}).get("n_params_per_example"),
        "fit_seconds": blob.get("fit_info", {}).get("seconds"),
        "pool_distinct": blob.get("pool_distinct_components"),
        "frac_indefinite": blob.get("fraction_examples_indefinite"),
    }
    if at:
        out.update(
            {
                "naf_oc_at_train": at["naf_oc_mean"],
                "naf_raw_at_train": at["naf_mean"],
                "r2_oc_lin": at["r2_oc_lin"],
                "r2_oc_naca": at["r2_oc_naca"],
                "wilcoxon_p": at["wilcoxon_p_oc"],
            }
        )
    if ood:
        out["naf_oc_mean_above_train"] = sum(r["naf_oc_mean"] for r in ood) / len(ood)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="NACA Phase 10 -- ablation table")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--only", type=str, default="", help="comma-separated arm names")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = Path(args.collect_dir) if args.collect_dir else latest_run(cfg.paths.runs, "collect")
    plan = build_plan(cfg)
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        plan = [p for p in plan if p["arm"] in keep]

    print(f"[abl] collect dir: {d}")
    print(f"[abl] {len(plan)} arms: {[p['tag'] for p in plan]}")
    if args.dry_run:
        return 0

    rows = []
    t0 = time.time()
    for i, item in enumerate(plan, 1):
        cmd = [
            PY, "-u", str(ROOT / "scripts" / "04_fit_naca.py"),
            "--config", str(args.config),
            "--collect-dir", str(d),
            "--tag", item["tag"],
            "--group-size", str(args.group_size),
            "--skip-h3",
            *item["flags"],
        ]
        print(f"\n[abl] ({i}/{len(plan)}) {item['arm']}={item['value']} -> tag={item['tag']}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[abl] FAILED (exit {proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
            rows.append({"arm": item["arm"], "value": item["value"], "tag": item["tag"], "failed": True})
            continue
        res = read_result(d, item["tag"], cfg.alpha_train)
        rows.append({"arm": item["arm"], "value": item["value"], "tag": item["tag"], "failed": False, **res})
        print(
            f"[abl]   NAF_oc={res.get('naf_oc_at_train', float('nan')):+.4f}  "
            f"params/example={res.get('n_params_per_example')}  "
            f"fit={res.get('fit_seconds', 0) / 60:.1f}m"
        )

    df = pd.DataFrame(rows)
    out = d / "ablation_table.csv"
    df.to_csv(out, index=False)
    print(f"\n[abl] {len(rows)} arms in {(time.time() - t0) / 60:.1f} min -> {out}\n")
    cols = [c for c in ("arm", "value", "naf_oc_at_train", "r2_oc_naca", "n_params_per_example", "wilcoxon_p") if c in df]
    print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
