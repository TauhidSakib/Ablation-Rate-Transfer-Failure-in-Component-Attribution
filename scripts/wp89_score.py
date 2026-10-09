"""Score WP8 and WP9 against the WP8/WP9 pre-registration.

Thresholds are committed in advance, before either setup is trained. They are
hard-coded here, not passed in, so a run cannot quietly move them:

  C1 transfer   oracle rho strictly monotone DECREASING in alpha, AND
                pooled+aggregate beats single-alpha on RAW R2 at held-out alpha
  C2 aggregate  mean(w) rises >= 100x from single-alpha to pooled
  C3 coarse     exact NAF at the coarse level >= 0.30
  C4 concentr.  NAF(K=25) / NAF(K=100) >= 10, and both above the precision floor

A criterion is PASS or FAIL on the inequality alone. Where a setup fails, the
measured number is printed next to the threshold it missed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))

C1_MONOTONE_ALPHAS = (0.01, 0.05, 0.10, 0.20)
C2_MEAN_W_RATIO = 100.0
C3_NAF_COARSE = 0.30
C4_CONCENTRATION = 10.0

NA = float("nan")


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _isnan(v) -> bool:
    return not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v))


def naf_at_train(run: Path, tag: str, alpha_train: float):
    j = _load(run / ("naca_summary_" + tag + ".json"))
    if not j:
        return NA
    rows = [r for r in j["across_alpha"] if abs(r["alpha_test"] - alpha_train) < 1e-9]
    return rows[0]["naf_oc_mean"] if rows else NA


def collect_facts(run, label: str) -> dict:
    f: dict = {"label": label, "run": run.name if run else "-"}
    if run is None or not Path(run).exists():
        return f
    run = Path(run)

    alpha_train = 0.10
    cfgp = run / "config_resolved.yaml"
    if cfgp.exists():
        for line in cfgp.read_text(encoding="utf-8").split("\n"):
            if line.startswith("alpha_train:"):
                alpha_train = float(line.split(":")[1])
            elif line.startswith("n_components:"):
                f["N"] = int(line.split(":")[1])
            elif line.startswith("arch:"):
                f["arch"] = line.split(":", 1)[1].strip()
            elif line.startswith("dataset:"):
                f["dataset"] = line.split(":", 1)[1].strip()
            elif line.startswith("m_train:"):
                f["m_train"] = int(line.split(":")[1])
    f["alpha_train"] = alpha_train

    # ---- C1 ---------------------------------------------------------------
    wp1d = _load(run / "wp1d_summary.json") or {}
    orc = {round(float(r["alpha"]), 4): r for r in wp1d.get("oracle_rows", [])}
    f["oracle_rho"] = {a: orc.get(a, {}).get("oracle_rho", NA) for a in C1_MONOTONE_ALPHAS}
    f["var_fm"] = {a: orc.get(a, {}).get("target_var", NA) for a in C1_MONOTONE_ALPHAS}
    vals = [f["oracle_rho"][a] for a in C1_MONOTONE_ALPHAS]
    f["monotone"] = (
        all(not _isnan(v) for v in vals)
        and all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    )

    rows = wp1d.get("structure_rows", [])
    pool = set(orc)
    held = [r for r in rows if round(float(r["alpha_test"]), 4) not in pool]
    f["heldout_alphas"] = sorted(round(float(r["alpha_test"]), 4) for r in held)
    if held:
        f["ood_r2_single"] = sum(r["A_linear_single__r2"] for r in held) / len(held)
        f["ood_r2_pooled"] = sum(r["C_global_pooled__r2"] for r in held) / len(held)
    else:
        f["ood_r2_single"] = NA
        f["ood_r2_pooled"] = NA
    f["C1"] = bool(
        f["monotone"]
        and not _isnan(f["ood_r2_pooled"])
        and f["ood_r2_pooled"] > f["ood_r2_single"]
    )

    # ---- C2 ---------------------------------------------------------------
    ag = _load(run / "wp1b_aggregate.json") or {}
    ks = ag.get("kappa_single_alpha", NA)
    kp = ag.get("kappa_pooled", NA)
    f["mean_w_single"] = ks
    f["mean_w_pooled"] = kp
    if _isnan(ks) or _isnan(kp) or ks == 0:
        f["mean_w_ratio"] = NA
    else:
        f["mean_w_ratio"] = abs(kp) / abs(ks)
    f["C2"] = bool(not _isnan(f["mean_w_ratio"]) and f["mean_w_ratio"] >= C2_MEAN_W_RATIO)

    # ---- C3 ---------------------------------------------------------------
    w2 = _load(run / "wp2b_exhaustive.json") or {}
    blk = w2.get("block") or {}
    f["naf_coarse"] = blk.get("exact_naf", NA)
    f["n_coarse_units"] = blk.get("n_units", NA)
    f["C3"] = bool(not _isnan(f["naf_coarse"]) and f["naf_coarse"] >= C3_NAF_COARSE)

    # ---- C4 ---------------------------------------------------------------
    n25 = naf_at_train(run, "Kcm25", alpha_train)
    n100 = naf_at_train(run, "Kcm100", alpha_train)
    pf = _load(run / "precision_floor.json") or {}
    floor = pf.get("floor_fraction_of_residual", NA)
    # A floor outside [0,1) is a wrong-checkpoint artifact, not a floor.
    if _isnan(floor) or not (0.0 <= floor < 1.0):
        floor = NA
    f["naf25"] = n25
    f["naf100"] = n100
    f["floor"] = floor
    f["concentration"] = NA if (_isnan(n100) or n100 <= 0 or _isnan(n25)) else n25 / n100
    above = (
        not _isnan(floor)
        and not _isnan(n25)
        and not _isnan(n100)
        and n25 > floor
        and n100 > floor
    )
    f["both_above_floor"] = above
    f["C4"] = bool(
        above and not _isnan(f["concentration"]) and f["concentration"] >= C4_CONCENTRATION
    )
    return f


def fmt(v, spec: str) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return v
    if _isnan(v):
        return "-"
    if spec == "d":
        return str(int(v))
    if spec == "s":
        return str(v)
    return format(float(v), spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed1", type=Path, default=ROOT / "runs" / "YOUR_SEED1_COLLECT_DIR")
    ap.add_argument("--wp8", type=Path, default=None)
    ap.add_argument("--wp9", type=Path, default=None)
    a = ap.parse_args()

    def latest(pat):
        c = sorted((ROOT / "runs").glob(pat))
        return c[-1] if c else None

    setups = [
        collect_facts(a.seed1, "CIFAR-10 ResNet-18"),
        collect_facts(a.wp8 or latest("*_B_collect"), "CIFAR-100 ResNet-18"),
        collect_facts(a.wp9 or latest("*_C_collect"), "CIFAR-10 ViT-Tiny"),
    ]

    w, c = 30, 24
    total = w + 3 * c
    print("=" * total)
    print("WP8 / WP9 -- three setups side by side")
    print("=" * total)
    line = " " * w
    for s in setups:
        line += f"{s['label'][:c - 1]:>{c}}"
    print(line)

    def row(name, key, spec=".4f", sub=None):
        ln = f"{name:<{w}}"
        for s in setups:
            v = s.get(key, NA)
            if sub is not None:
                v = v.get(sub, NA) if isinstance(v, dict) else NA
            ln += f"{fmt(v, spec):>{c}}"
        print(ln)

    dash = "-" * total
    print(dash)
    row("architecture", "arch", "s")
    row("dataset", "dataset", "s")
    row("N components", "N", "d")
    row("m_train (main pool)", "m_train", "d")
    print(dash)
    for al in C1_MONOTONE_ALPHAS:
        row(f"oracle rho @ alpha={al:g}", "oracle_rho", ".4f", sub=al)
    row("monotone decreasing", "monotone", "s")
    print(dash)
    for al in C1_MONOTONE_ALPHAS:
        row(f"Var(f_M) @ alpha={al:g}", "var_fm", ".3f", sub=al)
    print(dash)
    row("mean(w) single-alpha", "mean_w_single", ".3e")
    row("mean(w) pooled", "mean_w_pooled", ".3e")
    row("  ratio", "mean_w_ratio", ".1f")
    print(dash)
    row("OOD raw R2 single-alpha", "ood_r2_single", "+.3f")
    row("OOD raw R2 pooled+aggr", "ood_r2_pooled", "+.3f")
    print(dash)
    row("NAF coarse (exact)", "naf_coarse", ".4f")
    row("  coarse units", "n_coarse_units", "d")
    row("NAF(K=25)", "naf25", ".5f")
    row("NAF(K=100)", "naf100", ".5f")
    row("  concentration ratio", "concentration", ".1f")
    row("precision floor", "floor", ".6f")
    row("  both above floor", "both_above_floor", "s")

    print()
    print("=" * total)
    print("PRE-REGISTERED CRITERIA (thresholds fixed before any number existed)")
    print("=" * total)
    crit = [
        ("C1 transfer", "C1", "oracle rho strictly decreasing AND pooled > single on raw R2 OOD"),
        ("C2 aggregate", "C2", f"mean(w) ratio >= {C2_MEAN_W_RATIO:g}x"),
        ("C3 coarse non-additivity", "C3", f"NAF_coarse >= {C3_NAF_COARSE}"),
        ("C4 concentration", "C4", f"NAF(25)/NAF(100) >= {C4_CONCENTRATION:g}, both above floor"),
    ]
    for name, key, desc in crit:
        ln = f"{name:<{w}}"
        for s in setups:
            v = s.get(key)
            ln += f"{('PASS' if v else 'FAIL' if v is not None else '-'):>{c}}"
        print(ln)
        print("    " + desc)

    out = ROOT / "results" / "wp89_score.json"
    out.write_text(json.dumps(setups, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
