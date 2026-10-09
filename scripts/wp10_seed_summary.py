"""Cross-seed summary of the main result over all five ResNet-18 / CIFAR-10 seeds.

Budget is NOT pooled silently: seeds 1-2 were collected at m = 50,000 and seeds
3-5 at m = 25,000, so every table carries the budget column and the aggregate
statistics are reported both over all five and over the matched-budget subset.

For the transfer result we report an effect size as well as the raw values:
Cohen's d_z on the paired within-seed drop in oracle rho from alpha = 0.01 to
alpha = 0.20, which is the comparison the monotonicity criterion is about.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
ALPHAS = (0.01, 0.05, 0.10, 0.20)

# (label, run-dir glob, m_train)
SEEDS = [
    ("seed 1", "20260824_015507_A_collect", 50000),
    ("seed 2", "20260826_124436_A-s1_collect", 50000),
    ("seed 3", "*_A-s2_collect", 25000),
    ("seed 4", "*_A-s3_collect", 25000),
    ("seed 5", "*_A-s4_collect", 25000),
]


def load(pattern: str):
    if "*" in pattern:
        c = sorted((ROOT / "runs").glob(pattern))
        return c[-1] if c else None
    p = ROOT / "runs" / pattern
    return p if p.exists() else None


def _j(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def naf_at(run: Path, tag: str, alpha=0.10):
    j = _j(run / f"naca_summary_{tag}.json")
    if not j:
        return float("nan")
    at = [r for r in j["across_alpha"] if abs(r["alpha_test"] - alpha) < 1e-9]
    return at[0]["naf_oc_mean"] if at else float("nan")


def stats(vals):
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan")), 0
    mean = sum(v) / n
    if n < 2:
        return mean, float("nan"), (float("nan"), float("nan")), n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))
    # t(0.975, df=n-1); n=5 -> 2.776, n=4 -> 3.182, n=3 -> 4.303, n=2 -> 12.706
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.776)
    half = tcrit * sd / math.sqrt(n)
    return mean, sd, (mean - half, mean + half), n


def fmt(v, spec=".4f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return format(v, spec)


def main() -> int:
    rows = []
    for label, pat, m in SEEDS:
        d = load(pat)
        r = {"seed": label, "m": m, "dir": d.name if d else "-"}
        if d is None:
            rows.append(r)
            continue
        w = _j(d / "wp1d_summary.json") or {}
        orc = {round(float(x["alpha"]), 4): x["oracle_rho"] for x in w.get("oracle_rows", [])}
        r["rho"] = {a: orc.get(a, float("nan")) for a in ALPHAS}
        vals = [r["rho"][a] for a in ALPHAS]
        r["monotone"] = (all(not math.isnan(v) for v in vals)
                         and all(vals[i] > vals[i + 1] for i in range(3)))
        r["drop"] = (vals[0] - vals[3]) if not any(math.isnan(v) for v in vals) else float("nan")
        b = (_j(d / "wp2b_exhaustive.json") or {}).get("block") or {}
        r["naf_coarse"] = b.get("exact_naf", float("nan"))
        pf = (_j(d / "precision_floor.json") or {}).get("floor_fraction_of_residual", float("nan"))
        r["floor"] = pf if (isinstance(pf, (int, float)) and 0.0 <= pf < 1.0) else float("nan")
        r["naf25"] = naf_at(d, "Kcm25")
        r["naf100"] = naf_at(d, "Kcm100")
        rows.append(r)

    W = 100
    print("=" * W)
    print("MAIN RESULT ACROSS FIVE SEEDS -- ResNet-18 / CIFAR-10")
    print("=" * W)
    print("  Budgets differ: seeds 1-2 at m=50,000, seeds 3-5 at m=25,000. Not pooled silently.")
    print()
    hdr = (f"  {'seed':<8}{'m':>8}" + "".join(f"{'rho@' + format(a, 'g'):>10}" for a in ALPHAS)
           + f"{'monotone':>10}{'drop':>8}")
    print(hdr)
    print("  " + "-" * (W - 4))
    for r in rows:
        if "rho" not in r:
            print(f"  {r['seed']:<8}{r['m']:>8}   (missing)")
            continue
        line = f"  {r['seed']:<8}{r['m']:>8}"
        for a in ALPHAS:
            line += f"{fmt(r['rho'][a]):>10}"
        line += f"{('yes' if r['monotone'] else 'NO'):>10}{fmt(r['drop'], '.4f'):>8}"
        print(line)

    print()
    print(f"  {'':<8}{'':>8}" + "".join(f"{'':>10}" for _ in ALPHAS))
    for name, sel in (("all five", rows), ("m=25k only", [r for r in rows if r["m"] == 25000])):
        if not all("rho" in r for r in sel):
            continue
        line = f"  {name:<8}{'':>8}"
        for a in ALPHAS:
            mean, sd, ci, n = stats([r["rho"][a] for r in sel])
            line += f"{fmt(mean):>10}"
        print(line + f"   <- mean (n={len(sel)})")
        line = f"  {'  sd':<8}{'':>8}"
        for a in ALPHAS:
            mean, sd, ci, n = stats([r["rho"][a] for r in sel])
            line += f"{fmt(sd):>10}"
        print(line)

    print()
    print("  95% CI on oracle rho (all five seeds, t-interval, df=4):")
    for a in ALPHAS:
        mean, sd, ci, n = stats([r["rho"][a] for r in rows if "rho" in r])
        print(f"    alpha={a:<6g} mean {fmt(mean)}  sd {fmt(sd)}  "
              f"95% CI [{fmt(ci[0])}, {fmt(ci[1])}]   n={n}")

    # ---- effect size for the transfer result --------------------------------
    drops = [r["drop"] for r in rows if "drop" in r and not math.isnan(r["drop"])]
    mean, sd, ci, n = stats(drops)
    dz = mean / sd if sd and not math.isnan(sd) and sd > 0 else float("nan")
    print()
    print("=" * W)
    print("EFFECT SIZE -- transfer result")
    print("=" * W)
    print("  Paired within-seed drop in ORACLE rho from alpha=0.01 to alpha=0.20.")
    print(f"    per seed : {', '.join(fmt(d, '.4f') for d in drops)}")
    print(f"    mean     : {fmt(mean)}   sd {fmt(sd)}   95% CI [{fmt(ci[0])}, {fmt(ci[1])}]  n={n}")
    print(f"    Cohen's d_z = mean/sd = {fmt(dz, '.2f')}")
    nmono = sum(1 for r in rows if r.get("monotone"))
    print(f"    oracle rho strictly monotone decreasing on {nmono} of {len(rows)} seeds")

    # ---- NAF ---------------------------------------------------------------
    print()
    print("=" * W)
    print("NON-ADDITIVE FRACTION")
    print("=" * W)
    print(f"  {'seed':<8}{'m':>8}{'NAF_coarse':>13}{'NAF(K=25)':>12}{'xfloor':>9}"
          f"{'NAF(K=100)':>12}{'xfloor':>9}{'ratio':>8}{'floor':>11}")
    for r in rows:
        if "naf25" not in r:
            continue
        f_ = r["floor"]
        x25 = r["naf25"] / f_ if f_ and not math.isnan(f_) else float("nan")
        x100 = r["naf100"] / f_ if f_ and not math.isnan(f_) else float("nan")
        ratio = (r["naf25"] / r["naf100"]) if r["naf100"] and r["naf100"] > 0 else float("nan")
        print(f"  {r['seed']:<8}{r['m']:>8}{fmt(r['naf_coarse']):>13}"
              f"{fmt(r['naf25'], '.5f'):>12}{fmt(x25, '.1f'):>9}"
              f"{fmt(r['naf100'], '.5f'):>12}{fmt(x100, '.1f'):>9}"
              f"{fmt(ratio, '.1f'):>8}{fmt(r['floor'], '.6f'):>11}")

    print()
    for key, label in (("naf_coarse", "NAF_coarse"), ("naf25", "NAF(K=25)"),
                       ("naf100", "NAF(K=100)")):
        mean, sd, ci, n = stats([r[key] for r in rows if key in r])
        print(f"  {label:<12} mean {fmt(mean, '.5f')}  sd {fmt(sd, '.5f')}  "
              f"95% CI [{fmt(ci[0], '.5f')}, {fmt(ci[1], '.5f')}]  n={n}")

    out = ROOT / "results" / "wp10_seed_summary.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
