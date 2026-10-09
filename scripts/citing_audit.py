"""Full citing-works pull on Shah et al. 2024 (COAR), with abstract-level triage.

Paginates the Semantic Scholar citations endpoint with backoff, then flags any
citing work whose title or abstract touches the three things that would make it a
competitor rather than a neighbour: non-additive / higher-order component
attribution, the sampling design itself, or identifiability.
"""
from __future__ import annotations
import json, re, time, urllib.request, urllib.error
from pathlib import Path

BASE = "https://api.semanticscholar.org/graph/v1"
PAPER = "arXiv:2404.11534"
FIELDS = "title,abstract,year,venue,externalIds,url,citationCount"
OUT = Path(r"D:\naca\results\citing_audit.json")

RISK = {
    "higher-order/non-additive": r"higher[- ]order|non[- ]?additive|interaction|second[- ]order|synerg",
    "sampling design":           r"ablation rate|mask distribut|sampling (design|protocol|scheme)|subset size|cardinalit|stratif",
    "identifiability":           r"identifiab|collinear|null space|rank[- ]deficien|degenerat|unidentif|gauge",
    "surrogate class":           r"surrogate|datamodel|kolmogorov|kan\b|spline|polynomial|fourier|walsh|m\u00f6bius|mobius",
}

def get(url: str, tries: int = 8):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "naca-citing-audit/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = min(90, 5 * 2 ** k)
                print(f"    HTTP {e.code}; retry {k+1}/{tries} in {wait}s")
                time.sleep(wait); continue
            raise
        except Exception as e:
            wait = min(90, 5 * 2 ** k)
            print(f"    {type(e).__name__}; retry {k+1}/{tries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"gave up on {url}")

def main() -> int:
    papers, offset = [], 0
    while True:
        url = f"{BASE}/paper/{PAPER}/citations?fields={FIELDS}&limit=100&offset={offset}"
        print(f"[pull] offset {offset}")
        j = get(url)
        batch = [c["citingPaper"] for c in j.get("data", [])]
        papers += batch
        if "next" not in j or not batch:
            break
        offset = j["next"]
        time.sleep(3)

    print(f"\n[pull] {len(papers)} citing works\n")
    flagged = []
    for p in papers:
        text = f"{p.get('title') or ''} {p.get('abstract') or ''}".lower()
        hits = [k for k, pat in RISK.items() if re.search(pat, text)]
        if not p.get("abstract"):
            hits.append("NO ABSTRACT INDEXED")
        if hits:
            flagged.append({**p, "risk_tags": hits})

    flagged.sort(key=lambda p: -len([t for t in p["risk_tags"] if t != "NO ABSTRACT INDEXED"]))
    print("=" * 78)
    print(f"FLAGGED: {len(flagged)} of {len(papers)}")
    print("=" * 78)
    for p in flagged:
        real = [t for t in p["risk_tags"] if t != "NO ABSTRACT INDEXED"]
        print(f"\n[{len(real)} tags] {p.get('year')}  {p.get('title')}")
        print(f"    venue : {p.get('venue') or '-'}")
        print(f"    ids   : {p.get('externalIds')}")
        print(f"    tags  : {', '.join(p['risk_tags'])}")

    OUT.write_text(json.dumps({"n_total": len(papers), "all": papers, "flagged": flagged},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
