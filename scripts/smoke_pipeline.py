"""End-to-end plumbing check on synthetic counterfactuals.

Builds a miniature ``collect``-shaped run directory whose targets come from a
*known* linear + indefinite-quadratic model, so the downstream scripts can be
exercised without waiting on real collection -- and so their outputs can be
checked against a ground truth we control.

Because the truth here is genuinely non-additive, a correct pipeline must report
a clearly positive NAF. A near-zero NAF on this data is a bug, not a finding.

    python scripts/smoke_pipeline.py --out D:\\naca\\runs\\smoke_A_collect
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

from naca.components import LayerSpec  # noqa: E402
from naca.masks import MaskSpec, MaskStore  # noqa: E402

N_COMP = 320
K_TRUE = 24
E = 48
M_TRAIN = 12_000
M_TEST = 1_500
ALPHA_TRAIN = 0.10
ALPHAS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]


def synth_targets(store: MaskStore, W, b, pool, A_true) -> np.ndarray:
    X = store.unpack().astype(np.float32)
    Vp = X[:, pool]
    quad = np.einsum("mk,ekl,ml->me", Vp, A_true, Vp, optimize=True) * 0.5
    return (X @ W + b[None, :] + quad).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "smoke_A_collect")
    ap.add_argument("--quad-scale", type=float, default=0.6)
    args = ap.parse_args()
    d = args.out
    d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.4, size=(N_COMP, E)).astype(np.float32)
    b = rng.normal(0, 0.5, size=E).astype(np.float32)
    # bias the interacting components toward the "deep" end of the index so the
    # H3 depth stratification has something real to find
    pool = np.sort(rng.choice(np.arange(N_COMP // 2, N_COMP), size=K_TRUE, replace=False))
    A_true = np.zeros((E, K_TRUE, K_TRUE), dtype=np.float32)
    for e in range(E):
        U = rng.normal(0, 1.0, size=(K_TRUE, 3)).astype(np.float32)
        P = rng.normal(0, 1.0, size=(K_TRUE, 3)).astype(np.float32)
        M = U @ U.T - P @ P.T
        np.fill_diagonal(M, 0.0)
        A_true[e] = args.quad_scale * M

    train = MaskStore.generate(MaskSpec(N_COMP, ALPHA_TRAIN, M_TRAIN, 0, kind="train"))
    train.save(d / "masks_train.npz")
    Yt = synth_targets(train, W, b, pool, A_true)
    np.save(d / "Y_train.npy", Yt)
    share = float(np.mean(np.var(Yt, axis=0)))
    print(f"[smoke] train: m={M_TRAIN} N={N_COMP} E={E}  Var(y)={share:.3f}")

    for alpha in ALPHAS:
        tag = f"a{alpha:g}".replace(".", "p")
        st = MaskStore.generate(
            MaskSpec(N_COMP, alpha, M_TEST, 99, kind=f"test_{tag}")
        )
        st.save(d / f"masks_test_{tag}.npz")
        np.save(d / f"Y_test_{tag}.npy", synth_targets(st, W, b, pool, A_true))

    np.save(d / "eval_indices.npy", np.arange(E))
    np.save(d / "clean_margins.npy", rng.normal(2.0, 3.0, size=E).astype(np.float32))

    # a plausible layer partition so the H3 stratification has groups to use
    layers, cursor = [], 0
    for name, n in (
        ("conv1", 32),
        ("layer1.0.conv1", 32),
        ("layer2.0.conv1", 64),
        ("layer3.0.conv1", 64),
        ("layer4.0.conv1", 128),
    ):
        layers.append(
            LayerSpec(name, n, cursor, cursor + n, name.split(".")[0].replace("conv1", "stem"), False)
        )
        cursor += n
    (d / "component_index.json").write_text(
        json.dumps(
            {
                "arch": "smoke",
                "N": N_COMP,
                "layers": [
                    {
                        "name": s.name,
                        "n_filters": s.n_filters,
                        "start": s.start,
                        "stop": s.stop,
                        "depth_group": s.depth_group,
                        "is_downsample": s.is_downsample,
                    }
                    for s in layers
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(d / "true_pool.npy", pool)
    print(f"[smoke] wrote synthetic collect dir -> {d}")
    print(
        "[smoke] next:\n"
        f"  python scripts/03_fit_linear.py --config configs/smoke.yaml --collect-dir {d}\n"
        f"  python scripts/04_fit_naca.py   --config configs/smoke.yaml --collect-dir {d}\n"
        f"  python scripts/07_figures.py    --config configs/smoke.yaml --collect-dir {d}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
