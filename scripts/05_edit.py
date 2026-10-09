"""Phases 8 and 9 -- COAR-EDIT baseline, then NACA-EDIT, verified by real ablation.

Two tasks from the base paper, both on Setup A:

* ``--task fix``    (base paper 5.1) -- raise the margin on a small set of
  mispredicted examples without disturbing the rest.
* ``--task forget`` (base paper 5.2) -- suppress one class while preserving the
  others.

COAR-EDIT runs **first** and must reproduce the base paper qualitatively before
NACA-EDIT is believed. Both are then measured the same way: the selected
components are really ablated in the model and the true ``(epsilon, delta)`` from
Definition 3 is read off. Surrogate-predicted effects are search signal only and
are never reported as results (CLAUDE.md 2.5, 8).

The interaction model used here is fitted on a **shared** pool, because greedy
selection has to compare the same component index across examples.
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

from naca.coar import CoarModel  # noqa: E402
from naca.collect import preload_eval  # noqa: E402
from naca.components import build_component_index  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.datasets import load_dataset, num_classes_for  # noqa: E402
from naca.editing import (  # noqa: E402
    coar_edit_select,
    fdr_candidate_filter,
    naca_edit_select,
)
from naca.interaction import InteractionModel, fit_interaction  # noqa: E402
from naca.masks import MaskStore  # noqa: E402
from naca.metrics import edit_epsilon_delta  # noqa: E402
from naca.models import build_model, load_checkpoint  # noqa: E402
from naca.screening import shared_pool  # noqa: E402


def latest_run(runs: Path, suffix: str) -> Path:
    cands = sorted(p for p in runs.glob(f"*_{suffix}") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"no run directory matching *_{suffix} under {runs}")
    return cands[-1]


def build_splits(task: str, labels: np.ndarray, clean: np.ndarray, cfg, rng) -> tuple:
    """Target / reference example indices, and the edit direction."""
    if task == "fix":
        wrong = np.nonzero(clean < 0)[0]
        if wrong.size == 0:
            raise RuntimeError("no mispredicted eval examples to fix")
        n_t = min(25, wrong.size)
        target = rng.choice(wrong, size=n_t, replace=False)
        rest = np.setdiff1d(np.arange(len(clean)), target)
        reference = rng.choice(rest, size=min(200, rest.size), replace=False)
        return np.sort(target), np.sort(reference), "increase"

    if task == "forget":
        cls = cfg.backdoor.target_class
        target = np.nonzero(labels == cls)[0]
        rest = np.nonzero(labels != cls)[0]
        reference = rng.choice(rest, size=min(300, rest.size), replace=False)
        return np.sort(target), np.sort(reference), "suppress"

    raise ValueError(f"unknown task {task!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="NACA Phases 8-9 -- editing")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--task", choices=["fix", "forget"], default="forget")
    ap.add_argument("--k-grid", type=str, default="2,4,8,16,32,64")
    ap.add_argument("--lambda-edit", type=float, default=None)
    ap.add_argument("--no-fdr", action="store_true")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--refit", action="store_true", help="refit the shared-pool model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.lambda_edit is not None:
        cfg.lambda_edit = args.lambda_edit
    if args.no_fdr:
        cfg.fdr_filter = False
    d = Path(args.collect_dir) if args.collect_dir else latest_run(cfg.paths.runs, "collect")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    k_grid = [int(x) for x in args.k_grid.split(",")]
    print(f"[edit] collect dir: {d}  task={args.task}  fdr={cfg.fdr_filter}")

    # ---- model and eval data (needed for the REAL ablation) ----------------
    model = build_model(cfg.arch, num_classes=num_classes_for(cfg.dataset), device=device)
    load_checkpoint(model, cfg.paths.models / f"{cfg.setup}_{cfg.arch}.pt")
    model.eval()
    index = build_component_index(
        model, arch=cfg.arch, expected_n=cfg.n_components, verbose=False
    )
    test_ds = load_dataset(cfg.dataset, cfg.paths.data, train=False, device=device, download=False)
    eval_idx = np.load(d / "eval_indices.npy")
    eval_ds = test_ds.subset(eval_idx)
    x_eval, y_eval = preload_eval(eval_ds.images, eval_ds.labels, eval_ds.mean, eval_ds.std)
    labels = eval_ds.labels.cpu().numpy()
    clean = np.load(d / "clean_margins.npy")

    rng = np.random.default_rng(cfg.seed)
    target, reference, direction = build_splits(args.task, labels, clean, cfg, rng)
    print(
        f"[edit] |S_T|={target.size} |S_R|={reference.size} direction={direction}  "
        f"clean margin on S_T: {clean[target].mean():+.3f}"
    )

    # ---- Stage-1 attributions + a SHARED-pool interaction model ------------
    coar = CoarModel.load(d / "W.npz")
    S = np.load(d / "screen_scores.npy")
    pool = shared_pool(coar.W, S, cfg.K, mode=cfg.screening)
    print(f"[edit] shared pool: K={pool.size}")

    model_path = d / "naca_model_shared.npz"
    if model_path.exists() and not args.refit:
        naca = InteractionModel.load(model_path)
        print("[edit] loaded cached shared-pool interaction model")
    else:
        masks = MaskStore.load(d / "masks_train.npz")
        Y = np.asarray(np.load(d / "Y_train.npy", mmap_mode="r"))
        lin = np.asarray(np.load(d / "lin_pred_train.npy", mmap_mode="r"))
        naca = fit_interaction(
            masks, Y, lin, pool, coar.W, coar.b,
            r_U=cfg.r_U, r_P=cfg.r_P, lambda_fm=cfg.lambda_fm,
            freeze_linear=cfg.freeze_linear, lr=cfg.fm_lr, epochs=cfg.fm_epochs,
            batch_size=cfg.fm_batch_size, group_size=args.group_size,
            device=cfg.device, seed=cfg.seed,
        )
        naca.save(model_path)

    # ---- candidate pool filter ---------------------------------------------
    candidates = None
    if cfg.fdr_filter:
        candidates, _ = fdr_candidate_filter(coar.W, target, reference, q=cfg.fdr_q)

    # ---- REAL ablation measurement -----------------------------------------
    from naca.ablation import Ablator
    from naca.collect import _forward_margins

    ablator = Ablator(model, index)
    buf = torch.empty(int(x_eval.shape[0]), dtype=torch.float32, device=device)

    def measure(components):
        if components is None or len(components) == 0:
            return clean[target], clean[reference]
        v = torch.ones(index.N, device=device)
        v[torch.as_tensor(np.asarray(components), device=device, dtype=torch.long)] = 0.0
        with ablator.ablated(v):
            _forward_margins(model, x_eval, y_eval, buf, cfg.eval_sub_batch, torch.bfloat16)
        m = buf.cpu().numpy()
        return m[target], m[reference]

    # ---- sweep k for both methods -------------------------------------------
    rows = []
    for k in k_grid:
        sel_coar = coar_edit_select(
            coar.W, target, reference, k, direction=direction, candidates=candidates
        )
        sel_naca = naca_edit_select(
            naca.w_pool, naca.A, pool, target, reference, k,
            direction=direction, lambda_edit=cfg.lambda_edit,
            gain_tol=cfg.gain_tol, candidates=candidates,
        )
        for sel in (sel_coar, sel_naca):
            ta, ra = measure(sel.components)
            eff = edit_epsilon_delta(
                clean[target], ta, clean[reference], ra, n_edit=len(sel)
            )
            rows.append(
                {
                    "task": args.task,
                    "method": sel.method,
                    "k_requested": k,
                    "k_selected": len(sel),
                    **eff.to_dict(),
                    "target_margin_after": float(ta.mean()),
                    "target_margin_before": float(clean[target].mean()),
                }
            )
            print(
                f"[edit] k={k:<3d} {sel.method:<10s} "
                f"delta={eff.delta:7.4f} epsilon={eff.epsilon:7.4f} "
                f"(signed target {eff.delta_signed:+7.4f}) "
                f"margin {clean[target].mean():+.3f} -> {ta.mean():+.3f}"
            )

    df = pd.DataFrame(rows)
    out = d / f"edit_{args.task}.csv"
    df.to_csv(out, index=False)

    # ---- verdict -------------------------------------------------------------
    print(f"\n{'-' * 74}")
    piv = df.pivot_table(
        index="k_requested", columns="method", values=["delta", "epsilon"]
    )
    print(piv.to_string())

    wins = 0
    for k in k_grid:
        a = df[(df.k_requested == k) & (df.method == "naca-edit")]
        b = df[(df.k_requested == k) & (df.method == "coar-edit")]
        if a.empty or b.empty:
            continue
        # better = larger effect on the target per unit collateral on the reference
        ra = a.delta.iloc[0] / max(a.epsilon.iloc[0], 1e-9)
        rb = b.delta.iloc[0] / max(b.epsilon.iloc[0], 1e-9)
        wins += int(ra > rb)
    print(
        f"\n[H4] NACA-EDIT achieved a better delta/epsilon ratio at "
        f"{wins}/{len(k_grid)} budgets"
    )
    (d / f"edit_{args.task}_summary.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "direction": direction,
                "n_target": int(target.size),
                "n_reference": int(reference.size),
                "fdr_filter": cfg.fdr_filter,
                "lambda_edit": cfg.lambda_edit,
                "K": int(pool.size),
                "naca_wins": wins,
                "n_budgets": len(k_grid),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[edit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
