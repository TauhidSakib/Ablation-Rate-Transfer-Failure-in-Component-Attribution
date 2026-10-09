"""Phase 3 -- collect component counterfactuals for the train and held-out sets.

Produces, under ``runs/<stamp>_<setup>_collect/``:

* ``masks_train.npz`` / ``Y_train.npy``    -- m masks at alpha_train
* ``masks_test_a<alpha>.npz`` / ``Y_test_a<alpha>.npy`` -- m_test masks per alpha
* ``clean_margins.npy``                    -- f_M(z, empty) baseline
* ``eval_indices.npy``                     -- which test images were used
* ``collect_metrics.json``                 -- throughput, VRAM, integrity checks

Every ``Y`` is a memmap that doubles as its own checkpoint, so an interrupted
run resumes from the last flushed mask. Re-running with the same ``--run-dir``
picks up where it stopped.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from naca.collect import (  # noqa: E402
    clean_margins,
    collect_counterfactuals,
    preload_eval,
    throughput_probe,
)
from naca.components import build_component_index  # noqa: E402
from naca.config import load_config, make_run_dir  # noqa: E402
from naca.datasets import load_dataset, num_classes_for  # noqa: E402
from naca.masks import MaskSpec, MaskStore  # noqa: E402
from naca.models import build_model, evaluate, load_checkpoint  # noqa: E402


def acquire_lock(run_dir: Path) -> Path:
    """Refuse to start if another collector is already working this run dir.

    Background shells that get torn down can leave their Python children
    running. Three orphaned collectors once ended up writing the same memmap
    concurrently: the data survived (every mask index is deterministic, so they
    all wrote identical values) but throughput collapsed from 7.6 to 1.9 masks/s
    while they fought for RAM and the GPU. A PID lock makes that impossible to
    repeat by accident.
    """
    import psutil

    lock = run_dir / "collect.lock"
    if lock.exists():
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
            pid = int(info["pid"])
        except (ValueError, KeyError, OSError):
            pid = -1
        if pid > 0 and psutil.pid_exists(pid):
            try:
                other = psutil.Process(pid)
                cmd = " ".join(other.cmdline())
            except psutil.Error:
                cmd = ""
            if "02_collect" in cmd:
                raise RuntimeError(
                    f"another collector is already running on this run dir "
                    f"(pid {pid}). Stop it first, or delete {lock} if that pid "
                    f"is stale."
                )
        print(f"[collect] clearing stale lock from pid {pid}")
    lock.write_text(
        json.dumps({"pid": os.getpid(), "started": time.time()}), encoding="utf-8"
    )
    return lock


def assert_disk(min_gb: float) -> None:
    """Hard rule 7: fail fast rather than dying at 80% completion."""
    free = shutil.disk_usage("D:\\").free / 1024**3
    if free < min_gb:
        raise RuntimeError(
            f"only {free:.1f} GB free on D:, need >= {min_gb:.0f} GB for this run"
        )
    print(f"[collect] disk OK: {free:.1f} GB free on D:")


def main() -> int:
    ap = argparse.ArgumentParser(description="NACA Phase 3 -- counterfactual collection")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--run-dir", type=Path, default=None, help="resume into this run dir")
    ap.add_argument("--m-train", type=int, default=None, help="override m_train")
    ap.add_argument("--probe-only", action="store_true", help="time 100 masks and exit")
    ap.add_argument("--skip-test-grid", action="store_true")
    ap.add_argument(
        "--extra-train-alphas",
        type=str,
        default="",
        help="comma-separated extra alpha_train values for the multi-alpha danger "
        "baseline, e.g. 0.05,0.20",
    )
    ap.add_argument(
        "--extra-train-m",
        type=int,
        default=20_000,
        help="masks per extra alpha_train set",
    )
    ap.add_argument("--only-extra", action="store_true", help="collect only the extra sets")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.m_train is not None:
        cfg.m_train = args.m_train
    assert_disk(min_gb=5.0)

    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(cfg, "collect")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[collect] run dir: {run_dir}")
    lock = acquire_lock(run_dir)
    atexit.register(lambda: lock.unlink(missing_ok=True))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or (cfg.paths.models / f"{cfg.setup}_{cfg.arch}.pt")

    # ---- model ----------------------------------------------------------
    model = build_model(cfg.arch, num_classes=num_classes_for(cfg.dataset), device=device)
    meta = load_checkpoint(model, ckpt)
    model.eval()
    index = build_component_index(
        model, arch=cfg.arch, include_downsample=False, expected_n=cfg.n_components
    )
    print(
        f"[collect] loaded {ckpt.name} "
        f"(test_acc={meta.get('metrics', {}).get('test_acc', float('nan')):.4f}), N={index.N}"
    )

    # ---- eval examples ---------------------------------------------------
    test_ds = load_dataset(cfg.dataset, cfg.paths.data, train=False, device=device, download=False)
    acc = evaluate(model, test_ds)
    print(f"[collect] sanity: reloaded model test accuracy {acc:.4f}")

    idx_path = run_dir / "eval_indices.npy"
    if idx_path.exists():
        eval_idx = np.load(idx_path)
    else:
        rng = np.random.default_rng(cfg.seed)
        eval_idx = np.sort(
            rng.choice(len(test_ds), size=cfg.n_eval_examples, replace=False)
        )
        np.save(idx_path, eval_idx)
    eval_ds = test_ds.subset(eval_idx)

    x_eval, y_eval = preload_eval(
        eval_ds.images, eval_ds.labels, eval_ds.mean, eval_ds.std
    )
    print(
        f"[collect] eval set preloaded: {len(eval_ds)} images, "
        f"{x_eval.numel() * 4 / 1024**2:.1f} MB resident (no DataLoader from here on)"
    )

    clean = clean_margins(
        model, x_eval, y_eval, sub_batch=cfg.eval_sub_batch, amp_dtype=torch.bfloat16
    )
    np.save(run_dir / "clean_margins.npy", clean)
    print(
        f"[collect] clean margins: mean={clean.mean():.3f} "
        f"min={clean.min():.3f} max={clean.max():.3f} "
        f"({(clean > 0).mean():.1%} correctly classified)"
    )

    # ---- throughput probe -------------------------------------------------
    probe_masks = MaskStore.generate(
        MaskSpec(cfg.n_components, cfg.alpha_train, 128, cfg.seed, kind="probe")
    )
    probe = throughput_probe(
        model,
        index,
        probe_masks,
        x_eval,
        y_eval,
        n_probe=100,
        sub_batch=cfg.eval_sub_batch,
        dtype_forward=cfg.dtype_forward,
    )
    total_masks = cfg.m_train + (0 if args.skip_test_grid else len(cfg.alpha_test_grid) * cfg.m_test)
    probe["total_masks_planned"] = total_masks
    probe["projected_total_minutes"] = total_masks / probe["masks_per_sec"] / 60.0
    print(
        f"[collect] probe: {probe['masks_per_sec']:.1f} masks/s "
        f"({probe['ms_per_mask']:.1f} ms/mask), peak VRAM {probe['peak_vram_gb']:.2f} GB\n"
        f"[collect] projected: {total_masks} masks -> "
        f"{probe['projected_total_minutes']:.0f} min"
    )
    (run_dir / "probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    if args.probe_only:
        return 0

    metrics: dict = {"probe": probe, "sets": {}}
    t_all = time.time()

    # ---- extra alpha_train sets for the multi-alpha danger baseline -------
    extra = [float(x) for x in args.extra_train_alphas.split(",") if x.strip()]
    for alpha in extra:
        tag = f"a{alpha:g}".replace(".", "p")
        spec = MaskSpec(
            cfg.n_components, alpha, args.extra_train_m, cfg.seed, kind=f"train_{tag}"
        )
        store = MaskStore.generate(spec)
        store.save(run_dir / f"masks_train_{tag}.npz")
        print(f"[collect] extra train set at alpha={alpha}, m={args.extra_train_m}")
        Y_x, _ = collect_counterfactuals(
            model,
            index,
            store,
            x_eval,
            y_eval,
            run_dir / f"Y_train_{tag}.npy",
            sub_batch=cfg.eval_sub_batch,
            dtype_forward=cfg.dtype_forward,
            checkpoint_every=cfg.checkpoint_every,
        )
        metrics["sets"][f"train_{tag}"] = {
            "alpha": alpha,
            "m": args.extra_train_m,
            "y_var_per_example_mean": float(Y_x.var(axis=0).mean()),
        }
    if args.only_extra:
        metrics["total_minutes"] = (time.time() - t_all) / 60.0
        (run_dir / "collect_metrics_extra.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(f"[collect] extra sets done in {metrics['total_minutes']:.1f} min")
        return 0

    # ---- training set -----------------------------------------------------
    train_spec = MaskSpec(cfg.n_components, cfg.alpha_train, cfg.m_train, cfg.seed, kind="train")
    train_path = run_dir / "masks_train.npz"
    if train_path.exists():
        # Resume: load rather than regenerate. Regenerating 50k masks costs both
        # time and a transient allocation this machine may not have to spare.
        train_masks = MaskStore.load(train_path)
        if train_masks.spec != train_spec:
            raise ValueError(
                f"saved mask store at {train_path} has spec {train_masks.spec}, "
                f"config wants {train_spec}. Move it aside or fix the config."
            )
        print("[collect] loaded existing train mask store")
    else:
        train_masks = MaskStore.generate(train_spec)
        train_masks.save(train_path)
    print(f"[collect] train masks: {train_masks}")
    print(f"[collect]   {json.dumps(train_masks.sanity_report())}")

    Y_train, _ = collect_counterfactuals(
        model,
        index,
        train_masks,
        x_eval,
        y_eval,
        run_dir / "Y_train.npy",
        sub_batch=cfg.eval_sub_batch,
        dtype_forward=cfg.dtype_forward,
        checkpoint_every=cfg.checkpoint_every,
    )
    metrics["sets"]["train"] = {
        "alpha": cfg.alpha_train,
        "m": cfg.m_train,
        "y_mean": float(Y_train.mean()),
        "y_std": float(Y_train.std()),
        "y_var_per_example_mean": float(Y_train.var(axis=0).mean()),
    }

    # ---- held-out grid ----------------------------------------------------
    if not args.skip_test_grid:
        for alpha in cfg.alpha_test_grid:
            tag = f"a{alpha:g}".replace(".", "p")
            spec = MaskSpec(cfg.n_components, alpha, cfg.m_test, cfg.seed, kind=f"test_{tag}")
            mpath = run_dir / f"masks_test_{tag}.npz"
            if mpath.exists():
                store = MaskStore.load(mpath)
            else:
                store = MaskStore.generate(spec)
                store.save(mpath)
            Y_t, _ = collect_counterfactuals(
                model,
                index,
                store,
                x_eval,
                y_eval,
                run_dir / f"Y_test_{tag}.npy",
                sub_batch=cfg.eval_sub_batch,
                dtype_forward=cfg.dtype_forward,
                checkpoint_every=cfg.checkpoint_every,
                log_every=cfg.m_test,
            )
            # Var(f_M) per alpha is the signal-to-noise diagnostic of the design spec 2.7
            metrics["sets"][tag] = {
                "alpha": alpha,
                "m": cfg.m_test,
                "y_mean": float(Y_t.mean()),
                "y_var_per_example_mean": float(Y_t.var(axis=0).mean()),
            }
            print(
                f"[collect] alpha={alpha:<6.3f} Var(f_M)={metrics['sets'][tag]['y_var_per_example_mean']:.4f} "
                f"mean={metrics['sets'][tag]['y_mean']:.3f}"
            )

    metrics["total_minutes"] = (time.time() - t_all) / 60.0
    metrics["n_eval_examples"] = int(len(eval_ds))
    metrics["N"] = index.N
    metrics["model_test_acc"] = acc
    (run_dir / "collect_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n[collect] ALL DONE in {metrics['total_minutes']:.1f} min -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
