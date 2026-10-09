"""Phase 2 -- train the target model, and gate on test accuracy.

Gate (the design spec section 6): test accuracy ~= 91%, the base paper's number for
CIFAR-10 ResNet-18. Read as a *floor*: exceeding it is not a failure, so the
gate fails below ``target_acc - acc_tolerance`` and merely reports above
``target_acc + acc_tolerance``. Beating the base paper does not invalidate a
reproduction; falling short of it does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from naca.components import build_component_index  # noqa: E402
from naca.config import load_config, make_run_dir  # noqa: E402
from naca.datasets import load_dataset, make_backdoor_train_split, num_classes_for  # noqa: E402
from naca.models import (  # noqa: E402
    build_model,
    evaluate,
    save_checkpoint,
    train_classifier,
)


def backdoor_success_rate(model, test_ds, cfg) -> dict:
    """How often the trigger flips a non-target image to the target class."""
    from naca.datasets import GPUTensorDataset, apply_backdoor_patch

    bd = cfg.backdoor
    keep = test_ds.labels != bd.target_class
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    sub = test_ds.subset(idx)
    triggered = GPUTensorDataset(
        images=apply_backdoor_patch(sub.images, bd),
        labels=sub.labels,
        mean=sub.mean,
        std=sub.std,
        name="triggered",
    )
    model.eval()
    with torch.inference_mode():
        preds = []
        for lo in range(0, len(triggered), 1000):
            hi = min(lo + 1000, len(triggered))
            x = triggered.normalized(lo, hi)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds.append(model(x).float().argmax(1))
        preds = torch.cat(preds)
    asr = float((preds == bd.target_class).float().mean())
    clean_acc = evaluate(model, test_ds)
    return {"attack_success_rate": asr, "clean_test_acc": clean_acc}


def main() -> int:
    ap = argparse.ArgumentParser(description="NACA Phase 2 -- train target model")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "setup_a.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="checkpoint path override")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    ckpt = args.out or (cfg.paths.models / f"{cfg.setup}_{cfg.arch}.pt")
    if ckpt.exists() and not args.force:
        print(f"[train] checkpoint already exists: {ckpt}\n[train] pass --force to retrain")
        return 0

    run_dir = make_run_dir(cfg, "train")
    print(f"[train] run dir: {run_dir}")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---- data -----------------------------------------------------------
    t0 = time.time()
    train_ds = load_dataset(cfg.dataset, cfg.paths.data, train=True, device=device)
    test_ds = load_dataset(cfg.dataset, cfg.paths.data, train=False, device=device)
    if cfg.backdoor.enabled:
        train_ds = make_backdoor_train_split(train_ds, cfg.backdoor, seed=cfg.seed)
    print(
        f"[train] data ready in {time.time() - t0:.1f}s -- "
        f"train {len(train_ds)}, test {len(test_ds)}, "
        f"{(train_ds.nbytes() + test_ds.nbytes()) / 1024**2:.0f} MB resident on {device}"
    )

    # ---- model ----------------------------------------------------------
    model = build_model(cfg.arch, num_classes=num_classes_for(cfg.dataset), device=device)
    index = build_component_index(
        model, arch=cfg.arch, include_downsample=False, expected_n=cfg.n_components
    )
    (run_dir / "component_index.json").write_text(
        json.dumps(index.to_dict(), indent=2), encoding="utf-8"
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {cfg.arch}: {n_params / 1e6:.2f}M params, N={index.N} components")

    # ---- train ----------------------------------------------------------
    result = train_classifier(model, train_ds, test_ds, cfg.train, seed=cfg.seed)

    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    print(f"[train] peak VRAM {peak:.2f} GB, wall {result.seconds / 60:.1f} min")

    # ---- gate -----------------------------------------------------------
    target, tol = cfg.train.target_acc, cfg.train.acc_tolerance
    acc = result.test_acc
    floor = target - tol
    passed = acc >= floor
    print(f"\n{'-' * 66}")
    print(f"[gate] test accuracy ours={acc:.4f}  base paper={target:.4f}  floor={floor:.4f}")
    if not passed:
        print(f"[gate] FAILED -- {acc:.4f} < {floor:.4f}. Do not proceed to Phase 3.")
    elif acc > target + tol:
        print(f"[gate] PASSED (above the paper by {acc - target:+.4f}; a floor, not a target)")
    else:
        print("[gate] PASSED -- matches the base paper within tolerance")

    metrics = {
        "test_acc": acc,
        "train_acc": result.train_acc,
        "paper_test_acc": target,
        "gate_floor": floor,
        "gate_passed": bool(passed),
        "epochs": result.epochs,
        "seconds": result.seconds,
        "peak_vram_gb": peak,
        "n_components": index.N,
        "n_params": n_params,
        "history": result.history,
    }
    if cfg.backdoor.enabled:
        bd = backdoor_success_rate(model, test_ds, cfg)
        metrics["backdoor"] = bd
        print(f"[backdoor] attack success rate {bd['attack_success_rate']:.4f}")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if passed:
        save_checkpoint(model, ckpt, meta={"metrics": metrics, "config": cfg.to_dict()})
        print(f"[train] checkpoint -> {ckpt}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
