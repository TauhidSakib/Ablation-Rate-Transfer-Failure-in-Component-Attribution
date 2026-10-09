"""Is the measured NAF above the numerical precision floor?

All counterfactuals are collected under bfloat16 autocast. bf16 carries 8 mantissa
bits, so a margin of order 10 carries roughly 0.02-0.03 of representation error.
That error is a *deterministic* function of the mask rather than resampling noise,
but neither surrogate can model it, so it lands in the residual of both -- and
unmodelable residual variance inflates the denominator of

    NAF = (R2_NACA - R2_lin) / (1 - R2_lin)

while leaving the numerator roughly unchanged. **It biases NAF toward zero.**

The measured NAF is 0.0005 of the residual variance. A back-of-envelope estimate
put the bf16 contribution at the same order, which would mean the null is partly a
precision artifact. That estimate is not good enough to rely on, so this script
measures the thing directly: recompute a sample of the *same* masks in fp32,
compare against the stored bf16 values, and express the discrepancy as a share of
the residual variance the interaction model is competing for.

Decision rule, fixed in advance:

  floor << NAF   -> the null stands; report the floor as a precision analysis
  floor ~ NAF    -> the null is not resolvable at bf16; recollect in fp32
  floor >> NAF   -> the null is an artifact and must be withdrawn

**This script does not refit anything.** 3,000 masks cannot identify a
3,904-coefficient model, so recomputing NAF directly from this sample is not
possible. Instead it measures the noise variance and *corrects* the NAF already
measured on the full 50,000-mask fit. The correction is exact under the standard
additive-noise model:

Write the measured target as y = y_true + e with e uncorrelated with what either
surrogate can express. Then each model's residual sum picks up sigma_e^2:

    NAF_meas = (SS_lin_true - SS_naca_true) / (SS_lin_true + sigma_e^2)
    NAF_true = (SS_lin_true - SS_naca_true) /  SS_lin_true

so, using SS_lin_meas = SS_lin_true + sigma_e^2,

    NAF_true = NAF_meas * SS_lin_meas / (SS_lin_meas - sigma_e^2)

The numerator is unchanged by e; only the denominator is inflated. The correction
is therefore always an inflation, which is why the bias runs toward zero. The
detection floor -- the smallest NAF distinguishable from zero -- is
sigma_e^2 / SS_lin_meas.

The block-level experiment (wp2b) sidesteps all of this by collecting in fp32:
512 masks costs about 70 seconds, so precision is free at that granularity and
the exhaustive result carries no noise-floor caveat at all.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from naca.ablation import Ablator  # noqa: E402
from naca.collect import _forward_margins, preload_eval  # noqa: E402
from naca.components import build_component_index  # noqa: E402
from naca.config import load_config  # noqa: E402
from naca.datasets import load_dataset, num_classes_for  # noqa: E402
from naca.masks import MaskStore  # noqa: E402
from naca.models import build_model, load_checkpoint  # noqa: E402


@torch.inference_mode()
def margins_for(model, index, masks, rows, x_eval, y_eval, sub_batch, dtype):
    ablator = Ablator(model, index)
    buf = torch.empty(int(x_eval.shape[0]), dtype=torch.float32, device=x_eval.device)
    out = np.empty((len(rows), int(x_eval.shape[0])), dtype=np.float32)
    try:
        for n, t in enumerate(rows):
            v = torch.from_numpy(masks.unpack(t, t + 1)[0].astype(np.float32)).to(x_eval.device)
            ablator.apply(v)
            _forward_margins(model, x_eval, y_eval, buf, sub_batch, dtype)
            out[n] = buf.cpu().numpy()
    finally:
        ablator.restore()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="precision floor for NAF")
    ap.add_argument("--config", type=Path, default=None,
                    help="defaults to the collect dir's own config_resolved.yaml")
    ap.add_argument("--collect-dir", type=Path, default=None)
    ap.add_argument("--n-masks", type=int, default=3000)
    args = ap.parse_args()

    # The config MUST describe the model that produced this collect dir. Defaulting
    # to configs/setup_a.yaml silently loaded the seed-1 checkpoint against seed-2
    # margins and reported the between-model difference as a "precision floor"
    # (8.69 of residual variance). Prefer the run's own resolved config.
    d = args.collect_dir
    if args.config is not None:
        cfg = load_config(args.config)
        d = d or sorted(cfg.paths.runs.glob("*_collect"))[-1]
    else:
        if d is None:
            cfg = load_config(ROOT / "configs" / "setup_a.yaml")
            d = sorted(cfg.paths.runs.glob("*_collect"))[-1]
        own = Path(d) / "config_resolved.yaml"
        if not own.exists():
            raise SystemExit(f"[prec] no config_resolved.yaml in {d}; pass --config explicitly")
        cfg = load_config(own)
        print(f"[prec] config from run dir: {own}")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[prec] collect dir: {d}\n")

    model = build_model(cfg.arch, num_classes=num_classes_for(cfg.dataset), device=device)
    load_checkpoint(model, cfg.paths.models / f"{cfg.setup}_{cfg.arch}.pt")
    model.eval()
    index = build_component_index(model, arch=cfg.arch, expected_n=cfg.n_components, verbose=False)
    test_ds = load_dataset(cfg.dataset, cfg.paths.data, train=False, device=device, download=False)
    eval_ds = test_ds.subset(np.load(d / "eval_indices.npy"))
    x_eval, y_eval = preload_eval(eval_ds.images, eval_ds.labels, eval_ds.mean, eval_ds.std)

    # ---- hard guard: does this checkpoint actually produce this run's margins? ----
    # Catches any model/collect-dir mismatch, not just the seed one that motivated it.
    cm_path = d / "clean_margins.npy"
    if cm_path.exists():
        stored_clean = np.asarray(np.load(cm_path), dtype=np.float32).ravel()
        # Compare in the dtype the run was COLLECTED in, so this tests model
        # identity rather than precision. Comparing fp32 against bf16-stored
        # values leaves the bf16 representation error (max ~0.19 here) in the
        # residual, which sits right at any sensible model-identity tolerance.
        collect_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                         "fp32": torch.float32}[cfg.dtype_forward]
        buf = torch.empty(int(x_eval.shape[0]), dtype=torch.float32, device=x_eval.device)
        _forward_margins(model, x_eval, y_eval, buf, cfg.eval_sub_batch, collect_dtype)
        got_clean = buf.cpu().numpy().ravel()
        n = min(stored_clean.size, got_clean.size)
        # Use MEAN absolute deviation, not max: bf16 rounding has a heavy tail, so
        # max|dev| is ~0.13 even for the correct checkpoint. Calibrated on this
        # pair of checkpoints, mean|dev| is 0.005 for the right model and 2.00 for
        # the wrong one against a margin sd of 4.33 -- a factor of ~380.
        dev = float(np.abs(stored_clean[:n] - got_clean[:n]).mean())
        tol = 0.02 * float(np.abs(stored_clean[:n]).std()) + 1e-3
        print(f"[prec] checkpoint/collect-dir agreement: mean|clean margin dev| "
              f"{dev:.4f} (tol {tol:.4f}, margin sd {stored_clean[:n].std():.3f})")
        if dev > tol:
            raise SystemExit(
                f"[prec] ABORT: the loaded checkpoint does not reproduce {cm_path.name} "
                f"(max deviation {dev:.4f} > tol {tol:.4f}). This collect dir was produced "
                f"by a different model. Pass the matching --config. Any 'precision floor' "
                f"computed here would be a between-model difference, not a precision floor."
            )
    else:
        print("[prec] WARNING: no clean_margins.npy; cannot verify checkpoint matches run")

    masks = MaskStore.load(d / "masks_train.npz")
    Y_stored = np.load(d / "Y_train.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(masks), size=min(args.n_masks, len(masks)), replace=False))
    print(f"[prec] recomputing {len(rows)} masks in fp32 (stored values were bf16)")

    y32 = margins_for(model, index, masks, rows, x_eval, y_eval, cfg.eval_sub_batch, torch.float32)
    y16 = np.asarray(Y_stored[rows], dtype=np.float32)

    diff = y32 - y16
    print()
    print("  bf16 vs fp32 discrepancy on identical masks")
    print(f"      mean        {diff.mean():+.6f}")
    print(f"      sd          {diff.std():.6f}")
    print(f"      max|.|      {np.abs(diff).max():.6f}")
    print(f"      margin sd   {y32.std():.4f}   (for scale)")

    noise_var = float(diff.var())

    # residual variance the interaction model competes for, at alpha_train
    lin_pred = np.asarray(np.load(d / "lin_pred_train.npy", mmap_mode="r"))[rows]
    resid = y16 - lin_pred
    resid_var = float(resid.var())
    total_var = float(y16.var())
    r2_lin = 1.0 - resid_var / max(total_var, 1e-12)

    # The FLOOR is noise/residual and needs no NACA fit. Only the NAF comparison
    # and the bias correction do, so a missing fit must not lose the floor --
    # WP8 ran only tagged K-sweep fits and no untagged base fit, and this raised
    # FileNotFoundError *after* the floor had already been computed.
    naf_measured = float("nan")
    naf_source = "none"
    cands = [d / "naca_summary.json"] + sorted(d.glob("naca_summary_*.json"))
    for c in cands:
        if not c.exists():
            continue
        try:
            summ = json.loads(c.read_text(encoding="utf-8"))
            at = [r for r in summ["across_alpha"] if r["alpha_test"] == cfg.alpha_train]
        except Exception:
            continue
        if at:
            naf_measured = float(at[0]["naf_oc_mean"])
            naf_source = c.name
            break
    if naf_source == "none":
        print("[prec] no NACA fit found in this run dir; reporting the floor only "
              "(NAF comparison and bias correction skipped)")
    else:
        print(f"[prec] NAF read from {naf_source}")

    floor = noise_var / max(resid_var, 1e-12)
    print()
    print("  precision floor")
    print(f"      total variance of margins        {total_var:.4f}")
    print(f"      residual variance after linear   {resid_var:.4f}   (R2_lin = {r2_lin:.4f})")
    print(f"      bf16 noise variance              {noise_var:.6f}")
    print(f"      FLOOR = noise / residual         {floor:.6f}   ({floor * 100:.4f}% of residual)")
    print(f"      measured NAF at alpha_train      {naf_measured:.6f}   ({naf_measured * 100:.4f}%)")
    ratio = naf_measured / max(floor, 1e-30)
    print(f"      NAF / floor                      {ratio:.2f}x")

    # bias correction: only the denominator of NAF is inflated by the noise
    denom = max(resid_var - noise_var, 1e-12)
    naf_corrected = naf_measured * resid_var / denom
    print()
    print("  bias-corrected estimate (see module docstring for the derivation)")
    print(f"      NAF as measured (bf16)           {naf_measured:.6f}")
    print(f"      NAF bias-corrected               {naf_corrected:.6f}")
    print(f"      inflation factor                 {resid_var / denom:.4f}x")
    print(f"      detection floor                  {floor:.6f}")
    print(f"      -> report as: NAF = {naf_corrected:.5f} "
          f"(floor {floor:.5f}); values below the floor are not resolvable")

    print()
    if math.isnan(naf_measured):
        verdict = ("FLOOR ONLY -- no NACA fit in this run dir, so no NAF comparison. "
                   "The floor itself is valid.")
    elif ratio > 5:
        verdict = "NULL STANDS -- the measured NAF is well above the bf16 precision floor"
    elif ratio > 1.5:
        verdict = "MARGINAL -- NAF exceeds the floor but not comfortably; report both"
    else:
        verdict = "NOT RESOLVABLE at bf16 -- recollect in fp32 before claiming the null"
    print(f"  VERDICT: {verdict}")

    out = {
        "naf_source": naf_source,
        "n_masks": int(len(rows)),
        "diff_mean": float(diff.mean()),
        "diff_sd": float(diff.std()),
        "diff_max_abs": float(np.abs(diff).max()),
        "margin_sd": float(y32.std()),
        "total_var": total_var,
        "residual_var": resid_var,
        "r2_lin_on_subset": float(r2_lin),
        "noise_var": noise_var,
        "floor_fraction_of_residual": floor,
        "naf_measured": naf_measured,
        "naf_over_floor": float(ratio),
        "naf_bias_corrected": float(naf_corrected),
        "inflation_factor": float(resid_var / denom),
        "verdict": verdict,
        "note": "no refit performed; NAF corrected analytically from the full-data fit",
    }
    (d / "precision_floor.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[prec] wrote {d / 'precision_floor.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
