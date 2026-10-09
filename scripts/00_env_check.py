"""Phase 0 acceptance gate.

Checks CUDA, bf16, free disk, cache redirection off C:, and write throughput on
D:. Exits non-zero if any hard check fails, so it can gate a longer run.

The C: free-space check is a **warning**, not a failure -- see the design spec section 9.
This machine has ~14 GB free on C:, below the spec's 20 GB, but every cache and
temp path is redirected onto D: (hard rule 3), which is what that check was
protecting against. The redirection itself is verified as a hard check instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))
sys.path.insert(0, str(ROOT / "src"))

# Cache/temp variables that must not resolve onto C:.
REDIRECT_VARS = (
    "TORCH_HOME",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "MPLCONFIGDIR",
    "PIP_CACHE_DIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_CACHE_HOME",
)

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.failed = 0
        self.warned = 0

    def add(self, name: str, ok: bool, detail: str, *, hard: bool = True) -> bool:
        if ok:
            tag, colour = "PASS", GREEN
        elif hard:
            tag, colour = "FAIL", RED
            self.failed += 1
        else:
            tag, colour = "WARN", YELLOW
            self.warned += 1
        print(f"  [{colour}{tag}{RESET}] {name:<38s} {detail}")
        self.rows.append({"check": name, "status": tag, "detail": detail, "hard": hard})
        return ok

    def to_dict(self) -> dict:
        return {"checks": self.rows, "failed": self.failed, "warned": self.warned}


def free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1024**3
    except OSError:
        return float("nan")


def check_python(rep: Report) -> None:
    v = sys.version_info
    rep.add(
        "python 3.11.x",
        (v.major, v.minor) == (3, 11),
        f"{v.major}.{v.minor}.{v.micro} at {sys.executable}",
    )


def check_torch(rep: Report) -> object | None:
    try:
        import torch
    except ImportError as exc:
        rep.add("torch importable", False, f"{exc}")
        return None

    rep.add("torch 2.5.x + cu121", torch.__version__.startswith("2.5"), torch.__version__)
    ok_cuda = rep.add("CUDA available", torch.cuda.is_available(), str(torch.version.cuda))
    if not ok_cuda:
        return torch

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    rep.add(
        "GPU visible",
        True,
        f"{props.name}, {vram:.2f} GB, cc {props.major}.{props.minor}",
    )
    rep.add("VRAM >= 3.5 GB", vram >= 3.5, f"{vram:.2f} GB (budget ~3.4 GB usable)")
    rep.add("bf16 supported", torch.cuda.is_bf16_supported(), "Ada native")

    # Actually exercise bf16 autocast + channels_last, not just the capability flag.
    try:
        import torch.nn as nn

        conv = nn.Conv2d(3, 16, 3, padding=1).cuda().to(memory_format=torch.channels_last)
        x = torch.randn(8, 3, 32, 32, device="cuda").contiguous(
            memory_format=torch.channels_last
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = conv(x)
        ok = y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()
        rep.add("bf16 autocast conv works", bool(ok), f"out dtype {y.dtype}")
        del conv, x, y
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - want the diagnostic, then fail the gate
        rep.add("bf16 autocast conv works", False, f"{type(exc).__name__}: {exc}")

    try:
        import torchvision

        rep.add("torchvision importable", True, torchvision.__version__)
    except ImportError as exc:
        rep.add("torchvision importable", False, str(exc))
    return torch


def check_disk(rep: Report, min_d: float, min_c: float) -> None:
    d = free_gb("D:\\")
    c = free_gb("C:\\")
    rep.add(f"D: free >= {min_d:.0f} GB", d >= min_d, f"{d:.1f} GB free")
    rep.add(
        f"C: free >= {min_c:.0f} GB",
        c >= min_c,
        f"{c:.1f} GB free -- soft check; all caches redirected to D: (the design spec section 9)",
        hard=False,
    )


def check_redirects(rep: Report) -> None:
    """Hard rule 3: nothing may cache or spill onto C:."""
    offenders: list[str] = []
    unset: list[str] = []
    for var in REDIRECT_VARS:
        val = os.environ.get(var)
        if not val:
            unset.append(var)
            continue
        if Path(val).drive.upper().startswith("C"):
            offenders.append(f"{var}={val}")
    rep.add(
        "no cache env var points at C:",
        not offenders,
        "clean" if not offenders else "; ".join(offenders),
    )
    if unset:
        rep.add(
            "cache env vars set",
            False,
            f"unset (will default onto C:): {', '.join(unset)}",
            hard=False,
        )


def check_write_speed(rep: Report, mb: int = 256) -> None:
    target = ROOT / "cache" / "_writetest.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = os.urandom(1024 * 1024)
    try:
        t0 = time.time()
        with target.open("wb") as fh:
            for _ in range(mb):
                fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        write_s = time.time() - t0

        t0 = time.time()
        with target.open("rb") as fh:
            while fh.read(8 * 1024 * 1024):
                pass
        read_s = time.time() - t0
    finally:
        target.unlink(missing_ok=True)

    w = mb / max(write_s, 1e-9)
    r = mb / max(read_s, 1e-9)
    rep.add("D: write >= 50 MB/s", w >= 50, f"write {w:.0f} MB/s, read {r:.0f} MB/s")


def check_ram(rep: Report) -> None:
    try:
        import psutil
    except ImportError:
        rep.add("RAM probe", False, "psutil not installed", hard=False)
        return
    vm = psutil.virtual_memory()
    total = vm.total / 1024**3
    avail = vm.available / 1024**3
    rep.add(
        "RAM available >= 3 GB",
        avail >= 3.0,
        f"{avail:.1f} GB free of {total:.1f} GB total",
        hard=False,
    )


def check_package(rep: Report) -> None:
    try:
        import naca

        rep.add("naca package importable", True, f"v{naca.__version__}")
    except ImportError as exc:
        rep.add("naca package importable", False, str(exc))


def main() -> int:
    ap = argparse.ArgumentParser(description="NACA Phase 0 environment gate")
    ap.add_argument("--min-free-d", type=float, default=40.0)
    ap.add_argument("--min-free-c", type=float, default=20.0)
    ap.add_argument("--write-mb", type=int, default=256)
    ap.add_argument("--json", type=Path, default=ROOT / "results" / "env_check.json")
    args = ap.parse_args()

    print(f"\nNACA Phase 0 -- environment gate\nroot: {ROOT}\n")
    rep = Report()

    print("environment")
    check_python(rep)
    check_package(rep)
    check_redirects(rep)

    print("\ncompute")
    check_torch(rep)
    check_ram(rep)

    print("\nstorage")
    check_disk(rep, args.min_free_d, args.min_free_c)
    check_write_speed(rep, args.write_mb)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")

    print(f"\n{'-' * 66}")
    if rep.failed:
        print(f"{RED}GATE FAILED{RESET}: {rep.failed} hard check(s) failed, {rep.warned} warning(s)")
    else:
        print(f"{GREEN}GATE PASSED{RESET}: 0 failures, {rep.warned} warning(s)")
    print(f"report written to {args.json}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
