"""Configuration objects for NACA.

Every knob lives here or in ``configs/*.yaml`` -- nothing is hard-coded in the
scripts (CLAUDE.md section 5.6). ``load_config`` validates aggressively and
fails loudly, because a silent config typo costs hours on a 50k-mask run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "Paths",
    "TrainConfig",
    "BackdoorConfig",
    "Config",
    "load_config",
    "set_seeds",
    "make_run_dir",
]

# --------------------------------------------------------------------------
# Project root. Hard rule 3: never write to C:.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("NACA_ROOT", r"D:\naca"))


@dataclass
class Paths:
    """All filesystem locations, forced under ``root``."""

    root: Path = PROJECT_ROOT
    data: Path = field(default=None)  # type: ignore[assignment]
    models: Path = field(default=None)  # type: ignore[assignment]
    masks: Path = field(default=None)  # type: ignore[assignment]
    attribs: Path = field(default=None)  # type: ignore[assignment]
    results: Path = field(default=None)  # type: ignore[assignment]
    runs: Path = field(default=None)  # type: ignore[assignment]
    cache: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for name in ("data", "models", "masks", "attribs", "results", "runs", "cache"):
            if getattr(self, name) is None:
                setattr(self, name, self.root / name)
            else:
                setattr(self, name, Path(getattr(self, name)))

    def mkdirs(self) -> None:
        for name in ("data", "models", "masks", "attribs", "results", "runs", "cache"):
            getattr(self, name).mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in ("root", "data", "models", "masks", "attribs", "results", "runs", "cache")
        }


@dataclass
class TrainConfig:
    """Hyper-parameters for training / fine-tuning the target model."""

    epochs: int = 24
    lr: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    batch_size: int = 512
    lr_schedule: Literal["triangular", "cosine", "step"] = "triangular"
    lr_peak_epoch: int = 5
    # ViT (WP9) does not train with the SGD recipe the ResNets use. Defaulted to
    # "sgd" so every existing setup is bit-for-bit unaffected.
    optimizer: Literal["sgd", "adamw"] = "sgd"
    warmup_epochs: int = 0
    target_acc: float = 0.91  # Phase 2 acceptance gate
    acc_tolerance: float = 0.015


@dataclass
class BackdoorConfig:
    """Setup A-bd: blue 4x4 square, top-left, on 50% of training 'airplane' images."""

    enabled: bool = False
    target_class: int = 0  # CIFAR-10 'airplane'
    patch_size: int = 4
    patch_row: int = 0
    patch_col: int = 0
    # RGB in [0,1]; "blue" per CLAUDE.md section 3
    patch_rgb: tuple[float, float, float] = (0.0, 0.0, 1.0)
    poison_fraction: float = 0.5


@dataclass
class Config:
    """Full experiment configuration."""

    # --- identity -------------------------------------------------------
    setup: str = "A"
    arch: str = "resnet18_cifar"
    dataset: str = "cifar10"

    # --- component / ablation design -----------------------------------
    n_components: int | None = None  # None => resolve empirically, then assert
    alpha_train: float = 0.10
    m_train: int = 50_000
    alpha_test_grid: list[float] = field(
        default_factory=lambda: [0.01, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
    )
    m_test: int = 2_000
    n_eval_examples: int = 1_000

    # --- Stage 2: screening ---------------------------------------------
    K: int = 100
    screening: Literal["marginal", "residual", "union"] = "residual"

    # --- Stage 3: indefinite low-rank interaction model -----------------
    r_U: int = 8
    r_P: int = 8
    lambda_ridge: float = 1.0  # Stage-1 ridge (bias never penalised)
    lambda_fm: float = 1e-4  # Stage-3 ridge on U, P
    freeze_linear: bool = True
    fm_lr: float = 1e-2
    fm_epochs: int = 20
    fm_batch_size: int = 1024
    fm_init_std: float = 0.01
    psd_only: bool = False  # ablation: standard FM, A = UU^T

    # --- Stage 4: editing ------------------------------------------------
    fdr_filter: bool = True
    fdr_q: float = 0.10
    lambda_edit: float = 1.0
    k_edit: int = 32
    gain_tol: float = 1e-6

    # --- runtime ----------------------------------------------------------
    seed: int = 0
    batch_size: int = 512  # generic
    eval_sub_batch: int = 500  # forward sub-batch during collection
    use_compile: bool = False  # hard rule 2
    solver: Literal["auto", "cholesky", "cg"] = "auto"
    device: str = "cuda"
    dtype_forward: Literal["bf16", "fp16", "fp32"] = "bf16"
    num_workers: int = 0  # hard rule 1
    cg_tol: float = 1e-6
    cg_max_iter: int = 500
    checkpoint_every: int = 1_000
    min_free_gb_d: float = 40.0
    min_free_gb_c: float = 20.0

    # --- nested ----------------------------------------------------------
    train: TrainConfig = field(default_factory=TrainConfig)
    backdoor: BackdoorConfig = field(default_factory=BackdoorConfig)
    paths: Paths = field(default_factory=Paths)

    # ---------------------------------------------------------------- API
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail loudly on anything that would waste a long run."""
        if not 0.0 < self.alpha_train < 1.0:
            raise ValueError(f"alpha_train must be in (0,1), got {self.alpha_train}")
        for a in self.alpha_test_grid:
            if not 0.0 < a < 1.0:
                raise ValueError(f"alpha_test_grid entry out of (0,1): {a}")
        if self.m_train < 1_000:
            raise ValueError(f"m_train={self.m_train} is too small to identify anything")
        if not 1 <= self.K <= 100_000:
            raise ValueError(f"K out of range: {self.K}")
        if self.screening not in ("marginal", "residual", "union"):
            raise ValueError(f"unknown screening mode: {self.screening!r}")
        if self.solver not in ("auto", "cholesky", "cg"):
            raise ValueError(f"unknown solver: {self.solver!r}")
        if self.r_U < 1 or self.r_P < 0:
            raise ValueError(f"bad ranks r_U={self.r_U}, r_P={self.r_P}")
        if self.psd_only and self.r_P != 0:
            # PSD ablation is exactly "drop the negative block".
            self.r_P = 0
        if self.num_workers != 0:
            raise ValueError("hard rule 1: num_workers must be 0 on Windows")
        if self.use_compile:
            # allowed, but must be a deliberate opt-in
            print("[config] WARNING: torch.compile enabled; Triton on Windows is unreliable")
        if not 0.0 < self.fdr_q < 1.0:
            raise ValueError(f"fdr_q must be in (0,1), got {self.fdr_q}")
        self.paths.mkdirs()

    # ------------------------------------------------------------------
    @property
    def run_tag(self) -> str:
        return f"{self.setup}_{self.arch}_{self.dataset}"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["paths"] = self.paths.as_dict()
        return d

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------


def _coerce(raw: dict[str, Any]) -> Config:
    raw = dict(raw)
    nested = {}
    for key, cls in (("train", TrainConfig), ("backdoor", BackdoorConfig), ("paths", Paths)):
        if key in raw:
            sub = raw.pop(key)
            if sub is None:
                continue
            if not isinstance(sub, dict):
                raise TypeError(f"config section {key!r} must be a mapping, got {type(sub)}")
            valid = {f.name for f in dataclasses.fields(cls)}
            unknown = set(sub) - valid
            if unknown:
                raise ValueError(f"unknown keys in {key!r}: {sorted(unknown)}")
            nested[key] = cls(**sub)

    valid_top = {f.name for f in dataclasses.fields(Config)}
    unknown = set(raw) - valid_top
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")

    if "backdoor" in nested and isinstance(nested["backdoor"].patch_rgb, list):
        nested["backdoor"].patch_rgb = tuple(nested["backdoor"].patch_rgb)  # type: ignore[assignment]

    return Config(**raw, **nested)


def load_config(path: str | Path, **overrides: Any) -> Config:
    """Load a YAML config, apply keyword overrides, validate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw)}")
    for k, v in overrides.items():
        if v is None:
            continue
        raw[k] = v
    cfg = _coerce(raw)
    return cfg


def set_seeds(seed: int) -> dict[str, int]:
    """Seed python / numpy / torch. Returns what was set, for the run record."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deliberately NOT enabling full cuDNN determinism (CLAUDE.md 5.4):
    # reproducibility comes from persisting the mask matrix instead.
    torch.backends.cudnn.benchmark = True
    return {"python": seed, "numpy": seed, "torch": seed}


def make_run_dir(cfg: Config, name: str) -> Path:
    """``runs/<timestamp>_<setup>_<name>/`` with the resolved config written in."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.paths.runs / f"{stamp}_{cfg.setup}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config_resolved.yaml")
    (run_dir / "seeds.json").write_text(json.dumps(set_seeds(cfg.seed), indent=2), encoding="utf-8")
    return run_dir
