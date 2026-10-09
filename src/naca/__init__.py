"""NACA -- Non-Additive Component Attribution.

Extends COAR (Shah, Ilyas & Madry 2024, arXiv:2404.11534) with a low-rank
*indefinite* interaction model fitted on the counterfactual data COAR already
collects -- zero additional forward passes through the target model.

Build order (the design spec section 6) is diagnostic -> estimator -> editing, and is
not to be reordered: the diagnostic yields a reportable measurement whatever the
sign of the result.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .ablation import Ablator, ablated_from_mask, mask_from_ablated
from .components import ComponentIndex, LayerSpec, build_component_index
from .config import Config, load_config, make_run_dir, set_seeds
from .masks import MaskSpec, MaskStore

__all__ = [
    "__version__",
    "Ablator",
    "ablated_from_mask",
    "mask_from_ablated",
    "ComponentIndex",
    "LayerSpec",
    "build_component_index",
    "Config",
    "load_config",
    "make_run_dir",
    "set_seeds",
    "MaskSpec",
    "MaskStore",
]
