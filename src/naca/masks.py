"""Bit-packed ablation-mask store.

A ``50,000 x 22,720`` float32 mask matrix is 4.5 GB; bit-packed along the
component axis it is 142 MB (CLAUDE.md 5.2). Masks are generated from a seeded
``np.random.default_rng`` in fixed-size chunks, so the whole matrix regenerates
deterministically from ``(seed, N, alpha, m)`` -- :meth:`MaskStore.verify`
re-derives it and compares, which is the reproducibility guarantee that lets us
skip full cuDNN determinism.

Sampling: each mask ablates a **uniformly random alpha-fraction subset** of the
components, i.e. exactly ``k = round(alpha * N)`` components chosen without
replacement -- not independent Bernoulli draws. This matters: fixed-size subsets
give every pair the same co-ablation probability, which is what the
identifiability calculation in the proposal assumes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["MaskSpec", "MaskStore"]

_GEN_CHUNK = 1024  # masks per generation chunk; bounds peak RAM
# At N=3904 a 4096-row chunk needs a 64 MB float32 noise array plus a 16 MB
# byte block. On a 7.6 GB machine with the pagefile already deep that is
# enough to fail; 1024 rows costs 16 MB + 4 MB and runs the same.


@dataclass(frozen=True)
class MaskSpec:
    """Everything needed to regenerate a mask matrix bit-for-bit."""

    n_components: int
    alpha: float
    m: int
    seed: int
    kind: str = "train"  # 'train' | 'test' -- keeps streams disjoint

    @property
    def k_ablated(self) -> int:
        """Exact number of components ablated per mask."""
        k = int(round(self.alpha * self.n_components))
        return max(1, min(self.n_components - 1, k))

    @property
    def stream_seed(self) -> int:
        """Distinct RNG stream per (seed, kind, alpha, m) so sets never collide."""
        key = f"{self.seed}|{self.kind}|{self.alpha!r}|{self.m}|{self.n_components}"
        return int(np.frombuffer(_blake(key), dtype=np.uint32)[0])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k_ablated"] = self.k_ablated
        return d


def _blake(s: str) -> bytes:
    import hashlib

    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def _generate_packed(spec: MaskSpec) -> np.ndarray:
    """Generate the full bit-packed mask matrix, chunked to bound RAM."""
    n, k, m = spec.n_components, spec.k_ablated, spec.m
    rng = np.random.default_rng(spec.stream_seed)
    n_bytes = (n + 7) // 8
    packed = np.empty((m, n_bytes), dtype=np.uint8)

    for lo in range(0, m, _GEN_CHUNK):
        hi = min(lo + _GEN_CHUNK, m)
        rows = hi - lo
        # argpartition over uniform noise == uniform k-subset without replacement
        noise = rng.random((rows, n), dtype=np.float32)
        ablate = np.argpartition(noise, k - 1, axis=1)[:, :k]
        block = np.ones((rows, n), dtype=np.uint8)
        np.put_along_axis(block, ablate, 0, axis=1)
        packed[lo:hi] = np.packbits(block, axis=1)
        del noise, ablate, block
    return packed


class MaskStore:
    """Bit-packed masks with lazy per-chunk unpacking."""

    def __init__(self, packed: np.ndarray, spec: MaskSpec) -> None:
        expected_bytes = (spec.n_components + 7) // 8
        if packed.dtype != np.uint8:
            raise TypeError(f"packed masks must be uint8, got {packed.dtype}")
        if packed.shape != (spec.m, expected_bytes):
            raise ValueError(
                f"packed shape {packed.shape} != expected {(spec.m, expected_bytes)}"
            )
        self.packed = packed
        self.spec = spec

    # ------------------------------------------------------------------
    @classmethod
    def generate(cls, spec: MaskSpec) -> MaskStore:
        return cls(_generate_packed(spec), spec)

    def __len__(self) -> int:
        return self.spec.m

    @property
    def N(self) -> int:
        return self.spec.n_components

    # ------------------------------------------------------------------
    def unpack(self, lo: int = 0, hi: int | None = None) -> np.ndarray:
        """Rows ``[lo, hi)`` as a dense ``(rows, N)`` uint8 array of 0/1."""
        hi = self.spec.m if hi is None else hi
        if not 0 <= lo <= hi <= self.spec.m:
            raise IndexError(f"row range [{lo},{hi}) outside [0,{self.spec.m})")
        block = np.unpackbits(self.packed[lo:hi], axis=1)
        return block[:, : self.N]

    def unpack_rows(self, rows: np.ndarray) -> np.ndarray:
        """Arbitrary (possibly unsorted) row indices as a dense ``(len(rows), N)`` array."""
        rows = np.asarray(rows, dtype=np.int64)
        block = np.unpackbits(self.packed[rows], axis=1)
        return block[:, : self.N]

    def torch_chunk(
        self,
        lo: int,
        hi: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Rows ``[lo, hi)`` as a torch tensor on ``device``."""
        arr = self.unpack(lo, hi)
        return torch.from_numpy(arr).to(device=device, dtype=dtype, non_blocking=True)

    def iter_chunks(self, chunk: int = 1024):
        """Yield ``(lo, hi, dense_uint8_block)`` over the whole store."""
        for lo in range(0, self.spec.m, chunk):
            hi = min(lo + chunk, self.spec.m)
            yield lo, hi, self.unpack(lo, hi)

    # ------------------------------------------------------------------
    def n_ablated(self, lo: int = 0, hi: int | None = None) -> np.ndarray:
        """Per-mask count of ablated components -- should be exactly ``k`` everywhere."""
        dense = self.unpack(lo, hi)
        return (dense == 0).sum(axis=1)

    def column_frequency(self, chunk: int = 1024, max_rows: int | None = 8192) -> np.ndarray:
        """Per-component fraction of masks in which it is *kept*. Should be ~1-alpha.

        Sampled by default: this is a logging diagnostic, and sweeping all 50,000
        rows to produce it is not worth the allocation on a memory-tight machine.
        Pass ``max_rows=None`` for the exact figure.
        """
        n = self.spec.m if max_rows is None else min(self.spec.m, max_rows)
        acc = np.zeros(self.N, dtype=np.float64)
        seen = 0
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            acc += self.unpack(lo, hi).sum(axis=0)
            seen += hi - lo
        return acc / max(seen, 1)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, packed=self.packed, spec=json.dumps(self.spec.to_dict())
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> MaskStore:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"mask store not found: {path}")
        with np.load(path, allow_pickle=False) as z:
            packed = z["packed"]
            raw = json.loads(str(z["spec"]))
        raw.pop("k_ablated", None)
        return cls(packed, MaskSpec(**raw))

    # ------------------------------------------------------------------
    def verify(self) -> bool:
        """Regenerate from the seed and check the packed bytes are identical."""
        return bool(np.array_equal(self.packed, _generate_packed(self.spec)))

    def sanity_report(self) -> dict:
        """Cheap invariants worth logging before a 50k-mask run."""
        counts = self.n_ablated(0, min(1024, self.spec.m))
        freq = self.column_frequency()
        return {
            "m": self.spec.m,
            "N": self.N,
            "alpha": self.spec.alpha,
            "k_ablated": self.spec.k_ablated,
            "k_observed_unique": sorted(set(int(c) for c in counts)),
            "keep_freq_mean_sampled": float(freq.mean()),
            "keep_freq_min": float(freq.min()),
            "keep_freq_max": float(freq.max()),
            "expected_keep_freq": 1.0 - self.spec.k_ablated / self.N,
            "packed_mb": round(self.packed.nbytes / 1024**2, 2),
            "dense_mb_equivalent": round(self.spec.m * self.N * 4 / 1024**2, 2),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MaskStore(m={self.spec.m}, N={self.N}, alpha={self.spec.alpha}, "
            f"k={self.spec.k_ablated}, kind={self.spec.kind!r}, "
            f"{self.packed.nbytes / 1024**2:.1f} MB packed)"
        )
