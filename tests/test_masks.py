"""Test 10 -- bit-packed mask round-trip, plus sampling invariants."""

from __future__ import annotations

import numpy as np
import pytest

from naca.masks import MaskSpec, MaskStore


@pytest.fixture(scope="module")
def store() -> MaskStore:
    return MaskStore.generate(MaskSpec(n_components=2306, alpha=0.10, m=512, seed=0))


# ------------------------------------------------------------------ test 10
def test_pack_unpack_round_trip_is_exact(store):
    """Test 10: unpack(pack(x)) == x, bit for bit, including the ragged tail."""
    dense = store.unpack()
    assert dense.shape == (len(store), store.N)
    assert set(np.unique(dense)) <= {0, 1}
    repacked = np.packbits(dense, axis=1)
    assert np.array_equal(repacked, store.packed), "re-packing the unpacked masks differs"


def test_unpack_chunks_agree_with_full(store):
    full = store.unpack()
    stitched = np.concatenate([blk for _, _, blk in store.iter_chunks(97)], axis=0)
    assert np.array_equal(full, stitched)
    assert np.array_equal(store.unpack(10, 20), full[10:20])


def test_unpack_rows_arbitrary_order(store):
    rows = np.array([7, 0, 511, 3])
    assert np.array_equal(store.unpack_rows(rows), store.unpack()[rows])


def test_ragged_tail_bits_are_dropped():
    """N not a multiple of 8 must not leak padding bits into the mask."""
    spec = MaskSpec(n_components=13, alpha=0.3, m=64, seed=1)
    s = MaskStore.generate(spec)
    dense = s.unpack()
    assert dense.shape == (64, 13)
    assert np.all((dense == 0).sum(axis=1) == spec.k_ablated)


# ------------------------------------------------------- sampling invariants
def test_exact_k_ablated_per_mask(store):
    counts = store.n_ablated()
    k = store.spec.k_ablated
    assert np.all(counts == k), (
        f"uniform alpha-fraction subsets must ablate exactly {k} components; "
        f"saw {sorted(set(counts.tolist()))[:5]}"
    )
    assert k == round(0.10 * 2306)


def test_column_frequency_is_uniform(store):
    """Every component must be kept with the same probability, ~1-alpha."""
    freq = store.column_frequency()
    expected = 1.0 - store.spec.k_ablated / store.N
    assert abs(freq.mean() - expected) < 1e-9
    # 512 masks is a small sample; allow a generous but finite band
    assert freq.std() < 0.03, f"per-component keep rates are not uniform (std={freq.std():.4f})"


def test_determinism_and_verify(store):
    again = MaskStore.generate(store.spec)
    assert np.array_equal(store.packed, again.packed)
    assert store.verify()


def test_streams_are_disjoint_across_kind_and_alpha():
    base = MaskSpec(n_components=256, alpha=0.1, m=128, seed=0, kind="train")
    other_kind = MaskSpec(n_components=256, alpha=0.1, m=128, seed=0, kind="test")
    other_alpha = MaskSpec(n_components=256, alpha=0.2, m=128, seed=0, kind="train")
    a = MaskStore.generate(base).packed
    b = MaskStore.generate(other_kind).packed
    c = MaskStore.generate(other_alpha).packed
    assert not np.array_equal(a, b), "train and test mask streams collide"
    assert not np.array_equal(a, c)


def test_save_load_round_trip(store, tmp_path):
    path = store.save(tmp_path / "masks.npz")
    back = MaskStore.load(path)
    assert np.array_equal(back.packed, store.packed)
    assert back.spec == store.spec
    assert back.verify()


def test_storage_is_actually_compact(store):
    rep = store.sanity_report()
    assert rep["packed_mb"] < rep["dense_mb_equivalent"] / 8


def test_k_clamped_at_extremes():
    assert MaskSpec(n_components=100, alpha=0.001, m=4, seed=0).k_ablated == 1
    assert MaskSpec(n_components=100, alpha=0.999, m=4, seed=0).k_ablated == 99


def test_bad_ranges_rejected(store):
    with pytest.raises(IndexError):
        store.unpack(0, len(store) + 1)
