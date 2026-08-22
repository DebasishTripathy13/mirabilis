"""Tests for the tiered store's caching policy.

These run on CPU so the policy logic is verified without a GPU. The policy is
what determines bytes-moved-per-token, which is what determines throughput --
so it is worth testing independently of any device behaviour.
"""

import pytest
import torch

from corestream.sources import SyntheticSource
from corestream.store import LFUAdmission, TieredWeightStore

CHUNK = 4096


def make_store(num_chunks=8, hot_chunks=4, threshold=2, device="cpu"):
    keys = [f"chunk.{i}" for i in range(num_chunks)]
    source = SyntheticSource(keys, chunk_bytes=CHUNK)
    return (
        TieredWeightStore(
            source=source,
            hot_budget_bytes=hot_chunks * source.nbytes(keys[0]),
            device=device,
            admission=LFUAdmission(threshold=threshold),
        ),
        keys,
    )


def test_get_returns_correct_data():
    store, keys = make_store()
    tensor = store.get(keys[0])
    assert tensor.numel() > 0
    torch.testing.assert_close(tensor, store.source.host_tensor(keys[0]))


def test_first_touch_is_a_miss():
    store, keys = make_store()
    store.get(keys[0])
    assert store.stats.misses == 1
    assert store.stats.hot_hits == 0


def test_lfu_admission_requires_repeat_demand():
    """A chunk touched once must not become resident.

    This is the scan-resistance property: without it, a single pass over
    cold chunks evicts the genuinely hot working set.
    """
    store, keys = make_store(threshold=3, device="cpu")
    store.hot_budget_bytes = 10 * CHUNK  # ample room; admission is the gate
    store.device = torch.device("cpu")

    # On CPU the store never admits, so assert on the policy directly.
    policy = LFUAdmission(threshold=3)
    assert not policy.should_admit("k", hits=1)
    assert not policy.should_admit("k", hits=2)
    assert policy.should_admit("k", hits=3)


def test_threshold_one_admits_immediately():
    """Threshold 1 is the right setting for dense models.

    Every layer is touched exactly once per token in a fixed cycle, so there
    is no cold tail to protect against and delaying admission only wastes
    the first cycle.
    """
    policy = LFUAdmission(threshold=1)
    assert policy.should_admit("k", hits=1)


def test_stats_track_requests():
    store, keys = make_store()
    for _ in range(4):
        store.get(keys[0])
    assert store.stats.total_requests == 4
    assert 0.0 <= store.stats.stall_free_rate <= 1.0
    assert store.stats.bytes_demanded == 4 * CHUNK


def test_unknown_key_raises():
    store, keys = make_store()
    with pytest.raises(KeyError):
        store.get("nonexistent")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hot_tier_respects_budget():
    store, keys = make_store(num_chunks=16, hot_chunks=4, threshold=1, device="cuda")
    for key in keys:
        store.get(key)
        store.get(key)  # second touch crosses the threshold
    assert store.hot_bytes <= store.hot_budget_bytes
    assert len(store.hot_keys) <= 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_eviction_is_lru():
    store, keys = make_store(num_chunks=8, hot_chunks=2, threshold=1, device="cuda")
    for key in keys[:2]:
        store.get(key)
        store.get(key)
    assert set(store.hot_keys) == set(keys[:2])

    store.get(keys[0])  # refresh keys[0], making keys[1] least-recent
    for _ in range(2):
        store.get(keys[2])
    assert keys[1] not in store.hot_keys
    assert keys[0] in store.hot_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cached_reads_do_not_recross_the_bus():
    """A HOT hit must not increment bytes_promoted.

    Throughput is bandwidth-bound, so this counter is the one that maps
    directly to tok/s. If a cache hit still moved bytes, the cache would be
    decorative.
    """
    store, keys = make_store(num_chunks=4, hot_chunks=4, threshold=1, device="cuda")
    store.get(keys[0])
    store.get(keys[0])
    baseline = store.stats.bytes_promoted
    store.get(keys[0])
    assert store.stats.bytes_promoted == baseline


class CountingSource:
    """Wraps a source and counts how often the host bytes are materialised."""

    def __init__(self, inner):
        self.inner = inner
        self.host_calls = 0

    def keys(self):
        return self.inner.keys()

    def nbytes(self, key):
        return self.inner.nbytes(key)

    def host_tensor(self, key):
        self.host_calls += 1
        return self.inner.host_tensor(key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_chunks_do_not_re_materialise_host_bytes():
    """Regression: a pinned chunk must not touch the source again.

    Obtaining the host tensor is expensive for a real checkpoint -- a chunk
    spans many tensors, so the source packs them into a fresh contiguous
    buffer on every call, an allocation plus a full memcpy of the layer.
    Doing that before checking the pinned tier meant every transfer paid for
    a buffer it then discarded, and it cost roughly a third of end-to-end
    throughput while every test still passed.
    """
    keys = [f"chunk.{i}" for i in range(4)]
    inner = SyntheticSource(keys, chunk_bytes=CHUNK)
    source = CountingSource(inner)
    store = TieredWeightStore(
        source=source,
        hot_budget_bytes=0,  # force the transfer path, not the VRAM cache
        device="cuda",
        pinned_budget_bytes=8 * CHUNK,
    )

    for key in keys:
        store.get(key)
    first_pass = source.host_calls
    assert first_pass == len(keys), "each chunk should be read once to be pinned"

    for _ in range(3):
        for key in keys:
            store.get(key)
    assert source.host_calls == first_pass, (
        f"pinned chunks re-read the source {source.host_calls - first_pass} times"
    )
