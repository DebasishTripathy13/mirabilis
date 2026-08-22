"""End-to-end tests for the streaming engine.

The central invariant: the engine must not move more bytes than the model
actually demands. Because throughput is bandwidth-bound, every redundant byte
is directly lost throughput -- a prefetch that transfers a chunk and then
discards it, forcing `get` to transfer it again, is not a missed optimisation
but a doubling of the cost.
"""

import pytest
import torch

from corestream.engine import EngineConfig, StreamingEngine, zipf_router
from corestream.loaders import DensePlan, MoEPlan
from corestream.sources import SyntheticSource

CHUNK = 256 * 1024
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def dense_engine(layers=8, hot_chunks=2, depth=2):
    keys = [f"layer.{i}" for i in range(layers)]
    source = SyntheticSource(keys, chunk_bytes=CHUNK)
    config = EngineConfig(
        hot_budget_bytes=hot_chunks * source.nbytes(keys[0]),
        prefetch_depth=depth,
        workers=2,
    )
    return StreamingEngine(source, DensePlan(num_layers=layers), config)


def moe_engine(layers=4, experts=16, top_k=2, hot_chunks=8, depth=2):
    keys = [f"shared.{i}" for i in range(layers)]
    keys += [f"expert.{i}.{e}" for i in range(layers) for e in range(experts)]
    source = SyntheticSource(keys, chunk_bytes=CHUNK)
    config = EngineConfig(
        hot_budget_bytes=hot_chunks * CHUNK,
        prefetch_depth=depth,
        workers=2,
    )
    plan = MoEPlan(num_layers=layers, num_experts=experts, experts_per_token=top_k)
    return StreamingEngine(source, plan, config)


def test_dense_run_completes():
    with dense_engine() as engine:
        report = engine.run(tokens=3)
    assert report.tokens == 3
    assert report.wall_seconds > 0


def test_moe_requires_a_router():
    with moe_engine() as engine:
        with pytest.raises(ValueError, match="router"):
            engine.run(tokens=1)


def test_moe_run_completes():
    with moe_engine() as engine:
        report = engine.run(tokens=3, router=zipf_router(16, 2))
    assert report.tokens == 3


@cuda_only
def test_prefetch_does_not_duplicate_transfers():
    """Regression: prefetching must not double the bytes moved.

    A prefetch that fails the admission threshold used to discard the chunk
    it had just transferred, so the following `get` transferred it again.
    That made prefetching actively harmful -- it inflated traffic on an
    engine whose throughput is set by exactly that traffic.
    """
    layers, tokens = 8, 3
    with dense_engine(layers=layers, hot_chunks=2) as engine:
        report = engine.run(tokens=tokens)

    demanded = layers * tokens * CHUNK
    assert engine.store.stats.bytes_demanded == demanded
    assert report.bytes_moved <= demanded * 1.05, (
        f"moved {report.bytes_moved} bytes for {demanded} bytes of demand "
        f"({report.bytes_moved / demanded:.2f}x)"
    )


@cuda_only
def test_staged_chunks_are_claimed_not_retransferred():
    """A prefetched chunk must be consumed by `get`, not re-fetched."""
    with dense_engine(layers=8, hot_chunks=1, depth=3) as engine:
        engine.run(tokens=3)
        assert engine.store.stats.staged_hits > 0


@cuda_only
def test_roofline_is_not_derived_from_the_run():
    """Utilization must be measured against fixed hardware capability.

    Deriving the ceiling from the run's own bandwidth reports ~100%
    regardless of how much traffic was wasted, which makes the metric
    meaningless.
    """
    keys = [f"layer.{i}" for i in range(8)]
    source = SyntheticSource(keys, chunk_bytes=CHUNK)
    reference = 8.0 * 1024**3
    config = EngineConfig(
        hot_budget_bytes=0,
        prefetch_depth=1,
        workers=1,
        reference_bandwidth_bytes_per_sec=reference,
    )
    with StreamingEngine(source, DensePlan(num_layers=8), config) as engine:
        report = engine.run(tokens=2)
    assert report.roofline_utilization <= 1.0


@cuda_only
def test_cache_reduces_bytes_moved_for_moe():
    """The core claim for MoE: skewed routing makes caching pay.

    With Zipf-distributed routing a minority of experts absorb most traffic,
    so a cache holding that head should serve a large share of requests. The
    dense case has no such head, which is why it cannot be sped up this way.
    """
    router = zipf_router(16, 2, skew=1.6, seed=1)
    with moe_engine(experts=16, top_k=2, hot_chunks=12) as engine:
        report = engine.run(tokens=12, router=router)
    assert report.stall_free_rate > 0.3


@cuda_only
def test_static_pinning_beats_lru_on_cyclic_access():
    """LRU is worst-case for a cycle larger than the cache.

    A dense model revisits every layer once per token in fixed order. Under
    LRU each layer is evicted by the ones that follow it before it comes
    round again, so the hit rate is zero regardless of cache size. Refusing
    to evict keeps a fixed subset resident and turns that into savings
    proportional to cache-over-working-set.
    """
    from corestream.store import LFUAdmission, LRUEviction, StaticPinning

    # Enough tokens that the first pass -- which must load everything
    # regardless of policy -- does not dominate the measurement.
    layers, hot_chunks, tokens = 16, 4, 16
    keys = [f"layer.{i}" for i in range(layers)]

    def measure(eviction):
        source = SyntheticSource(keys, chunk_bytes=CHUNK)
        config = EngineConfig(
            hot_budget_bytes=hot_chunks * CHUNK, prefetch_depth=2, workers=2
        )
        with StreamingEngine(
            source,
            DensePlan(num_layers=layers),
            config,
            admission=LFUAdmission(threshold=1),
            eviction=eviction,
        ) as engine:
            return engine.run(tokens=tokens).savings_rate

    lru = measure(LRUEviction())
    pinned = measure(StaticPinning())

    assert lru < 0.05, f"expected LRU to collapse on cyclic access, got {lru:.1%}"
    # A fixed resident subset is hit once per cycle, so savings approach cache
    # size over working set, less the first pass which must load either way.
    ceiling = (hot_chunks / layers) * (tokens - 1) / tokens
    assert pinned > 0.6 * ceiling, f"pinned saved {pinned:.1%}, ceiling {ceiling:.1%}"
    assert pinned > 4 * max(lru, 0.01)


def test_report_summary_renders():
    with dense_engine() as engine:
        report = engine.run(tokens=2)
    text = report.summary()
    assert "ceiling (with cache)" in text
    assert "bus utilization" in text
    assert "tok/s" in text
