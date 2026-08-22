"""Tests for the prefetch scheduler."""

import torch

from corestream.scheduler import PrefetchScheduler
from corestream.sources import SyntheticSource
from corestream.store import LFUAdmission, TieredWeightStore


def make_scheduler(num_chunks=8, workers=2):
    keys = [f"chunk.{i}" for i in range(num_chunks)]
    source = SyntheticSource(keys, chunk_bytes=4096)
    store = TieredWeightStore(
        source=source,
        hot_budget_bytes=4 * source.nbytes(keys[0]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        admission=LFUAdmission(threshold=1),
    )
    return PrefetchScheduler(store, workers=workers), store, keys


def test_hints_are_processed():
    scheduler, store, keys = make_scheduler()
    try:
        scheduler.hint(keys[:4])
        assert scheduler.wait_idle(timeout=10)
        assert scheduler.stats.chunks_prefetched == 4
    finally:
        scheduler.shutdown()


def test_duplicate_hints_are_deduped():
    """Re-hinting a chunk already queued must not transfer it twice."""
    scheduler, store, keys = make_scheduler()
    try:
        scheduler.hint([keys[0]] * 5)
        assert scheduler.wait_idle(timeout=10)
        assert scheduler.stats.hints_deduped >= 4
    finally:
        scheduler.shutdown()


def test_unknown_key_is_recorded_not_raised():
    """A failed prefetch must not kill the worker.

    `get` retries synchronously and surfaces the error there, where it can be
    attributed to a specific step.
    """
    scheduler, store, keys = make_scheduler()
    try:
        scheduler.hint(["does.not.exist"])
        assert scheduler.wait_idle(timeout=10)
        assert scheduler.stats.errors == 1
        scheduler.hint([keys[0]])
        assert scheduler.wait_idle(timeout=10)
        assert scheduler.stats.chunks_prefetched == 1
    finally:
        scheduler.shutdown()


def test_shutdown_is_idempotent():
    scheduler, _, _ = make_scheduler()
    scheduler.shutdown()
    scheduler.shutdown()


def test_hints_after_shutdown_are_ignored():
    scheduler, store, keys = make_scheduler()
    scheduler.shutdown()
    scheduler.hint(keys)
    assert scheduler.stats.chunks_prefetched == 0


def test_context_manager_shuts_down():
    keys = [f"chunk.{i}" for i in range(4)]
    source = SyntheticSource(keys, chunk_bytes=4096)
    store = TieredWeightStore(source=source, hot_budget_bytes=0, device="cpu")
    with PrefetchScheduler(store, workers=1) as scheduler:
        scheduler.hint(keys[:1])
        assert scheduler.wait_idle(timeout=10)
    assert all(not t.is_alive() for t in scheduler._threads)
