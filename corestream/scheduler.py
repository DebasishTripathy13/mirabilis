"""Asynchronous prefetching: keep the bus busy while the GPU computes.

This is the TCP sliding-window idea applied to a memory hierarchy. A naive
streaming engine runs `load -> compute -> load -> compute`, leaving the bus
idle during compute and the GPU idle during load. Keeping several chunk
promotions in flight means the transfer for step N+1 is already underway
while step N executes.

How much this buys depends on the ratio of load time to compute time. On a
consumer laptop streaming a large model, loads dominate compute by roughly an
order of magnitude, so overlap alone cannot rescue throughput -- that is what
the HOT cache in `store.py` is for, by reducing the bytes that must move at
all. Overlap removes the compute time from the critical path; caching removes
the transfer itself.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from .store import TieredWeightStore


@dataclass
class SchedulerStats:
    hints_received: int = 0
    hints_deduped: int = 0
    chunks_prefetched: int = 0
    errors: int = 0


class PrefetchScheduler:
    """Promotes chunks in background threads ahead of their use.

    `workers` controls how many promotions may be in flight. More than two or
    three rarely helps: the copy engine and the NVMe queue are the shared
    bottleneck, and extra threads only add contention for pinned staging
    buffers.
    """

    def __init__(
        self,
        store: TieredWeightStore,
        workers: int = 2,
        queue_depth: int = 32,
    ):
        self.store = store
        self.stats = SchedulerStats()

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=queue_depth)
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._active = 0
        self._shutdown = False

        self._threads = [
            threading.Thread(target=self._worker, name=f"corestream-prefetch-{i}", daemon=True)
            for i in range(workers)
        ]
        for t in self._threads:
            t.start()

    def hint(self, keys: list[str]) -> None:
        """Request that these chunks be staged, in priority order.

        Non-blocking and best-effort: if the queue is full the hint is
        dropped rather than stalling the caller. A dropped hint costs a cache
        miss later, whereas blocking the compute thread costs throughput
        immediately -- so dropping is the correct trade.
        """
        for key in keys:
            with self._lock:
                if self._shutdown:
                    return
                self.stats.hints_received += 1
                if key in self._pending or self.store.tier_of(key).name == "HOT":
                    self.stats.hints_deduped += 1
                    continue
                self._pending.add(key)
            try:
                self._queue.put_nowait(key)
            except queue.Full:
                with self._lock:
                    self._pending.discard(key)
                return

    def _worker(self) -> None:
        while True:
            key = self._queue.get()
            if key is None:
                self._queue.task_done()
                return
            with self._lock:
                self._active += 1
            try:
                self.store.prefetch(key)
                self.stats.chunks_prefetched += 1
            except Exception:
                # A failed prefetch is recoverable: `get` will retry
                # synchronously and surface the error to the caller there,
                # where it can be attributed to a specific step.
                self.stats.errors += 1
            finally:
                with self._idle:
                    self._pending.discard(key)
                    self._active -= 1
                    self._idle.notify_all()
                self._queue.task_done()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until every queued prefetch has finished. Mainly for tests."""
        with self._idle:
            return self._idle.wait_for(
                lambda: not self._pending and self._active == 0, timeout=timeout
            )

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=5.0)

    def __enter__(self) -> "PrefetchScheduler":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
