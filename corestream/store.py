"""Tiered weight storage: VRAM (HOT) / host RAM (WARM) / disk (COLD).

The design borrows directly from tiered caching in CDNs and from OS virtual
memory:

*   **COLD -> WARM is delegated to the kernel.** Chunks are read through
    `mmap`, so the page cache holds recently-touched weights in RAM and
    performs readahead for us. Reimplementing that in Python would be both
    slower and less correct under memory pressure, and with enough free RAM a
    model becomes fully resident after one pass at zero implementation cost.

*   **WARM -> HOT is ours.** Host memory is staged through a pinned buffer
    pool and copied to the device on a dedicated CUDA stream, so promotion
    overlaps with compute instead of serialising against it.

*   **Admission is LFU-weighted, eviction is LRU.** A chunk earns VRAM
    residency only after it has been touched `admission_threshold` times.
    Pure LRU admits on first touch, which lets a one-off chunk evict a
    genuinely hot one -- the scan-resistance problem CDN cache-admission
    policies exist to solve. For an MoE model, where expert activation is
    heavily skewed, this is the difference between caching the experts that
    matter and thrashing on the tail.
"""

from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Protocol

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard runtime dependency
    torch = None  # type: ignore[assignment]


class Tier(Enum):
    COLD = "cold"  # on disk, reachable via mmap
    WARM = "warm"  # resident in host RAM (kernel page cache)
    PINNED = "pinned"  # resident in page-locked host RAM, DMA-ready
    HOT = "hot"  # resident in VRAM


class PinnedHostCache:
    """Page-locked host memory holding chunks in DMA-ready form.

    A transfer out of ordinary (pageable) host memory cannot be a straight
    DMA: the pages may be swapped or moved, so the data is first copied into
    a page-locked staging buffer and only then handed to the copy engine.
    That intermediate copy is CPU work sitting directly on the critical path,
    and it is the reason the engine plateaus around 6 GiB/s while the
    hardware sustains over 9 GiB/s for the same transfer size.

    Keeping a chunk permanently in pinned memory removes the copy entirely --
    the copy engine reads host RAM directly. The cost is that pinned pages
    cannot be swapped, so the budget must stay well inside physical RAM.

    Memory comes from a single large allocation carved into fixed slots.
    Pinning is a slow kernel operation, so one big `cudaHostAlloc` at startup
    is far cheaper than thousands of small ones during inference.

    The cache never evicts. Eviction would let a slot be overwritten while a
    previous DMA out of it was still in flight, which needs per-slot event
    tracking to do safely -- complexity that buys little, since the skewed
    access that makes an MoE worth caching also means the first chunks to
    earn a slot are the ones that keep being used.
    """

    def __init__(self, budget_bytes: int, slot_bytes: int, enabled: bool = True):
        self.enabled = (
            enabled
            and torch is not None
            and torch.cuda.is_available()
            and budget_bytes > 0
            and slot_bytes > 0
        )
        self.slot_bytes = slot_bytes
        self.capacity = int(budget_bytes // slot_bytes) if self.enabled else 0
        self._slab: "torch.Tensor | None" = None
        self._entries: dict[str, torch.Tensor] = {}
        self._next_slot = 0
        self._lock = threading.Lock()

        if self.capacity > 0:
            try:
                self._slab = torch.empty(
                    self.capacity * slot_bytes, dtype=torch.uint8, pin_memory=True
                )
            except (RuntimeError, MemoryError):
                # Pinning can fail if the budget exceeds what the OS will
                # lock. Degrading to the staging path is slower but correct.
                self.enabled = False
                self.capacity = 0

    def get(self, key: str) -> "torch.Tensor | None":
        if not self.enabled:
            return None
        return self._entries.get(key)

    def put(self, key: str, host: "torch.Tensor") -> "torch.Tensor | None":
        """Copy a chunk into pinned memory once, returning the pinned view."""
        if not self.enabled or self._slab is None:
            return None
        nbytes = host.numel() * host.element_size()
        if nbytes > self.slot_bytes:
            return None

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                return existing
            if self._next_slot >= self.capacity:
                return None
            index = self._next_slot
            self._next_slot += 1
            offset = index * self.slot_bytes
            view = (
                self._slab[offset : offset + nbytes]
                .view(host.dtype)
                .view(host.shape)
            )
            self._entries[key] = view

        # The one-time copy runs outside the lock: it is a plain memcpy, and
        # holding the lock across it would serialise every worker behind it.
        view.copy_(host)
        return view

    @property
    def resident(self) -> int:
        return len(self._entries)

    @property
    def full(self) -> bool:
        return self._next_slot >= self.capacity


class ChunkSource(Protocol):
    """Supplies chunk weights as host tensors.

    Implementations are expected to be mmap-backed so that the kernel page
    cache provides the WARM tier. A purely in-memory implementation is used
    in tests.
    """

    def keys(self) -> list[str]: ...

    def nbytes(self, key: str) -> int: ...

    def host_tensor(self, key: str) -> "torch.Tensor":
        """Return the chunk as a CPU tensor. May block on disk I/O."""
        ...


@dataclass
class StoreStats:
    """Counters that determine whether the cache is earning its keep."""

    hot_hits: int = 0
    staged_hits: int = 0
    misses: int = 0
    admissions: int = 0
    evictions: int = 0
    staged_discards: int = 0
    # The two figures that fully describe bus usage. `bytes_demanded` is what
    # the model asked for; `bytes_promoted` is what actually crossed the bus.
    # Their difference is signed and meaningful in both directions: demanded
    # above moved is what caching saved, moved above demanded is redundant
    # traffic. Tracking a separate "bytes saved" counter instead invites
    # double-counting, since a chunk that is prefetched and then read scores
    # once as a transfer and once as a hit for the very same bytes.
    bytes_demanded: int = 0
    bytes_promoted: int = 0

    @property
    def total_requests(self) -> int:
        return self.hot_hits + self.staged_hits + self.misses

    @property
    def stall_free_rate(self) -> float:
        """Share of requests that found their chunk already on the device.

        Measures prefetch coverage -- whether the scheduler stayed ahead of
        execution -- and counts staged hits, since a chunk transferred early
        still spares the caller a synchronous wait.

        This is *not* a measure of bytes saved: a prefetched chunk crossed
        the bus, just sooner. `bytes_saved` is the bandwidth figure; keeping
        the two apart is what stops a 99% "hit rate" from coexisting with
        zero actual savings.
        """
        served = self.hot_hits + self.staged_hits
        return served / self.total_requests if self.total_requests else 0.0

    @property
    def savings_rate(self) -> float:
        """Share of demanded bytes that never crossed the bus."""
        if not self.bytes_demanded:
            return 0.0
        return self.bytes_saved / self.bytes_demanded

    @property
    def bytes_saved(self) -> int:
        """Demanded bytes that never crossed the bus, thanks to caching.

        The number that actually moves achievable tok/s, since throughput is
        bandwidth-bound.
        """
        return max(0, self.bytes_demanded - self.bytes_promoted)

    @property
    def bytes_wasted(self) -> int:
        """Bytes moved beyond demand: redundant or mispredicted transfers."""
        return max(0, self.bytes_promoted - self.bytes_demanded)


class AdmissionPolicy(Protocol):
    def should_admit(self, key: str, hits: int) -> bool: ...


class EvictionPolicy(Protocol):
    def select_victim(
        self, resident: "OrderedDict[str, torch.Tensor]", incoming: str
    ) -> str | None:
        """Choose a chunk to evict, or None to refuse the incoming chunk."""
        ...


class LRUEviction:
    """Evict the least recently used chunk.

    Correct when access is skewed -- an MoE router concentrates traffic on a
    minority of experts, so recency predicts reuse.
    """

    def select_victim(
        self, resident: "OrderedDict[str, torch.Tensor]", incoming: str
    ) -> str | None:
        return next(iter(resident), None)


class StaticPinning:
    """Fill the cache once, then never evict.

    Cyclic access larger than the cache is the pathological case for LRU: by
    the time a layer is revisited it has been evicted by the ones that
    followed it, so the hit rate collapses to zero no matter how large the
    cache is (short of holding everything). This is the classic sequential-
    scan defeat of LRU, and it is exactly what a dense model does -- every
    layer, in fixed order, once per token.

    Refusing to evict converts that 0% into `cache_size / working_set`,
    because a fixed resident subset is hit on every pass. For a 3.6 GiB cache
    over a 15.6 GiB model that is the difference between saving nothing and
    saving roughly a quarter of all traffic.
    """

    def select_victim(
        self, resident: "OrderedDict[str, torch.Tensor]", incoming: str
    ) -> str | None:
        return None


@dataclass
class LFUAdmission:
    """Admit a chunk to VRAM only once it has proven repeat demand.

    `threshold=1` degenerates to admit-on-first-touch, which is the right
    behaviour for a dense model (every layer is touched exactly once per
    token, in a fixed cycle, so there is no tail to protect against).
    """

    threshold: int = 2

    def should_admit(self, key: str, hits: int) -> bool:
        return hits >= self.threshold


@dataclass
class StagingSlot:
    """A pinned host buffer plus the event marking its last copy-out."""

    buffer: "torch.Tensor"
    event: "torch.cuda.Event"
    dirty: bool = False


class PinnedStagingPool:
    """A recycled ring of page-locked host buffers, guarded by CUDA events.

    Pinned memory is required for `cudaMemcpyAsync` to run asynchronously and
    at full bandwidth; a copy from pageable memory is staged through a
    driver-internal bounce buffer and measures roughly two thirds the rate.
    Pinned allocation is expensive and the total is limited, so buffers are
    allocated once and recycled.

    Recycling is where the events matter. A buffer must not be overwritten
    while its previous copy is still in flight. The obvious guard --
    synchronising the copy stream after every transfer -- costs far more than
    it appears to: measured on an RTX 3060, per-chunk synchronisation caps
    throughput at 5.0 GiB/s against 7.9 GiB/s without it, because each
    synchronise drains the pipeline and leaves the copy engine idle while the
    CPU catches up.

    Recording an event per slot instead moves the guard to the point of
    reuse. With enough slots the copy has long finished by the time the ring
    comes round, so the wait is free and the copy engine never drains.
    """

    def __init__(self, slot_bytes: int, slots: int = 8, enabled: bool = True):
        self.slot_bytes = slot_bytes
        self.enabled = enabled and torch is not None and torch.cuda.is_available()
        self._free: queue.Queue = queue.Queue()
        self.slots = slots
        if self.enabled:
            for _ in range(slots):
                self._free.put(
                    StagingSlot(
                        buffer=torch.empty(
                            slot_bytes, dtype=torch.uint8, pin_memory=True
                        ),
                        event=torch.cuda.Event(),
                    )
                )

    @contextmanager
    def acquire(self, nbytes: int) -> Iterator["StagingSlot | None"]:
        """Yield a slot whose buffer is safe to overwrite, or None.

        A chunk larger than the slot size falls back to an unpinned copy
        rather than failing: correctness first, bandwidth second.
        """
        if not self.enabled or nbytes > self.slot_bytes:
            yield None
            return
        slot = self._free.get()
        if slot.dirty:
            # Blocks only if this slot's previous copy is somehow still in
            # flight after a full trip round the ring -- rare by design.
            slot.event.synchronize()
            slot.dirty = False
        try:
            yield slot
        finally:
            self._free.put(slot)


class TieredWeightStore:
    """Keeps the hottest chunks in VRAM and streams the rest on demand.

    Thread-safe: the prefetch scheduler promotes chunks from worker threads
    while the main thread executes the model.
    """

    def __init__(
        self,
        source: ChunkSource,
        hot_budget_bytes: int,
        device: str | None = None,
        admission: AdmissionPolicy | None = None,
        eviction: EvictionPolicy | None = None,
        staging_slots: int = 8,
        staged_chunks: int = 8,
        pinned_budget_bytes: int = 0,
    ):
        if torch is None:  # pragma: no cover
            raise RuntimeError("corestream requires PyTorch")

        self.source = source
        self.hot_budget_bytes = hot_budget_bytes
        self.admission = admission or LFUAdmission()
        self.eviction = eviction or LRUEviction()
        self.stats = StoreStats()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Insertion order doubles as LRU order; `move_to_end` on access.
        self._hot: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hot_bytes = 0
        self._hits: dict[str, int] = {}
        self._lock = threading.RLock()

        # Chunks the scheduler transferred but which have not earned VRAM
        # residency yet. Without this holding area a prefetch that fails the
        # admission test would discard the bytes it just moved, and the
        # subsequent `get` would move them again -- turning prefetching from
        # an optimisation into a straight doubling of bus traffic.
        self._staged: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._staged_bytes = 0
        self._staged_capacity = staged_chunks

        # Per-key locks so two threads racing to promote the same chunk do
        # the disk read and host->device copy once, not twice.
        self._inflight: dict[str, threading.Event] = {}

        largest = max((source.nbytes(k) for k in source.keys()), default=0)
        self.staging = PinnedStagingPool(
            slot_bytes=largest,
            slots=staging_slots,
            enabled=self.device.type == "cuda",
        )

        self.pinned_host = PinnedHostCache(
            budget_bytes=pinned_budget_bytes,
            slot_bytes=largest,
            enabled=self.device.type == "cuda",
        )

        # One copy stream per thread. A shared stream would interleave
        # transfers from different workers, so each would have to wait on the
        # others' copies before its own result was ordered against compute.
        self._thread_local = threading.local()

    # -- introspection -------------------------------------------------

    def tier_of(self, key: str) -> Tier:
        with self._lock:
            if key in self._hot or key in self._staged:
                return Tier.HOT
        return Tier.COLD

    @property
    def hot_keys(self) -> list[str]:
        with self._lock:
            return list(self._hot.keys())

    @property
    def hot_bytes(self) -> int:
        with self._lock:
            return self._hot_bytes

    # -- core access ---------------------------------------------------

    def get(self, key: str) -> "torch.Tensor":
        """Return the chunk on the compute device, loading it if necessary.

        Blocks only for the portion of the transfer the scheduler has not
        already completed. If the scheduler prefetched this chunk, this is a
        dictionary lookup.
        """
        with self._lock:
            self._hits[key] = self._hits.get(key, 0) + 1
            hits = self._hits[key]
            self.stats.bytes_demanded += self.source.nbytes(key)
            found = self._lookup_locked(key)
            if found is not None:
                if self.admission.should_admit(key, hits):
                    self._admit(key, found)
                return found
            inflight = self._inflight.get(key)

        if inflight is not None:
            # A prefetch for this chunk is already moving its bytes. Waiting
            # for it is strictly better than starting a second transfer of
            # the same data: on a bandwidth-bound engine the duplicate would
            # not merely waste capacity, it would contend with the very
            # transfer being waited on and make it finish later.
            inflight.wait(timeout=60.0)
            with self._lock:
                found = self._lookup_locked(key)
                if found is not None:
                    if self.admission.should_admit(key, hits):
                        self._admit(key, found)
                    return found

        with self._lock:
            self.stats.misses += 1

        device_tensor = self._transfer(key)

        if self.admission.should_admit(key, hits):
            self._admit(key, device_tensor)
        return device_tensor

    def _lookup_locked(self, key: str) -> "torch.Tensor | None":
        """Return a chunk already on the device, or None. Caller holds the lock."""
        cached = self._hot.get(key)
        if cached is not None:
            self._hot.move_to_end(key)
            self.stats.hot_hits += 1
            return cached

        staged = self._staged.pop(key, None)
        if staged is not None:
            self._staged_bytes -= staged.numel() * staged.element_size()
            self.stats.staged_hits += 1
            # Deliberately not counted as bytes saved: these bytes did cross
            # the bus, just earlier, during prefetch. Prefetching moves a
            # transfer off the critical path; only a resident chunk avoids
            # the transfer altogether.
            return staged
        return None

    def prefetch(self, key: str) -> None:
        """Stage a chunk ahead of use. Safe to call from any thread.

        Called by the scheduler for chunks the plan says are coming up. A
        chunk that is already HOT, or already being promoted by another
        thread, is a no-op.
        """
        with self._lock:
            if key in self._hot or key in self._staged or key in self._inflight:
                return
            event = threading.Event()
            self._inflight[key] = event

        try:
            tensor = self._transfer(key)
            with self._lock:
                hits = self._hits.get(key, 0)
            # Prefetch counts toward admission so that a chunk the plan keeps
            # requesting becomes resident, rather than being re-staged every
            # step because only `get` ever incremented its score.
            # Admission can be refused twice over: by policy (not enough
            # demand yet) or by capacity (no room and nothing evictable). A
            # chunk that fails either way must still be held, or the bytes
            # just moved are thrown away and the following `get` moves them
            # again.
            admitted = self.admission.should_admit(
                key, hits + 1
            ) and self._admit(key, tensor)
            if not admitted:
                self._stage(key, tensor)
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()

    # -- internals -----------------------------------------------------

    @property
    def _copy_stream(self) -> "torch.cuda.Stream":
        stream = getattr(self._thread_local, "stream", None)
        if stream is None:
            stream = torch.cuda.Stream()
            self._thread_local.stream = stream
        return stream

    def _transfer(self, key: str) -> "torch.Tensor":
        """Move one chunk from its backing store to the compute device.

        Contains no CPU-side synchronisation. Ordering against compute is
        established on the GPU: the compute stream is told to wait for the
        copy stream, which costs nothing on the CPU, rather than the CPU
        blocking until the copy lands.
        """
        host = self.source.host_tensor(key)

        if self.device.type != "cuda":
            self.stats.bytes_promoted += host.numel() * host.element_size()
            return host

        nbytes = host.numel() * host.element_size()
        stream = self._copy_stream

        # Fast path: the chunk already lives in page-locked memory, so the
        # copy engine can read it directly and no CPU copy is needed at all.
        pinned = self.pinned_host.get(key)
        if pinned is None and not self.pinned_host.full:
            pinned = self.pinned_host.put(key, host)
        if pinned is not None:
            with torch.cuda.stream(stream):
                device_tensor = torch.empty(
                    host.shape, dtype=host.dtype, device=self.device
                )
                device_tensor.copy_(pinned, non_blocking=True)
            device_tensor.record_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(stream)
            self.stats.bytes_promoted += nbytes
            return device_tensor

        with self.staging.acquire(nbytes) as slot:
            if slot is None:
                device_tensor = host.to(self.device, non_blocking=False)
            else:
                staged = slot.buffer[:nbytes].view(host.dtype).view(host.shape)
                staged.copy_(host)
                with torch.cuda.stream(stream):
                    device_tensor = torch.empty(
                        host.shape, dtype=host.dtype, device=self.device
                    )
                    device_tensor.copy_(staged, non_blocking=True)
                    slot.event.record(stream)
                slot.dirty = True
                # The tensor was allocated on the copy stream but will be read
                # on the compute stream. Without this the caching allocator
                # may hand its memory to someone else while the copy is still
                # running.
                device_tensor.record_stream(torch.cuda.current_stream())
                # GPU-side ordering: compute waits for the copy, the CPU does
                # not. This is what removes the per-chunk pipeline drain.
                torch.cuda.current_stream().wait_stream(stream)

        self.stats.bytes_promoted += nbytes
        return device_tensor

    def _admit(self, key: str, tensor: "torch.Tensor") -> bool:
        """Make a chunk VRAM-resident. Returns whether it became resident.

        Callers must respect a False return: the chunk is still only in the
        caller's hands, and dropping it discards a transfer that already
        happened.
        """
        if self.device.type != "cuda" or self.hot_budget_bytes <= 0:
            return False
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.hot_budget_bytes:
            return False  # Single chunk exceeds the budget; never resident.

        with self._lock:
            if key in self._hot:
                return True
            while self._hot_bytes + nbytes > self.hot_budget_bytes and self._hot:
                victim_key = self.eviction.select_victim(self._hot, key)
                if victim_key is None:
                    return False  # Policy refuses to evict; cache stays put.
                victim = self._hot.pop(victim_key)
                self._hot_bytes -= victim.numel() * victim.element_size()
                self.stats.evictions += 1
            if self._hot_bytes + nbytes > self.hot_budget_bytes:
                return False
            self._hot[key] = tensor
            self._hot_bytes += nbytes
            self.stats.admissions += 1
            return True

    def _stage(self, key: str, tensor: "torch.Tensor") -> None:
        """Hold a prefetched chunk until its step consumes it.

        Bounded and FIFO: staged chunks occupy VRAM, so an unbounded holding
        area would become a second cache competing with the real one, and
        would evict by accident rather than by policy. FIFO is right here
        because entries are consumed in the order the plan predicted them --
        an entry still present when the bound is hit was a mispredicted
        prefetch, so it is the correct one to drop.
        """
        if self.device.type != "cuda":
            return
        with self._lock:
            while len(self._staged) >= self._staged_capacity and self._staged:
                _, victim = self._staged.popitem(last=False)
                self._staged_bytes -= victim.numel() * victim.element_size()
                self.stats.staged_discards += 1
            self._staged[key] = tensor
            self._staged_bytes += tensor.numel() * tensor.element_size()

    def clear_hot(self) -> None:
        with self._lock:
            self._hot.clear()
            self._hot_bytes = 0
            self._staged.clear()
            self._staged_bytes = 0
        if torch is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
