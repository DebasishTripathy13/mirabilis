"""Hardware profiling and roofline estimation.

Streaming inference is bandwidth-bound, not compute-bound: per token the
engine must move every weight the token touches across the slowest tier
boundary in its path. So the honest question to answer *before* downloading
tens of gigabytes is:

    max_tokens_per_sec = effective_bandwidth / bytes_touched_per_token

This module measures the real bandwidths on the current machine rather than
trusting spec-sheet numbers, which overstate PCIe by roughly 2x once
protocol overhead and pageable-memory staging are accounted for.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field

GIB = 1024**3
MIB = 1024**2

# Fraction of free VRAM held back for the KV cache, activations, and
# allocator fragmentation. Weight caching may use the rest. Without an
# explicit reserve, long contexts OOM mid-generation because the KV cache
# grows into space the weight cache already claimed.
DEFAULT_KV_RESERVE_FRACTION = 0.35

# Leave headroom so filling the WARM tier never pushes the machine into swap.
DEFAULT_RAM_HEADROOM_GIB = 4.0


@dataclass
class HardwareProfile:
    """Measured capabilities of the current machine."""

    has_cuda: bool
    gpu_name: str
    vram_total_bytes: int
    vram_free_bytes: int
    ram_total_bytes: int
    ram_available_bytes: int
    disk_free_bytes: int
    # Measured, not nominal.
    pcie_pinned_bytes_per_sec: float
    disk_read_bytes_per_sec: float

    def hot_budget_bytes(
        self, kv_reserve_fraction: float = DEFAULT_KV_RESERVE_FRACTION
    ) -> int:
        """Bytes of VRAM available for caching weights.

        The KV cache and activations are carved out first; what remains is
        what the HOT tier may occupy.
        """
        if not self.has_cuda:
            return 0
        usable = self.vram_free_bytes * (1.0 - kv_reserve_fraction)
        return max(0, int(usable))

    def warm_budget_bytes(
        self, headroom_gib: float = DEFAULT_RAM_HEADROOM_GIB
    ) -> int:
        """Bytes of host RAM that may hold resident model weights.

        This is advisory: the WARM tier is backed by mmap and the kernel page
        cache, so the OS reclaims pages under pressure rather than OOMing. We
        report it so the CLI can tell the user whether a model will stay fully
        resident or thrash against disk.
        """
        return max(0, int(self.ram_available_bytes - headroom_gib * GIB))


@dataclass
class Roofline:
    """Theoretical ceiling for a given model shape on this machine."""

    bytes_per_token: int
    fits_in_ram: bool
    # The tier that actually limits throughput.
    bound_by: str
    bandwidth_bytes_per_sec: float
    max_tokens_per_sec: float
    notes: list[str] = field(default_factory=list)


def _measure_pcie_bandwidth(
    sample_bytes: int = 256 * MIB, repeats: int = 3
) -> float:
    """Measure real pinned host->device throughput.

    Pinned (page-locked) memory is used because that is what the scheduler
    uses in production; pageable memory measures roughly half this rate
    because the driver stages it through an internal pinned buffer first.
    """
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0

    host = torch.empty(sample_bytes, dtype=torch.uint8, pin_memory=True)
    device = torch.empty(sample_bytes, dtype=torch.uint8, device="cuda")

    # Warm up the copy engine and the allocator before timing.
    device.copy_(host, non_blocking=True)
    torch.cuda.synchronize()

    best = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        device.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if elapsed > 0:
            best = max(best, sample_bytes / elapsed)

    del device
    torch.cuda.empty_cache()
    return best


def _measure_disk_read(path: str, sample_bytes: int = 256 * MIB) -> float:
    """Measure cold sequential read throughput at `path`.

    O_DIRECT would be the rigorous way to bypass the page cache, but it
    imposes alignment constraints that vary by filesystem. Instead we write a
    fresh file and read it back with POSIX_FADV_DONTNEED, which is portable
    across ext4/xfs/btrfs and close enough for capacity planning.
    """
    tmp_dir = path if os.path.isdir(path) else os.path.dirname(path) or "."
    fd = None
    tmp_path = None
    try:
        handle, tmp_path = tempfile.mkstemp(dir=tmp_dir, prefix=".corestream-probe-")
        os.close(handle)

        block = os.urandom(MIB)
        with open(tmp_path, "wb") as f:
            for _ in range(sample_bytes // MIB):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())

        fd = os.open(tmp_path, os.O_RDONLY)
        # Evict what we just wrote so we measure the device, not the cache.
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

        start = time.perf_counter()
        read = 0
        while True:
            chunk = os.read(fd, 8 * MIB)
            if not chunk:
                break
            read += len(chunk)
        elapsed = time.perf_counter() - start
        return read / elapsed if elapsed > 0 else 0.0
    except OSError:
        return 0.0
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _read_meminfo() -> tuple[int, int]:
    """Return (total, available) host RAM in bytes from /proc/meminfo."""
    total = available = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key == "MemTotal":
                    total = int(rest.split()[0]) * 1024
                elif key == "MemAvailable":
                    available = int(rest.split()[0]) * 1024
                if total and available:
                    break
    except OSError:
        pass
    return total, available


def profile(probe_path: str = ".", run_benchmarks: bool = True) -> HardwareProfile:
    """Measure this machine's capabilities.

    Set `run_benchmarks=False` to skip the bandwidth probes (which take a few
    seconds and allocate 256 MiB) and report topology only.
    """
    has_cuda = False
    gpu_name = "none"
    vram_total = vram_free = 0

    try:
        import torch

        if torch.cuda.is_available():
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_free, vram_total = torch.cuda.mem_get_info(0)
    except ImportError:
        pass

    ram_total, ram_available = _read_meminfo()
    disk_free = shutil.disk_usage(probe_path).free

    pcie_bw = _measure_pcie_bandwidth() if (run_benchmarks and has_cuda) else 0.0
    disk_bw = _measure_disk_read(probe_path) if run_benchmarks else 0.0

    return HardwareProfile(
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        disk_free_bytes=disk_free,
        pcie_pinned_bytes_per_sec=pcie_bw,
        disk_read_bytes_per_sec=disk_bw,
    )


def roofline(
    profile: HardwareProfile,
    bytes_per_token: int,
    model_total_bytes: int,
    hot_cache_bytes: int = 0,
) -> Roofline:
    """Compute the throughput ceiling for a model shape on this machine.

    `bytes_per_token` is what the model actually touches per token: the full
    weight set for a dense model, but only the active experts plus shared
    layers for an MoE. That distinction is the single largest factor in
    achievable throughput, which is why MoE models are the interesting case
    for streaming inference.

    `hot_cache_bytes` is the portion of those touched bytes already resident
    in VRAM, which therefore does not cross the bus at all.
    """
    notes: list[str] = []

    streamed = max(0, bytes_per_token - hot_cache_bytes)
    if streamed == 0:
        return Roofline(
            bytes_per_token=bytes_per_token,
            fits_in_ram=True,
            bound_by="compute",
            bandwidth_bytes_per_sec=float("inf"),
            max_tokens_per_sec=float("inf"),
            notes=["Entire working set fits in VRAM; not bandwidth-bound."],
        )

    fits_in_ram = model_total_bytes <= profile.warm_budget_bytes()

    if not profile.has_cuda:
        # No GPU: weights are read into CPU-addressable memory only, so the
        # relevant bandwidth is disk (cold) or effectively unlimited (cached).
        bandwidth = profile.disk_read_bytes_per_sec or 1.0
        bound_by = "disk (no CUDA)"
        notes.append("No CUDA device; running CPU-only.")
    elif fits_in_ram:
        bandwidth = profile.pcie_pinned_bytes_per_sec or 1.0
        bound_by = "pcie"
        notes.append(
            "Model fits in RAM; after warmup weights stream over PCIe, not disk."
        )
    else:
        bandwidth = profile.disk_read_bytes_per_sec or 1.0
        bound_by = "disk"
        notes.append(
            "Model exceeds available RAM; every token pays disk reads. "
            "A smaller quantization would move this to the PCIe-bound regime "
            "and is typically several times faster."
        )

    return Roofline(
        bytes_per_token=bytes_per_token,
        fits_in_ram=fits_in_ram,
        bound_by=bound_by,
        bandwidth_bytes_per_sec=bandwidth,
        max_tokens_per_sec=bandwidth / streamed,
        notes=notes,
    )


def format_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"
