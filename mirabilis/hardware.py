"""What this machine can do, and what follows from it.

Every tuning decision downstream comes from three numbers: how much RAM is
free, how much VRAM is free, and how many *fast* cores there are. The last one
matters more than it looks -- on a hybrid CPU, adding the slow E-cores to a
memory-bound workload makes it slower, not faster.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

GIB = 1024**3


@dataclass
class Hardware:
    ram_total_gib: float
    ram_available_gib: float
    vram_total_gib: float
    vram_free_gib: float
    physical_cores: int
    logical_cores: int
    performance_cores: int
    gpu_name: str
    disk_free_gib: float
    # CPU ids, split by class. One entry per *physical* fast core, plus the
    # slower cores separately: decode is memory-bound, so what matters is how
    # many independent load/store paths are working, not how many threads
    # exist. Hyperthread siblings share those paths and add contention rather
    # than bandwidth.
    fast_core_ids: list[int] = field(default_factory=list)
    slow_core_ids: list[int] = field(default_factory=list)
    governor: str = ""
    min_perf_pct: int = 0

    @property
    def governor_warning(self) -> str:
        """Whether the CPU is allowed to downclock during inference.

        Memory-bound decode stalls on RAM constantly, and a scheduler-driven
        governor reads those stalls as idleness and drops the clock -- so the
        compute between stalls then runs slow too. Measured on this machine
        mid-generation, half the P-cores sat at 400 MHz against a 4.9 GHz
        ceiling.
        """
        if self.governor and self.governor != "performance":
            extra = (f", and cores may drop to {self.min_perf_pct}% of peak"
                     if self.min_perf_pct and self.min_perf_pct < 50 else "")
            return (f"CPU governor is '{self.governor}'{extra}. Memory-bound "
                    "decode stalls on RAM, which the governor reads as idle "
                    "and downclocks for. Setting it to 'performance' keeps the "
                    "clocks up:\n"
                    "  echo performance | sudo tee "
                    "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
        return ""

    def affinity_mask(self, threads: int) -> str:
        """Hex CPU mask for `threads`, filling fast cores before slow ones."""
        chosen = (self.fast_core_ids + self.slow_core_ids)[:threads]
        mask = 0
        for cpu in chosen:
            mask |= 1 << cpu
        return f"{mask:X}" if mask else ""

    @property
    def suggested_threads(self) -> int:
        """Threads to use when nothing has been measured.

        One per physical fast core, plus two slow cores. Measured on an
        i9-12900H: 6 physical P-cores alone gave 24.4 tok/s, adding two
        E-cores gave 26.5, and adding the rest fell back to 21.6. Enough
        independent traffic to saturate the bus, not so much that it jams.
        """
        fast = len(self.fast_core_ids)
        if not fast:
            return max(1, self.physical_cores)
        return fast + min(2, len(self.slow_core_ids))

    @property
    def has_gpu(self) -> bool:
        return self.vram_total_gib > 0

    @property
    def threads(self) -> int:
        """Threads to hand the inference engine.

        Measured on an i9-12900H: 14 threads gave 12.3 tok/s while 20 gave
        9.2. Decode is memory-bound, so the eight slow E-cores end up stalling
        the six fast ones rather than adding throughput. Physical cores is the
        right default; hyperthreads add contention for the same load/store
        units without adding bandwidth.
        """
        if self.fast_core_ids:
            return self.suggested_threads
        return max(1, self.physical_cores)

    @property
    def usable_ram_gib(self) -> float:
        """RAM a model may occupy without pushing the machine into swap."""
        return max(0.0, self.ram_available_gib - 2.0)

    @property
    def usable_vram_gib(self) -> float:
        """VRAM available for weights, after KV cache and activations."""
        return max(0.0, self.vram_free_gib - 1.2)

    def summary(self) -> str:
        lines = [
            f"GPU        {self.gpu_name}",
            f"VRAM       {self.vram_free_gib:.1f} GiB free of {self.vram_total_gib:.1f}",
            f"RAM        {self.ram_available_gib:.1f} GiB available of {self.ram_total_gib:.1f}",
            f"CPU        {self.physical_cores} cores / {self.logical_cores} threads"
            + (f"  ({self.performance_cores} performance)" if self.performance_cores else ""),
            f"threads    {self.threads} (used for inference)",
            f"disk free  {self.disk_free_gib:.0f} GiB",
        ]
        if self.governor:
            lines.append(f"governor   {self.governor}"
                         + (f" (floor {self.min_perf_pct}%)" if self.min_perf_pct else ""))
        return "\n".join(lines)


def _meminfo() -> tuple[float, float]:
    total = available = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key == "MemTotal":
                    total = int(rest.split()[0]) * 1024 / GIB
                elif key == "MemAvailable":
                    available = int(rest.split()[0]) * 1024 / GIB
    except OSError:
        pass
    return total, available


def _gpu() -> tuple[str, float, float]:
    if not shutil.which("nvidia-smi"):
        return "none", 0.0, 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        if not out:
            return "none", 0.0, 0.0
        name, total, free = [p.strip() for p in out[0].split(",")]
        return name, float(total) / 1024, float(free) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return "none", 0.0, 0.0


def _power_policy() -> tuple[str, int]:
    """Current CPU frequency governor and the floor it may drop to."""
    governor = ""
    min_pct = 0
    try:
        path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        if os.path.exists(path):
            with open(path) as f:
                governor = f.read().strip()
        pct = "/sys/devices/system/cpu/intel_pstate/min_perf_pct"
        if os.path.exists(pct):
            with open(pct) as f:
                min_pct = int(f.read().strip())
    except (OSError, ValueError):
        pass
    return governor, min_pct


def _core_ids() -> tuple[list[int], list[int]]:
    """Split logical CPUs into (one id per fast physical core, slow core ids).

    Fast cores are identified by peak clock, which is how Intel's hybrid parts
    separate P from E; there is no portable flag naming them. Hyperthread
    siblings are collapsed to one id each, because two threads on one core
    share the load/store units that decode is actually waiting on.
    """
    fast: list[int] = []
    slow: list[int] = []
    try:
        freqs: dict[int, int] = {}
        for cpu in range(os.cpu_count() or 1):
            path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
            if os.path.exists(path):
                with open(path) as f:
                    freqs[cpu] = int(f.read().strip())
        if not freqs:
            return [], []
        top = max(freqs.values())
        seen_cores: set[str] = set()
        for cpu in sorted(freqs):
            siblings_path = (
                f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            )
            key = str(cpu)
            if os.path.exists(siblings_path):
                with open(siblings_path) as f:
                    key = f.read().strip()
            if freqs[cpu] >= top * 0.95:
                if key in seen_cores:
                    continue          # a hyperthread sibling of one already taken
                seen_cores.add(key)
                fast.append(cpu)
            else:
                slow.append(cpu)
    except (OSError, ValueError):
        return [], []
    return fast, slow


def _cores() -> tuple[int, int, int]:
    """Return (physical, logical, performance) core counts.

    Performance cores are identified by their maximum clock: on Intel hybrid
    parts the P-cores boost materially higher than the E-cores, and there is no
    portable flag that names them directly.
    """
    logical = os.cpu_count() or 1
    physical = logical
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=20).stdout
        cores_per_socket = re.search(r"Core\(s\) per socket:\s+(\d+)", out)
        sockets = re.search(r"Socket\(s\):\s+(\d+)", out)
        if cores_per_socket and sockets:
            physical = int(cores_per_socket.group(1)) * int(sockets.group(1))
    except (OSError, subprocess.SubprocessError):
        pass

    performance = 0
    try:
        freqs: dict[int, int] = {}
        for cpu in range(logical):
            path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
            if os.path.exists(path):
                with open(path) as f:
                    freqs[cpu] = int(f.read().strip())
        if freqs:
            top = max(freqs.values())
            performance = sum(1 for v in freqs.values() if v >= top * 0.95)
    except (OSError, ValueError):
        pass

    return physical, logical, performance


def detect(probe_path: str = ".") -> Hardware:
    total, available = _meminfo()
    name, vram_total, vram_free = _gpu()
    physical, logical, performance = _cores()
    fast_ids, slow_ids = _core_ids()
    governor, min_pct = _power_policy()
    try:
        disk_free = shutil.disk_usage(probe_path).free / GIB
    except OSError:
        disk_free = 0.0
    return Hardware(
        ram_total_gib=total,
        ram_available_gib=available,
        vram_total_gib=vram_total,
        vram_free_gib=vram_free,
        physical_cores=physical,
        logical_cores=logical,
        performance_cores=performance,
        gpu_name=name,
        disk_free_gib=disk_free,
        fast_core_ids=fast_ids,
        slow_core_ids=slow_ids,
        governor=governor,
        min_perf_pct=min_pct,
    )
