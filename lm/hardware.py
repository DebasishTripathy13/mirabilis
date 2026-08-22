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
from dataclasses import dataclass

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
    )
