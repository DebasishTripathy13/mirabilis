"""Predict throughput before downloading anything.

Decoding one token reads every weight the token uses, and does very little
arithmetic per byte. Throughput is therefore set by how fast those bytes can be
delivered, not by how fast the machine can compute -- which means it can be
estimated from file size and memory tiers alone, before a single byte is
fetched.

The estimate is deliberately rough. Its job is to make the difference between
2 tok/s and 12 tok/s visible while there is still a choice to make, not to
predict the third decimal.
"""

from __future__ import annotations

from .hardware import Hardware

# Measured on an RTX 3060 Laptop with DDR5, and representative of that class of
# machine. The ratios matter far more than the absolute values: VRAM is roughly
# eight times RAM, and RAM is roughly twelve times NVMe. Those two cliffs are
# what the estimate is really reporting.
VRAM_GIBS = 292.0
RAM_GIBS = 36.0
NVME_GIBS = 2.9

# Architectures that route each token through a subset of their parameters.
_MOE_ARCHITECTURES = (
    "qwen3next", "qwen2moe", "qwen3moe", "qwen35moe", "qwen3vlmoe",
    "mixtral", "deepseek2", "deepseek3", "granitemoe", "olmoe", "phimoe",
    "gptoss", "gpt_oss", "jamba", "dbrx", "arctic", "hunyuan_moe",
    "llama4", "ernie4_5_moe", "smallthinker", "bailingmoe", "glm4moe",
)

# Typical share of parameters a mixture-of-experts model actually reads per
# token. Used only when the real expert counts are unknown, which is the case
# before the file is downloaded.
_DEFAULT_ACTIVE_FRACTION = 0.06


def is_moe_architecture(architecture: str) -> bool:
    name = (architecture or "").lower()
    return any(tag in name for tag in _MOE_ARCHITECTURES)


def tokens_per_second(hw: Hardware, file_gib: float, *,
                      is_moe: bool = False,
                      active_fraction: float | None = None) -> float:
    """Rough decode rate for a model of this size on this machine."""
    if file_gib <= 0:
        return 0.0

    if is_moe:
        fraction = active_fraction or _DEFAULT_ACTIVE_FRACTION
        # Attention and norms are read every token regardless of routing, and
        # are small enough to sit in VRAM; the experts are what stream.
        bytes_per_token = file_gib * max(fraction, 0.02)
    else:
        bytes_per_token = file_gib

    # Split the model across tiers the way the engine will: as much as fits in
    # VRAM, then RAM, then whatever is left comes off disk.
    on_gpu = min(file_gib, max(0.0, hw.usable_vram_gib))
    on_ram = min(max(0.0, file_gib - on_gpu), max(0.0, hw.usable_ram_gib))
    on_disk = max(0.0, file_gib - on_gpu - on_ram)

    if file_gib <= 0:
        return 0.0
    gpu_share, ram_share, disk_share = (on_gpu / file_gib,
                                        on_ram / file_gib,
                                        on_disk / file_gib)

    seconds = bytes_per_token * (
        gpu_share / VRAM_GIBS + ram_share / RAM_GIBS + disk_share / NVME_GIBS
    )
    # Attention, sampling, and framework overhead. Small next to the transfer
    # for any model worth this treatment, but not zero.
    seconds += 0.010
    return 1.0 / seconds if seconds > 0 else 0.0


def describe(hw: Hardware, file_gib: float, *, is_moe: bool = False,
             active_fraction: float | None = None) -> str:
    rate = tokens_per_second(hw, file_gib, is_moe=is_moe,
                             active_fraction=active_fraction)
    if rate >= 1.0:
        return f"~{rate:.0f} tok/s"
    return f"~{rate:.1f} tok/s"
