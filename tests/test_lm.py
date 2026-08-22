"""Tests for the lm CLI's decision logic.

The parts worth testing are the ones that decide things: which quantization to
install, and where each tensor should live. Both are pure functions of hardware
plus model shape, so they can be checked without a GPU or a model file.
"""

import json
import struct

import pytest

from lm.gguf import read_info
from lm.hardware import Hardware
from lm.hub import Quant, choose_quant
from lm.registry import default_name
from lm.tune import plan


def hw(ram_avail=28.0, vram_free=5.6, cores=14, perf=12, vram_total=6.0):
    return Hardware(
        ram_total_gib=30.5, ram_available_gib=ram_avail,
        vram_total_gib=vram_total, vram_free_gib=vram_free,
        physical_cores=cores, logical_cores=cores + 6,
        performance_cores=perf, gpu_name="test", disk_free_gib=200.0,
    )


class Info:
    """Stand-in for GGUFInfo."""
    def __init__(self, layers=48, experts=0, experts_used=0,
                 architecture="llama", embedding_length=4096, context_length=8192):
        self.layers = layers
        self.experts = experts
        self.experts_used = experts_used
        self.architecture = architecture
        self.embedding_length = embedding_length
        self.context_length = context_length
        self.name = ""

    @property
    def is_moe(self):
        return self.experts > 1


# -- quantization choice ---------------------------------------------------

def test_picks_best_quality_that_fits():
    quants = [
        Quant("IQ2_XXS", 24.4, []),
        Quant("Q2_K_XL", 28.1, []),
        Quant("Q4_K_M", 45.2, []),
    ]
    assert choose_quant(quants, budget_gib=30.3).label == "Q2_K_XL"


def test_does_not_pick_something_that_does_not_fit():
    """Crossing the RAM boundary costs more than the extra bits gain.

    RAM reads roughly twelve times faster than NVMe, so a larger quant that
    spills to disk is slower than a smaller one that does not.
    """
    quants = [Quant("IQ2_XXS", 24.4, []), Quant("Q4_K_M", 45.2, [])]
    assert choose_quant(quants, budget_gib=30.0).label == "IQ2_XXS"


def test_returns_none_when_nothing_fits():
    assert choose_quant([Quant("Q4_K_M", 45.2, [])], budget_gib=8.0) is None


def test_empty_repo_yields_none():
    assert choose_quant([], budget_gib=30.0) is None


# -- placement -------------------------------------------------------------

def test_small_model_goes_entirely_on_gpu():
    p = plan(hw(), Info(layers=32), file_gib=2.0)
    assert p.gpu_layers == 999
    assert p.cpu_moe_layers == 0


def test_dense_model_splits_layers_by_vram():
    p = plan(hw(), Info(layers=40), file_gib=18.5)
    assert 0 < p.gpu_layers < 40
    assert p.cpu_moe_layers == 0


def test_moe_keeps_experts_in_ram_and_attention_on_gpu():
    """The defining placement for MoE: by tensor role, not layer index."""
    p = plan(hw(), Info(layers=48, experts=512, experts_used=10,
                        architecture="qwen3next"), file_gib=28.1)
    assert p.gpu_layers == 999          # attention and norms on GPU
    assert p.cpu_moe_layers > 0         # expert banks in RAM


def test_threads_default_to_physical_cores():
    """Hyperthreads and E-cores measured slower on memory-bound decode."""
    p = plan(hw(cores=14, perf=12), Info(), file_gib=8.0)
    assert p.threads == 14


def test_measured_overrides_beat_the_model():
    p = plan(hw(), Info(layers=48, experts=512, experts_used=10),
             file_gib=28.1, override_ncmoe=36, override_threads=12)
    assert p.cpu_moe_layers == 36
    assert p.threads == 12
    assert "-ncmoe" in p.to_args()
    assert p.to_args()[p.to_args().index("-t") + 1] == "12"


def test_hybrid_attention_gets_a_smaller_kv_reserve():
    """Estimating every layer as full attention starves the weight budget.

    Qwen3-Next and similar interleave linear-attention layers whose state is
    tiny. Reserving for 48 full-attention layers left 1.4 GiB for weights when
    the real figure allowed 4.1 GiB.
    """
    dense_like = plan(hw(), Info(layers=48, experts=512, experts_used=10,
                                 architecture="llama"), file_gib=28.1)
    hybrid = plan(hw(), Info(layers=48, experts=512, experts_used=10,
                             architecture="qwen3next"), file_gib=28.1)
    assert hybrid.cpu_moe_layers < dense_like.cpu_moe_layers


def test_no_gpu_runs_on_cpu():
    p = plan(hw(vram_free=0.0, vram_total=0.0), Info(), file_gib=8.0)
    assert p.gpu_layers == 0
    assert not p.flash_attention


def test_warns_when_experts_exceed_ram():
    p = plan(hw(ram_avail=10.0), Info(layers=48, experts=512, experts_used=10),
             file_gib=45.0)
    assert any("WARNING" in n for n in p.notes)


def test_context_defaults_conservatively():
    """Context competes with weights for the same VRAM."""
    p = plan(hw(), Info(context_length=262144), file_gib=8.0)
    assert p.context == 4096


# -- naming ----------------------------------------------------------------

def test_default_name_is_short_and_includes_quant():
    name = default_name("unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF",
                        "Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL")
    assert name.startswith("qwen3-next-80b-a3b-instruct:")
    assert "q2_k_xl" in name


# -- GGUF parsing ----------------------------------------------------------

def _kv_string(key, value):
    kb = key.encode()
    vb = value.encode()
    return (struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8)
            + struct.pack("<Q", len(vb)) + vb)


def _kv_u32(key, value):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)


def test_reads_moe_metadata(tmp_path):
    path = tmp_path / "m.gguf"
    body = (_kv_string("general.architecture", "qwen3next")
            + _kv_u32("qwen3next.block_count", 48)
            + _kv_u32("qwen3next.expert_count", 512)
            + _kv_u32("qwen3next.expert_used_count", 10))
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                + struct.pack("<Q", 4) + body)
    info = read_info(str(path))
    assert info.architecture == "qwen3next"
    assert info.layers == 48
    assert info.is_moe
    assert info.experts_used == 10


def test_rejects_non_gguf(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"XXXX" + b"\0" * 32)
    with pytest.raises(ValueError, match="not a GGUF"):
        read_info(str(path))
