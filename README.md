# CoreStream

Tiered-memory streaming inference for consumer GPUs. Runs models that do not
fit in VRAM by streaming weights through a VRAM → RAM → SSD hierarchy, using
techniques borrowed from OS virtual memory, TCP, and CDN caching.

Combines the two ideas it grew out of — per-layer disk streaming (AirLLM) and
MoE expert streaming (colibrì) — into one engine, on the observation that
**a dense model is just an MoE in which every expert is always active**.

## The number that decides everything

Streaming inference is bandwidth-bound, not compute-bound. Per token, every
weight the token touches must cross the slowest tier boundary in its path:

```
max_tokens_per_sec = effective_bandwidth / bytes_touched_per_token
```

No amount of engine quality escapes that division. It is why this project
optimises for reducing bytes rather than for going faster, and why MoE models
and dense models land in completely different regimes on identical hardware.

## Measured results

Hardware: RTX 3060 Laptop (6 GB VRAM), 30 GB RAM, NVMe, PCIe 4.0 x16.
Measured host→device bandwidth **9.1 GiB/s** — worth noting the spec sheet
implies roughly double that, which is why the profiler measures rather than
assumes.

Both runs use the **same 15.6 GiB working set** and the same 3.6 GiB cache:

| | Dense (40 × 400 MiB) | MoE (48L × 128E, top-8) |
|---|---|---|
| Throughput | 0.38 tok/s | **6.74 tok/s** |
| Roofline ceiling | 0.60 tok/s | 8.93 tok/s |
| Roofline utilization | 63.1% | 75.5% |
| Bandwidth saved by cache | 14.2% | 44.9% |

**17.7x apart at identical size.** The cause is structural, not
implementational: the dense model touches all 15.6 GiB per token while the MoE
touches roughly 1 GiB. A cache also pays far better on MoE, because routing is
skewed and there is a hot subset worth holding — a dense model touches every
layer equally often, so there is no head of the distribution to cache.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

## Use

Profile the machine and project a model's throughput **before downloading it**
— this reads `config.json` only, a few kilobytes:

```bash
corestream doctor
corestream doctor Qwen/Qwen3-30B-A3B
```

Benchmark the streaming core on a given model shape:

```bash
corestream bench --layers 40 --chunk-mib 400 --tokens 3
corestream bench --moe --layers 48 --experts 128 --top-k 8 --chunk-mib 2.5
```

## How it works

| Stage | Technique | Borrowed from |
|---|---|---|
| disk → RAM | `mmap` + kernel readahead | OS demand paging |
| RAM → VRAM | pinned buffers, async copy on a dedicated stream | — |
| staying ahead | multiple transfers in flight | TCP sliding window |
| what to keep | LFU-weighted admission | CDN cache admission |
| what to drop | LRU (MoE) / static pinning (dense) | OS page replacement |
| what is next | router-history expert prediction | cellular handover pre-staging |

### Why eviction policy depends on model shape

A dense model revisits every layer once per token in fixed order. That cyclic
pattern is the **worst case for LRU**: each layer is evicted by the ones
following it before it comes round again, so the hit rate is zero regardless
of cache size. Refusing to evict — pinning a fixed subset — converts that into
savings proportional to cache-over-working-set. Measured on a 16-layer cycle
with a 4-layer cache: LRU 0.0%, static pinning 17.2%.

An MoE router concentrates traffic on a minority of experts, so recency does
predict reuse and LRU is correct there. The engine picks the policy from the
plan rather than leaving it to the caller.

## Status

The streaming core, policies, profiler, and benchmark harness are implemented
and tested (46 tests). Weights are exercised through `SyntheticSource` at
realistic sizes and access patterns.

Not yet done: wiring the store into a real Hugging Face forward pass, so this
does not yet generate text from a downloaded checkpoint. `SafetensorsSource`
and the chunk-mapping helpers exist for that next step.

## Notes

On Windows, keep weights on the WSL2 native ext4 filesystem, not `/mnt/c` —
the translation layer cripples the sequential read throughput this design
depends on.
