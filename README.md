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
Measured host→device bandwidth **9.3 GiB/s** — worth noting the spec sheet
implies roughly double that, which is why the profiler measures rather than
assumes.

Both runs use the **same 15.6 GiB working set** and the same weight cache:

| | Dense (40 × 400 MiB) | MoE (48L × 128E, top-8) |
|---|---|---|
| Throughput (steady state) | 0.43 tok/s | **19.93 tok/s** |
| Ceiling, no cache | 0.60 tok/s | 8.83 tok/s |
| Ceiling, given cache | 0.72 tok/s | 26.71 tok/s |
| Bandwidth saved by cache | 16.2% | 67.0% |
| Bus utilization | 59.7% | 74.6% |

**46x apart at identical size.** The cause is structural, not
implementational: the dense model touches all 15.6 GiB per token while the MoE
touches roughly 1 GiB. A cache also pays far better on MoE, because routing is
skewed and there is a hot subset worth holding — a dense model touches every
layer equally often, so there is no head of the distribution to cache. Note
the MoE run beats its own no-cache ceiling, which is the cache doing its job.

Steady state is measured after a warmup pass, since a real session fills its
caches once. `bench` reports cold start separately rather than hiding it.

### Where the speed came from

Starting point was 6.74 tok/s on the MoE shape. Three changes, each measured:

| Change | Effect |
|---|---|
| Drop per-chunk `cudaStreamSynchronize` for CUDA events | transfer path 4.31 → 5.98 GiB/s |
| Pinned host tier (DMA-ready weights, no CPU copy) | 18.04 → 20.83 tok/s, ±0.08 over 4 runs |
| Allocator slack in the VRAM budget | removed OOM-recovery stalls |

The first was worth the most. Synchronising after every chunk drains the copy
pipeline and idles the engine while the CPU catches up; with the sync removed,
ordering is established GPU-side via `wait_stream` and costs nothing.

Tuning workers, prefetch depth, and prediction width changed nothing — all
held bandwidth at ~5.9 GiB/s, which is what identified the transfer path
rather than the scheduler as the constraint.

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
corestream bench --layers 40 --chunk-mib 400 --tokens 4 --pinned-gib 8
corestream bench --moe --layers 48 --experts 128 --top-k 8 \
                 --chunk-mib 2.5 --tokens 30 --pinned-gib 8
```

`--pinned-gib` is the single biggest tuning knob. Set it as high as free RAM
comfortably allows; it is off by default because pinned memory cannot be
swapped and the safe ceiling depends on what else the machine is doing.

## How it works

| Stage | Technique | Borrowed from |
|---|---|---|
| disk → RAM | `mmap` + kernel readahead | OS demand paging |
| RAM → pinned RAM | one slab, carved into slots, filled once | — |
| pinned RAM → VRAM | direct DMA, event-guarded, no CPU copy | — |
| staying ahead | multiple transfers in flight | TCP sliding window |
| what to keep | LFU-weighted admission | CDN cache admission |
| what to drop | LRU (MoE) / static pinning (dense) | OS page replacement |
| what is next | router-history expert prediction | cellular handover pre-staging |

### Why a pinned host tier

A transfer out of pageable host memory cannot be a straight DMA — the pages
may move, so the driver first copies them into page-locked memory. That copy
is CPU work on the critical path. Holding weights permanently in pinned memory
removes it: the copy engine reads host RAM directly. The tradeoff is that
pinned pages cannot be swapped, so the budget must stay inside physical RAM
(`--pinned-gib`, default off).

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

## Which speedups actually help, measured

Techniques that work well on a datacenter GPU can lose on a 6 GB card,
because there they are limited by compute and here by bandwidth and by VRAM
that everything competes for. These were measured on Qwen2.5-3B-Instruct
(bf16, 5.2 GiB) rather than assumed.

### Where a token's time goes

| | ms/token | share |
|---|---|---|
| Weight transfer | 248 | 82% |
| Everything else | 54 | 18% |

Measuring plain transformers with the model fully resident gives 127 tok/s on
a 0.5B — 0.33 ms per layer. A 36-layer model therefore owes about 12 ms per
token to framework and compute. Everything else is moving weights.

**Consequence: optimising Python, hooks, or kernel launches is close to
worthless here.** The parameter swapping in the hooks costs 23 ms/token
against 248 ms of transfer. Only two things matter — moving fewer bytes, and
moving them closer to peak bandwidth.

Chasing the second of those found a real bug. `_transfer` materialised the
host bytes before checking whether the chunk was already in pinned memory,
and for a real checkpoint that means packing a fresh 145 MiB buffer per layer
only to discard it — roughly 5 GiB of pointless CPU memcpy per token.

| | before | after |
|---|---|---|
| time/token | 457.7 ms | **302.1 ms** |
| transfer share of token | 55% | 82% |
| Qwen2.5-3B throughput | 2.21 tok/s | **3.53 tok/s** |

At 82% transfer-bound the engine is near its floor: the ceiling at these
bytes is ~4.0 tok/s. Everything left has to come from moving fewer bytes.

### Speculative decoding loses on this hardware

Standard advice for bandwidth-bound decode is speculative decoding: a small
draft model proposes several tokens, the large model verifies them in one
pass. It genuinely works — 1.38x over its own baseline.

But the draft model has to be resident, and resident VRAM is weight cache
taken away from the target:

| config | tok/s | cache savings |
|---|---|---|
| 3 GiB weight cache, no draft | **2.21** | 55.0% |
| 1.5 GiB cache + 0.5B draft, speculative | 1.91 | 27.4% |

**Net 14% slower.** The 1.5 GiB spent on a draft model bought less than the
same 1.5 GiB spent caching target weights. This inverts on a card with VRAM to
spare, which is exactly why it is worth measuring rather than importing the
usual advice.

The outputs also diverged slightly ("a strategic location for the country" vs
"for controlling the country"). Speculative decoding is lossless in exact
arithmetic, but verifying several tokens in one batched pass rounds
differently from decoding them one at a time in bf16, and that is enough to
flip a token.

### Prompt-lookup decoding is free but situational

Drafting from the context itself costs no VRAM, so it does not compete with
the cache. It gave 1.27x on a prompt whose answer echoed the input and 1.02x
on an open-ended one. Worth enabling when output quotes input — summarising,
extraction, code edits — and harmless otherwise (`--prompt-lookup 10`).

### What actually wins

**Quantization**, because it is the only option that helps twice: it cuts
bytes moved per token *and* frees VRAM instead of consuming it. A 4-bit 3B is
about 1.7 GiB, which fits entirely in cache on this card — at which point
there is nothing left to stream.

That leads to the honest rule: **quantize until it fits, and stream only what
still does not.** Streaming earns its keep on models that remain too large
after quantization — a 30B at 4-bit is roughly 17 GiB, still far beyond 6 GB
of VRAM but comfortably inside 30 GB of RAM.

**Quantized checkpoints do not work yet, and the attempt is instructive.**

The manifest side is fine: a 4-bit `w4a16` Qwen2.5-7B reads correctly as
3.14 GiB of streamed weights (int32 packed plus bf16 scales) against 15 GiB
for the bf16 original. Two real bugs were fixed getting that far — the model
dtype must ignore integer tensors, or an AWQ checkpoint reads as an int32
model and cannot be instantiated at all; and released parameters need a
placeholder *of their own dtype*, since a quantized layer mixes int32 weights
with floating-point scales.

The blocker is architectural. transformers applies compressed-tensors with
`run_compressed=False` and registers a hook that decompresses the whole model
on the first forward pass. That needs every weight resident simultaneously —
exactly what streaming avoids — and would expand the checkpoint back to full
precision anyway. Forcing `run_compressed=True` produced a module layout that
did not match the checkpoint, at which point **the model generated `!!!!!!!!`
at 20 tok/s while every test passed**.

That failure mode drove the design decision worth keeping: `StreamingModel`
now refuses to start if any parameter has no source of weights. Nothing raises
on its own in that situation — the layer computes on an empty tensor and
generation looks fluent. Checking length is not sufficient either, because a
meta tensor reports its full nominal `numel`, so an entirely absent weight
looks well-formed.

Supporting quantized checkpoints properly means decompressing per layer inside
the forward hooks, which is the natural place for it. AWQ and GPTQ are a
separate blocker: both need `gptqmodel`, which does not build on Python 3.14
(its `pypcre` dependency fails to compile). It installs normally on 3.12.

## What this is not

**Not KV cache compression.** TurboQuant and similar work compress the KV
cache, which buys context length and concurrency. That is a different
bottleneck from this one. CoreStream is limited by *weight* bytes per token;
on a 6 GB card the KV reserve is under 2 GB, so compressing it 4x frees around
1.4 GB for more weight cache — worth roughly 20%, not the main lever. The two
are complementary rather than alternatives.

## Remaining headroom

At 19.93 tok/s against a 26.71 tok/s ceiling, about 25% is still on the table:

- **Bus utilization 74.6%** — more pinned coverage closes part of this.
- **Prefetch coverage 62%** — MoE routing is only partly predictable, so some
  stalls are inherent. Widening the prediction was measured and made things
  worse: it raised coverage 57%→59% while cutting savings 45%→38%.
- **Raising the ceiling itself** needs fewer bytes per token, not better
  scheduling. Multi-token verification (speculative decoding) is the standard
  answer for bandwidth-bound decode, though it pays less on MoE than on dense
  models because different tokens route to different experts.

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
