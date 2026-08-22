# CoreStream: Unified Tiered-Memory Inference Engine — Design Spec

Date: 2026-08-22
Status: v2 (revised after roofline analysis on target hardware)

## Purpose

Run LLMs that do not fit in a consumer laptop's GPU by streaming weights
through a VRAM → RAM → SSD hierarchy, borrowing techniques that OS virtual
memory, TCP, and CDN caching already use to hide a slow tier behind a fast one.

Target hardware (measured, this machine):

| Resource | Value |
|---|---|
| GPU | RTX 3060 Laptop, 6 GB VRAM |
| PCIe | 4.0 x16 (~20 GB/s realistic pinned host→device) |
| RAM | 30 GB total, ~25 GB available |
| Disk | NVMe on ext4, 240 GB free (~3.5 GB/s sequential) |
| OS | Native Linux (no WSL translation-layer penalty) |

## The roofline that drives every design decision

Streaming inference is bandwidth-bound, not compute-bound. Per generated
token, the engine must move every weight the token actually uses across the
slowest tier boundary in its path. That gives a hard ceiling:

```
max_tokens_per_sec  =  effective_bandwidth / bytes_touched_per_token
```

Measured against this machine:

| Path | Bytes/token | Ceiling |
|---|---|---|
| Dense 27B @ 4-bit, streamed from NVMe | ~16 GB | ~0.2 tok/s |
| Dense 27B @ 4-bit, resident in RAM, streamed over PCIe | ~16 GB | ~1.2 tok/s |
| MoE 30B-A3B @ 4-bit (~3B active), resident in RAM | ~1.7 GB | ~12 tok/s |
| MoE 30B-A3B with hot experts cached in VRAM | < 1.7 GB | > 12 tok/s |

Three conclusions follow, and they reshaped this spec from its v1 draft:

1. **Compute/IO overlap is a secondary win, not the headline.** A layer's
   load costs ~30x its compute on this hardware, so perfectly hiding compute
   behind IO reclaims a few percent. Worth doing, insufficient alone.
2. **Reducing bytes-touched-per-token is the primary win.** That means
   keeping weights in RAM rather than on disk (3x), and caching the
   most-used weights in VRAM so they are not re-sent over PCIe every token.
3. **MoE is the case worth optimizing.** Dense streaming is capped near
   1 tok/s by arithmetic no engine can escape. MoE touches a fraction of its
   parameters per token, so it clears 10 tok/s on the same box. A ~10x gap
   that is structural, not an implementation detail.

## Core structural insight

**A dense model is an MoE in which every expert is always active.**

So CoreStream does not have a "dense path" and an "MoE path". It has one
path that asks, per step, *which chunks does this token actually need?* — and
a dense model simply answers "all of them". This makes the unification
structural rather than a router bolted over two engines, and means the dense
capability falls out of the MoE work for free.

## Architecture

Python package `corestream/`, with C/CUDA only on the hot path.

- **`hardware.py`** — one-time profiler. Measures free VRAM, free RAM, NVMe
  sequential read, and actual pinned host→device PCIe bandwidth. Reserves
  KV-cache headroom, then reports the byte budget available for weight
  caching. Also computes the roofline above so the CLI can tell the user
  what tok/s is achievable *before* they download 16 GB.
- **`inspector.py`** — reads the model's HF `config.json`, classifies Dense
  vs MoE, extracts layer/expert topology, and produces the chunk manifest.
- **`store.py`** — `TieredWeightStore`. Every layer (dense) or expert (MoE)
  is a *chunk* with a tier: `HOT` (VRAM), `WARM` (host RAM), `COLD` (disk).
  - `COLD→WARM`: delegated to `mmap` + kernel readahead. We deliberately do
    not reimplement a RAM cache; the OS page cache is better tested than
    anything written here, and with 25 GB free RAM a 16 GB model becomes
    fully resident after one pass at zero implementation cost.
  - `WARM→HOT`: our code. Pinned staging buffers + `cudaMemcpyAsync`.
  - Admission is **LFU-weighted**: a chunk earns VRAM residency only after
    repeated hits, so rare experts cannot evict hot ones on a single touch.
  - Eviction is **LRU** among resident chunks.
- **`scheduler.py`** — `PrefetchScheduler`. A worker pool touching COLD
  chunks to trigger readahead, plus a dedicated CUDA copy stream promoting
  WARM→HOT, both overlapping the GPU's current compute. Prefetch depth is
  tunable (default 2).
- **`loaders/`** — `plan(step) -> [chunk_key]`. `DensePlan` returns every
  layer in fixed order. `MoEPlan` returns the attention chunks plus the
  experts the router selected, and additionally emits *predictive* hints for
  experts the recent router history suggests are likely next.
- **`native/`** — pinned-buffer pool and async copy helpers. The only
  hand-written C/CUDA in the project.
- **`cli.py`** — `corestream doctor` (profile + roofline, no download) and
  `corestream run <model>`.

## Prior art this design borrows from

| Technique | Origin | Applied here as |
|---|---|---|
| Demand paging + readahead | OS virtual memory | `mmap`-backed COLD→WARM; kernel does the readahead |
| Working-set model | OS virtual memory | HOT tier holds the chunks in the current execution window |
| Sliding window / pipelining | TCP | Multiple chunk promotions in flight rather than load→wait→compute |
| Cache admission policy | CDN tiered caching | LFU-weighted admission stops one-off chunks thrashing VRAM |
| Handover pre-staging | Cellular networks | Router-history predictive expert prefetch |
| Erlang capacity planning | Telecom trunk sizing | Size the HOT tier from measured hit-rate targets, not raw byte division (stretch goal) |

## Error handling

- **VRAM OOM on promotion** → evict an additional chunk and retry once; on
  repeated failure, execute that chunk on CPU for this step rather than crash.
- **Disk read failure** → one retry, then a hard error naming the failing
  shard. No silent corruption.
- **No CUDA** → the VRAM tier is disabled; everything runs RAM/CPU. Slower
  but functional, and the same code path.
- **KV cache growth** → reserved up front in the budget; if context growth
  would exceed the reserve, the HOT weight budget shrinks rather than OOMing.

## Testing / success criteria

- **Correctness** — outputs match the same model run through plain HF
  `transformers`, same seed and sampling.
- **Efficiency, measured against the roofline** — the benchmark reports
  achieved tok/s *as a percentage of the theoretical ceiling* for the
  measured bytes-per-token. This is the honest metric: "faster than AirLLM"
  can be satisfied by a 5% gain and prove nothing, whereas roofline
  utilization shows how much of the achievable bandwidth we actually captured.
  Target: > 70% of roofline on an MoE model.
- **Cache effectiveness** — report expert hit-rate in the HOT tier. This is
  the number that determines whether bytes/token drops below the naive
  estimate, and thus whether the LFU admission policy is earning its place.
- **Platform** — native Linux verified; CPU-only fallback verified by
  forcing `CUDA_VISIBLE_DEVICES=`.

## Deliberately out of scope for v1

- Training or fine-tuning. Inference only.
- Multi-GPU. This targets single-GPU consumer laptops.
- Custom quantization kernels. We consume already-quantized weights; we do
  not implement new quant formats.
- Erlang-style hit-rate-derived tier sizing — noted above as a stretch goal;
  v1 sizes the HOT tier by byte budget.

## Practical note for other users

On Windows/WSL2, model weights must live on the WSL2 native ext4 filesystem,
not `/mnt/c`. The 9p/DrvFs translation layer cripples the sequential read
throughput this design depends on. Not applicable to this machine (native
Linux) but a documented gotcha for the README.
