# Running large models on a 6 GB / 30 GB laptop

## Result

**Qwen3-Next-80B-A3B at 11.7 tok/s** on an RTX 3060 Laptop (6 GB) with 30 GB
of RAM — the same throughput Ollama gets on an 8B model on this machine, from
a model ten times larger.

| | model | sustained |
|---|---|---|
| **this configuration** | **Qwen3-Next-80B-A3B** (80B, ~3B active) | **11.7 tok/s** |
| Ollama, tuned | `ministral-3:8b` (8B dense) | 11.74 tok/s |
| Ollama, tuned | `qwen2.5-coder:32b` (32B dense) | 2.40 tok/s |

Reproduce with `./run-80b.sh`. Quality holds at this quantization: it answers
the "17 sheep, all but 9 run away" trick correctly, gets Canberra-not-Sydney
with the right reasoning, and writes correct iterative Fibonacci.

Measured placements, 48-token decode:

| configuration | tok/s |
|---|---|
| `-ngl 999 -ncmoe 40 -t 14` | **12.55** |
| `-ngl 999 --cpu-moe -t 14` | 12.32 |
| `-ngl 0 -t 14` (all CPU) | 12.29 |
| `-ngl 999 --cpu-moe -t 6` | 12.21 |
| `-ngl 999 --cpu-moe -t 20` | 9.18 |

Two things worth noting. GPU offload of attention barely moves the number
(12.32 vs 12.29 for pure CPU) because the run is bound by RAM bandwidth on the
expert path, not by the GPU. And **more threads made it slower**: `-t 20`
drops to 9.18 because the i9-12900H's eight slow E-cores stall the six fast
P-cores on memory-bound work. `-t 14` is the sweet spot.


Everything here is measured on this machine — RTX 3060 Laptop (6 GB), i9-12900H
(14 cores / 20 threads), 30 GB DDR5, NVMe — not taken from spec sheets, which
overstate several of these by 2x.

## The tier map

| tier | bandwidth | capacity | how measured |
|---|---|---|---|
| VRAM | 292 GiB/s | 5.5 GiB free | device-to-device copy |
| RAM (read) | 36.0 GiB/s | ~25 GiB usable | large tensor reduction |
| NVMe, 4 MiB random | 2.90 GiB/s | 219 GiB free | cold `pread` after `FADV_DONTNEED` |
| NVMe, 64 KiB random | 1.06 GiB/s | | same |
| PCIe host→device | 9.2 GiB/s | | pinned `cudaMemcpyAsync` |
| GPU compute | 20.4 TFLOPS bf16 | | 4096³ matmul |
| CPU compute | 0.43 TFLOPS fp32 | | 2048³ matmul, 14 threads |

## The one equation that predicts throughput

Decoding one token is **memory-bound, not compute-bound**: every weight the
token uses must be read, and almost nothing is done with it before moving on.
So:

```
tokens/sec  ≈  bandwidth of the tier holding the weights
               ÷ bytes that tier must supply per token
```

Checked against a real run — Qwen2.5-Coder-32B, Q4_K_M, 18.5 GiB, which
llama.cpp placed 78% CPU / 22% GPU:

```
CPU side:  0.78 × 18.5 GiB ÷ 36.0 GiB/s = 400 ms
GPU side:  0.22 × 18.5 GiB ÷ 292 GiB/s  =  14 ms
predicted: 2.4 tok/s        measured: 2.40 tok/s
```

llama.cpp is already running at the memory-bandwidth limit. There is no
software slack left to recover on that path.

## Why streaming weights to the GPU is the wrong design

RAM reads at **36 GiB/s**. PCIe moves at **9.2 GiB/s**. Shipping a weight to
the GPU to compute on it is therefore ~4x slower than computing where it
already sits, and that gap cannot be closed by better scheduling.

This is the mistake the `corestream/` engine in this repository makes, and why
it loses to llama.cpp by 4.5x. The correct rule is: **never move weights per
token — move the computation to the weights.**

A second, independent gap: PyTorch's CPU path manages 0.2 GiB/s on a bf16
GEMV against 36 GiB/s of available bandwidth. llama.cpp's hand-written
AVX-512 quantized kernels run near memory speed. Roughly 100x, and not
recoverable in PyTorch.

## Why a dense 70B cannot be fast here, at any quantization

A dense model touches every parameter for every token.

| model | size @ 4-bit | where it lives | ceiling |
|---|---|---|---|
| 70B dense | 35 GiB | mostly NVMe | ~0.1 tok/s |
| 70B dense @ 2-bit | 17.5 GiB | RAM | ~0.5 tok/s |
| 32B dense | 18.5 GiB | RAM + some GPU | 2.4 tok/s (measured) |

Nothing in the software stack changes this. 35 GiB must cross a 2.9 GiB/s
link every token.

## Why a large MoE can be fast

A mixture-of-experts model routes each token through a small fraction of its
parameters. Total size sets what must be *stored*; active parameters set what
must be *read per token* — and only the second one costs time.

| model | total @ 4-bit | active/token | ceiling if resident |
|---|---|---|---|
| Mixtral-8x7B (13B active) | 23.5 GiB | 6.5 GiB | ~1.4 tok/s |
| GLM-4.5-Air (12B active) | 53 GiB | 6.0 GiB | ~0.4 tok/s |
| Qwen3-30B-A3B (3B active) | 15 GiB | 1.5 GiB | ~6 tok/s |
| **Qwen3-Next-80B-A3B (3B active)** | **47 GiB** | **1.5 GiB** | **see below** |

The 80B is the shape that satisfies "70B or larger" *and* "fast": it stores
like an 80B and reads like a 3B.

To reach 10 tok/s the engine may move at most 0.92 GiB per token, which is
about 1.8B active parameters at 4-bit. Active-parameter count, not total
size, is the number to shop for.

## Configuration that follows from the above

For an MoE larger than RAM, the placement rule is not "N layers on the GPU".
It is **by tensor role**:

- **Attention, norms, embeddings → GPU.** Small, and touched on every token
  regardless of routing.
- **Expert banks → CPU/RAM.** Large, and only a few are read per token.

`llama-server` supports exactly this:

```bash
llama-server -m model.gguf -ngl 999 --cpu-moe -fa on -c 4096 -t 14
```

`--cpu-moe` keeps every expert tensor on the CPU while `-ngl 999` puts
everything else on the GPU. `-ncmoe N` offloads experts for only the first N
layers, which is the knob to tune when some experts do fit in VRAM.

For a model larger than RAM, leave mmap on: the kernel page cache then holds
the most-used experts automatically, and expert usage is skewed enough that
the hit rate is far better than the naive size ratio suggests.

## Choosing the quantization: fit RAM, then stop

Quantization is usually discussed as a quality/size tradeoff. On a machine
where the model is near the RAM boundary it is really a **tier** decision, and
the tiers are 12x apart: RAM reads at 36 GiB/s, NVMe at 2.9 GiB/s. Crossing
that boundary matters far more than the bits per weight.

Qwen3-Next-80B-A3B, ~25 GiB of usable RAM, with attention on the GPU:

| quant | file size | experts in RAM? | expected |
|---|---|---|---|
| Q4_K_M | 45–47 GiB | no — ~22 GiB from NVMe | ~7 tok/s |
| UD-Q3_K_XL | 33 GiB | mostly not | slow |
| **UD-Q2_K_XL** | **28.1 GiB** | yes, once ~5 GiB of attention is on the GPU | **target** |
| UD-IQ2_XXS | 24.4 GiB | yes outright | fastest, lowest quality |

The rule that follows: **quantize until the working set fits RAM, then stop.**
Going smaller than that buys nothing — the bottleneck has already moved
elsewhere — and costs quality for free.

Unsloth's `UD-` ("dynamic") quants matter here: they keep attention and other
sensitive tensors at higher precision while compressing the expert banks
hardest, which is exactly the right bias when the experts are what has to fit.

## What to shop for

1. **Active parameters** decide speed. Total size decides whether it fits.
2. **Quantize until the working set fits RAM**, then stop — going below what
   fits buys nothing and costs quality.
3. Prefer larger random reads: NVMe gives 2.90 GiB/s at 4 MiB but 1.06 GiB/s
   at 64 KiB, so tensor layout and readahead matter for disk-backed models.
