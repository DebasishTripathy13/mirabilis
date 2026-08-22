# Running large models on a 6 GB / 30 GB laptop

## Result

**Qwen3-Next-80B-A3B at ~26 tok/s** on an RTX 3060 Laptop (6 GB) with 30 GB
of RAM — the same throughput Ollama gets on an 8B model on this machine, from
a model ten times larger.

| | model | sustained |
|---|---|---|
| **this configuration** | **Qwen3-Next-80B-A3B** (80B, ~3B active) | **26.4 tok/s** |
| Ollama, tuned | `ministral-3:8b` (8B dense) | 11.74 tok/s |
| Ollama, tuned | `qwen2.5-coder:32b` (32B dense) | 2.40 tok/s |

Reproduce with `./run-80b.sh`. Quality holds at this quantization: it answers
the "17 sheep, all but 9 run away" trick correctly, gets Canberra-not-Sydney
with the right reasoning, and writes correct iterative Fibonacci.

### The GPU backend has to be found explicitly

Ollama ships `libggml-cuda.so` in a versioned subdirectory rather than beside
the binary, and ggml does not search subdirectories. Launched directly, the
server prints `no usable GPU found`, ignores `--gpu-layers`, and runs entirely
on the CPU -- while still working, just slowly. Pointing `GGML_BACKEND_PATH`
at the *file* fixes it; pointing it at the directory fails with
`Is a directory`.

This was worth **1.6x**: the same model and flags went from 12.98 to 21.27
tok/s once the backend loaded. Every measurement below the first table was
taken before this was found, and is CPU-only.

Two things also change once the GPU is real. `-ngl 999` stops being harmless:
llama.cpp reports `failed to fit params to free device memory, n_gpu_layers
already set by user to 999, abort` and gives up on its own layer fitter.
And placements that put expert layers on the GPU, which measured fine on CPU,
now run out of VRAM and fail outright.

### Free RAM beats every flag

The largest single gain measured here was not a placement or a thread count.
It was closing a browser.

A 28.1 GiB model against 23.1 GiB of *available* RAM leaves 5 GiB that cannot
stay in the page cache, and that share is re-read from NVMe on every token at
roughly a twelfth of RAM speed. Chrome was holding 5.3 GiB — almost exactly
the shortfall:

| | tok/s | model load |
|---|---|---|
| 23.1 GiB available (browser open) | 23.2 | 28–38 s |
| **28.0 GiB available (browser closed)** | **26.4** | **8 s** |

Nothing about the configuration changed. `lm doctor` now reports the shortfall
directly, because it is invisible otherwise: the engine does not complain, it
simply runs slower and with far more run-to-run variance as the cache thrashes.

The corollary for choosing a model: *available* RAM is the budget, not
installed RAM. On a 30 GiB laptop with a browser open, the real ceiling is
closer to 23.

### Trading context for VRAM does not work here

An obvious-looking move is to shrink the context so the KV cache frees VRAM
for more expert layers. Measured, it fails: `-c 2048` with `-ncmoe 41` or `39`
would not allocate at all. On a hybrid-attention model the KV cache at 4096 is
already only ~0.4 GiB, so halving it frees far less than the ~1 GiB each
additional expert layer needs.

### Thread placement: enough traffic to fill the road, not enough to jam it

Decode is memory-bound, so what matters is how many *independent* load/store
paths are pulling from RAM. That is not the same as how many threads exist,
and on a hybrid CPU it is not the same as how many cores exist either.

An i9-12900H presents 20 logical CPUs: 6 P-cores with two hyperthreads each
(CPUs 0-11) and 8 E-cores (12-19). Hyperthread siblings share the load/store
units, so a second thread on the same core adds contention without adding
bandwidth. E-cores have their own paths, so a couple of them genuinely help --
but they are slow enough that too many stall the fast cores waiting at the
end of each parallel region.

Measured, all with the same placement:

| threads | CPUs | tok/s |
|---|---|---|
| 12, unpinned | scheduler's choice | 22.6 |
| 6 | one per physical P-core | 24.4 |
| **8** | **6 physical P + 2 E** | **26.5** |
| 10 | 6 physical P + 4 E | 23.8 |
| 14 | 6 physical P + all 8 E | 21.6 |

Pinning matters as much as the count: without `-C <mask> --cpu-strict 1` the
scheduler is free to migrate a thread onto an E-core mid-run, and 12 unpinned
threads measured slower than 8 pinned ones. `lm` derives the mask from
`/sys` -- peak clock separates fast from slow cores, and
`thread_siblings_list` collapses hyperthreads to one entry each.

### Dividing the work between CPU and GPU

With the backend loaded, the split actually matters. VRAM reads eight times
faster than RAM, so every byte moved onto the card is worth it -- until the
card runs out, at which point the load fails outright rather than degrading.
The useful range turned out to be a narrow band just below "all experts in
RAM", which a coarse sweep missed entirely:

| configuration | tok/s |
|---|---|
| all experts in RAM (`-ncmoe 48`) | 21.30 |
| 2 expert layers on GPU (`-ncmoe 46`) | 22.27 |
| 4 expert layers on GPU (`-ncmoe 44`) | 22.99 |
| **5 expert layers on GPU + KV `q8_0`** | **23.10** |
| 6 expert layers on GPU (`-ncmoe 42`) | fails to allocate |

Attention and norms sit on the GPU throughout; what varies is how many
layers' expert banks join them. Quantizing the KV cache to `q8_0` is worth
one extra layer -- not for its own sake, but because the VRAM it frees is
immediately spent on weights.

**More threads made it slower**: `-t 20` measured 9.18 against 12.3 for
`-t 14` on the CPU-only path, because the i9-12900H's eight slow E-cores
stall its six fast P-cores on memory-bound work.

The progression on this model, each step measured:

| | tok/s |
|---|---|
| before the CUDA backend was found (CPU only) | 12.98 |
| GPU backend loaded, all experts in RAM | 21.47 |
| optimal split + KV quantization | **23.10** |


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

## Two approaches that do not help here, and why

Both `AirLLM` and `colibrì` solve a real problem, and it is not this one.
They make models *fit* that otherwise could not run at all. Fitting and speed
are different goals, and on this hardware they pull in opposite directions.

**AirLLM** keeps one layer on the GPU and streams the rest, reporting
Qwen3.8-27B in 3.33 GB of VRAM. That is a footprint number. The engine in
`corestream/` in this repository is the same architecture, built and measured
here: it loses about 4.5x to llama.cpp, because PCIe moves 9.2 GiB/s while RAM
reads at 36. Shipping a weight to the GPU to compute on it is four times
slower than computing where it already sits.

**colibrì** streams MoE experts from disk for models far larger than RAM —
744B to 2.8T. Its own published figures are 0.05–0.1 tok/s on a 25 GB box and
1.07 tok/s on a laptop-class RTX 5070 Ti. The configuration in this document
reaches 18–23 tok/s on an 80B. colibrì answers "how do I run a model ten times
larger than my machine at all", which is a different question.

### The one idea worth testing, tested

colibrì pins *individual experts* by measured routing heat, while `-ncmoe N`
bluntly takes the first N layers. That looked like a real gap, so it was
measured with `-ot`, which can place tensors by regex:

| experts placed on GPU | tok/s |
|---|---|
| last 5 layers (`-ncmoe 43`) | 22.70 |
| first 5 layers (`-ot`) | 22.13 |
| middle 5 layers (`-ot`) | 21.96 |

Layer choice is within noise. The reason is structural: a GGUF packs all 512
experts of a layer into one tensor (`blk.N.ffn_gate_exps.weight`), so the
finest placement llama.cpp can express is a whole layer. Per-expert pinning is
not representable in this format, whatever its merits in an engine that owns
its own container.

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
