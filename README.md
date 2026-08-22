# lm — run an 80B model on a 6 GB laptop

**Qwen3-Next-80B at ~26 tok/s** on an RTX 3060 Laptop (6 GB VRAM, 30 GB RAM) —
the same throughput Ollama gets on an *8B* model on that machine, from one ten
times larger.

```bash
pip install -e .
lm pull unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF   # picks the quant that fits
lm tune qwen3-next                                  # measures the fastest setup
lm run  qwen3-next                                  # chat
```

| | model | sustained |
|---|---|---|
| **this configuration** | Qwen3-Next-80B-A3B (80B total, ~3B active) | **26.4 tok/s** |
| Ollama, tuned | `ministral-3:8b` (8B dense) | 11.7 tok/s |
| Ollama, tuned | `qwen2.5-coder:32b` (32B dense) | 2.4 tok/s |

Everything below was measured on one machine. The numbers will differ on
yours; `lm tune` exists so the *configuration* does not have to be copied
along with them.

---

## The problem, stated as arithmetic

Generating one token means reading every weight that token uses, and doing
very little arithmetic with each byte before moving on. That makes decoding
**memory-bound, not compute-bound**, and it collapses to one equation:

```
tokens/sec  ≈  bandwidth of the tier holding the weights
               ÷  bytes that tier must supply per token
```

The tiers on this laptop, measured rather than taken from spec sheets — which
overstate several of these by 2x:

| tier | bandwidth | capacity |
|---|---|---|
| VRAM | 292 GiB/s | 5.6 GiB |
| RAM | 36 GiB/s | ~28 GiB |
| PCIe (host → GPU) | 9.2 GiB/s | — |
| NVMe, 4 MiB random | 2.9 GiB/s | 200+ GiB |
| NVMe, 64 KiB random | 1.06 GiB/s | — |

Check the equation against a real run. Qwen2.5-Coder-32B at Q4_K_M is 18.5 GiB,
which llama.cpp placed 78% on CPU and 22% on GPU:

```
CPU side:  0.78 × 18.5 GiB ÷ 36.0 GiB/s = 400 ms
GPU side:  0.22 × 18.5 GiB ÷ 292 GiB/s  =  14 ms
predicted: 2.4 tok/s          measured: 2.40 tok/s
```

The prediction is exact, which is what makes the rest of this tractable: you
can work out what a model will do before downloading it.

## The insight that makes an 80B viable

A dense model reads **every** parameter for **every** token. That is why the
32B above manages 2.4 tok/s and why a dense 70B cannot be fast here at any
quantization — 35 GiB has to cross a memory bus every single token, and no
software fixes that.

A mixture-of-experts model routes each token through a small fraction of
itself. Qwen3-Next-80B has 512 experts per layer and uses 10 of them per token:
80B parameters stored, about 3B read.

> Think of a library. A dense model is a reader who walks every aisle before
> answering any question. An MoE reader knows the catalogue and visits three
> shelves. The building is the same size; the walk is not.

**Total size decides what must be stored. Active parameters decide what must
be read per token — and only the second one costs time.**

| | total @ 4-bit | read per token | ceiling here |
|---|---|---|---|
| Llama-70B dense | 35 GiB | 35 GiB | ~0.1 tok/s |
| Qwen2.5-32B dense | 18.5 GiB | 18.5 GiB | 2.4 tok/s *(measured)* |
| **Qwen3-Next-80B-A3B** | 28 GiB @ Q2_K_XL | **~1 GiB** | **26.4 tok/s** *(measured)* |

So the thing to shop for is a **low active-parameter count**, not a small total
size. That single choice is worth more than every tuning flag combined.

---

## How the speed was found

Starting point: 12.98 tok/s. Each step below is a separate measurement on the
same model and prompt.

### 1. The GPU was never being used — 1.65x

The server had been printing this the whole time, and carrying on:

```
warning: no usable GPU found, --gpu-layers option will be ignored
```

Ollama ships `libggml-cuda.so` in a versioned subdirectory (`cuda_v13/`) that
ggml does not search. `LD_LIBRARY_PATH` is not enough — the backend is found
through `GGML_BACKEND_PATH`, which wants the **file**, not the directory
(pointing it at the directory fails with `Is a directory`).

**12.98 → 21.5 tok/s.** Every measurement taken before finding this was
CPU-only, including tuning runs that had looked sensible.

> The failure mode worth remembering: it did not crash, it did not warn
> loudly, it just ran at a third of the speed. `lm doctor` now reports which
> backend it found and says so when a GPU exists but no backend does.

### 2. Free RAM beats every flag — 1.14x

The 28.1 GiB model against 23.1 GiB of *available* RAM leaves 5 GiB that
cannot stay in the page cache, and that share is re-read from NVMe **on every
token** at a twelfth of RAM speed. Chrome was holding 5.3 GiB — almost exactly
the shortfall.

| | tok/s | model load |
|---|---|---|
| 23.1 GiB available (browser open) | 23.2 | 28–38 s |
| **28.0 GiB available (browser closed)** | **26.4** | **8 s** |

Nothing about the configuration changed. It also explains why results had been
swinging 15–20% between runs: the cache was thrashing.

**Available RAM is the budget, not installed RAM.** `lm doctor` now reports
the shortfall in gigabytes, because it is otherwise invisible.

### 3. Thread placement: fill the road, don't jam it — 1.15x

Decode is memory-bound, so what matters is how many **independent load/store
paths** are pulling from RAM. That is not the number of threads, and on a
hybrid CPU it is not the number of cores either.

An i9-12900H shows 20 logical CPUs: 6 P-cores with two hyperthreads each
(0–11) and 8 E-cores (12–19). Hyperthread siblings share the load/store units
decode is waiting on.

> Two vehicles in one lane do not move more freight. A couple of extra lanes
> do — until there are so many that everyone waits at the merge.

| threads | CPUs | tok/s |
|---|---|---|
| 12, unpinned | scheduler's choice | 22.6 |
| 6 | one per physical P-core | 24.4 |
| **8** | **6 physical P + 2 E** | **26.5** |
| 10 | 6 physical P + 4 E | 23.8 |
| 14 | 6 physical P + all 8 E | 21.6 |

Pinning matters as much as the count: unpinned, the scheduler migrates threads
onto E-cores mid-run, and 12 unpinned measured *slower* than 8 pinned.

`lm` derives the mask from `/sys` — peak clock separates fast cores from slow,
`thread_siblings_list` collapses hyperthreads to one entry each. The dense 27B
in this repo prefers **10** threads where the MoE prefers 8, which is exactly
why the tool measures rather than hard-coding a rule.

### 4. Placement and KV cache, searched together — 1.08x

Attention and norms are small and read every token, so they belong on the GPU.
Expert banks are large and mostly idle per token, so they belong in RAM. What
varies is how many layers' experts join attention on the card — and the useful
range is a narrow band, with a cliff at the end:

| | tok/s |
|---|---|
| all experts in RAM (`-ncmoe 48`) | 21.3 |
| 4 expert layers on GPU | 23.0 |
| **5 expert layers on GPU + KV cache `q8_0`** | **23.2** |
| 6 expert layers on GPU | **fails to allocate** |

Quantizing the KV cache is worth exactly one more layer — not for its own sake,
but because the VRAM it frees gets spent on weights. The two therefore have to
be searched *together*; tuning them separately misses it.

An earlier coarse sweep (85%/75%/65% of layers) found nothing but failures,
because the entire interesting region sits within a few layers of the top.

### 5. Concurrent streams — 1.67x aggregate

The GPU sits at 2–22% utilisation during decode. It cannot help a single
token, because layer N+1 depends on layer N. It *can* work on different tokens:

| concurrent streams | aggregate |
|---|---|
| 1 | 14.6 tok/s |
| **4** | **24.4 tok/s** |

> Carpooling. The trip costs the same whether the car carries one passenger or
> four; four tokens share each weight read.

Per-stream latency drops to ~6 tok/s, so this is the right trade for agents or
batch work and the wrong one for a person waiting on a reply. Enable with
`-np 4`.

---

## What was tested and gave nothing

Negative results, recorded because each one looked promising and cost time:

| tried | result | why |
|---|---|---|
| **MTP speculative decoding** | 2.70 vs 3.36 — *slower* | The head is present and published figures say 78–90%. But only ~4 of 11.4 GiB fits on the card, so most layers run on CPU, and verifying several drafted tokens at once is *compute*-bound — it trades a bottleneck this machine can afford for one it cannot. |
| **n-gram speculation** | 11.9 vs 12.0 on MoE; +5% on dense | Verifying N drafted tokens on an MoE reads the experts for N different routing decisions, so it does not amortise the way it does on dense. |
| **Huge pages** | 26.55 vs 26.31 — noise | `AnonHugePages` stayed at 2 MB. mmap'd weights are file-backed and file-backed THP is off; `--no-mmap` allocations are not in a form THP promotes. |
| **`--no-mmap`** | 26.20 vs 26.31 | No benefit, and a much slower load. |
| **Expert layer choice via `-ot`** | 22.7 / 22.1 / 22.0 for last / first / middle | Within noise. A GGUF packs all 512 experts of a layer into one tensor, so the finest placement expressible is a whole layer. |
| **Trading context for VRAM** | failed to allocate | On a hybrid-attention model the KV cache at 4096 is already only ~0.4 GiB, so halving it frees far less than the ~1 GiB another expert layer needs. |

## Why the GPU cannot take more of the load

It looks like waste — 2–22% utilisation, 23–33 W, clocks never leaving idle.
Three reasons, in order of how much they bind:

**The model is 97% routed experts.** By tensor kind, the file is
42.8 GiB-equivalent of `ffn_*_exps` against 1.3 GiB of everything else. The
standard advice to "put all always-active tensors on the GPU" is already
satisfied and costs under a gigabyte. What remains is expert weight, and about
17% of it fits.

**A layer's experts are one tensor.** llama.cpp places whole tensors, so a
layer is either GPU or CPU, never both — and whichever side is not working
idles.

**That split is where the missing throughput is.** The ten experts a token
routes to are independent of each other; nothing but the file layout stops five
running on the GPU while five run on the CPU. If they could, the two bandwidths
would add — 36 GiB/s plus 9.2 GiB/s, roughly 25% more. Splitting experts within
a layer is an open, unimplemented request
([llama.cpp#20528](https://github.com/ggml-org/llama.cpp/discussions/20528)),
and collecting it needs an engine that owns its own weight container.

> The GPU is a fast tunnel that is too narrow for the traffic, and the freight
> cannot be split between the tunnel and the surface road because it is bolted
> to one pallet.

### What the remaining time is

Accounting for a 38 ms token at 26.4 tok/s:

| | ms |
|---|---|
| always-active tensors, from VRAM | 2.8 |
| routed experts, ~17% from VRAM | 0.3 |
| routed experts, ~83% from RAM at sequential speed | 12.3 |
| **unaccounted** | **22** |

Sampled during decode, the CPU runs 5.4 of its 8 allotted cores (67%) with
5.9 core-seconds of iowait across an 11-second generation. So it is neither
compute-bound nor idle — the cores are **stalling**. Gathering ten scattered
experts out of a 512-expert tensor is random access, and random access does not
reach the bandwidth a sequential read does.

---

## Choosing a quantization: fit RAM, then stop

Quantization is usually discussed as a quality/size trade. Near the memory
boundary it is really a **tier** decision, and the tiers are 12x apart. Which
side of the line the working set lands on matters more than the bits per
weight.

`lm pull` shows this directly and picks accordingly:

```
quant                          size   fits memory     speed
IQ2_XXS                      24.4 GiB yes            ~23 tok/s
Q2_K_XL                      28.1 GiB yes <-- selected
Q4_K_M                       45.2 GiB  no             ~7 tok/s
```

The rule that follows: **quantize until the working set fits RAM, then stop.**
Smaller buys nothing once the bottleneck has moved, and costs quality for free.

For a *dense* model the rule inverts — every parameter is read per token, so a
larger quant is directly slower. `lm pull` applies a speed floor there
(`--min-speed`, default 4 tok/s) rather than maximising bits.

---

## Examples

```bash
lm doctor                      # hardware, GPU backend, RAM shortfall warning
lm doctor qwen3-next           # the placement plan and why

lm pull unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF        # auto quant
lm pull <repo> --quant Q4_K_M --name big                # force one
lm pull <repo> --min-speed 0                            # maximise quality

lm tune qwen3-next             # measure placement, then threads
lm tune qwen3-next --repeats 3 # more samples; noise is one-sided

lm run qwen3-next                        # interactive chat
lm run qwen3-next "explain X briefly"    # one-shot
lm serve qwen3-next                      # OpenAI-compatible API on :8099
lm ps ; lm stop ; lm list ; lm rm <name> --purge
```

In chat: `/bye` to exit, `/clear` to forget the conversation, `/system <text>`
to set a system prompt. Model names accept unambiguous prefixes, so
`lm run qwen3-next` is enough.

`lm serve` exposes `/v1/chat/completions`, so existing OpenAI clients work
against it unchanged with any dummy API key.

---

## Repository layout

| path | what it is |
|---|---|
| [`LM.md`](LM.md) | full CLI reference |
| [`LAPTOP-INFERENCE.md`](LAPTOP-INFERENCE.md) | the measured hardware model and every finding, failures included |
| `lm/` | the tool |
| `run-80b.sh` | standalone launcher, no CLI needed |
| `corestream/` | an earlier streaming engine that lost to llama.cpp by 4.5x — [post-mortem](docs/corestream-postmortem.md) |
| `tests/` | 93 tests |

### About `corestream/`

The repository is named after an engine that does not work well. It streams
weights from RAM to the GPU every token, which on this hardware is about 4.5x
slower than llama.cpp — PCIe moves 9.2 GiB/s while RAM reads at 36, so
computing where the weights already sit beats shipping them to the GPU.

It is kept, and its post-mortem with it, because building it produced the
bandwidth measurements the rest of this work is built on. A wrong architecture
measured carefully is more useful than a right one assumed.

## Requirements

`llama-server`, which Ollama already bundles — if Ollama is installed there is
nothing else to fetch. Set `LM_LLAMA_SERVER` to use your own llama.cpp build.
`lm doctor` reports what it found, including whether a GPU backend is present
and whether your largest model fits in available RAM.
