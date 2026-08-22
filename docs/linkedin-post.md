# LinkedIn post

Draft below. The first two lines are what shows before "see more", so they
carry the hook.

---

I got an 80-billion-parameter model running at 23 tokens/sec on a laptop with
6 GB of VRAM.

The interesting part isn't that it works. It's that the 2x speedup came from
four things, and I'd have guessed none of them.

Here's what actually mattered:

**1. The GPU was never being used.** The server printed "no usable GPU found"
and carried on anyway. Ollama ships its CUDA library in a subdirectory that
ggml doesn't search, and the fix is an environment variable pointing at the
file, not the folder. Worth 1.65x. Nothing crashed. It just ran at a third of
the speed for weeks.

**2. The CPU was downclocking mid-inference.** Throughput climbed with output
length — 9 tok/s for a short reply, 24 for a long one. That's not the model
warming up, it's the clock. Memory-bound inference spends most of its time
*stalled* on RAM, the governor reads that as idleness and drops to 400 MHz,
and then the compute between stalls runs slow too. Short replies never escape
it, which is exactly the chat case.

**3. A browser was holding 5 GB.** That 5 GB decided whether the model stayed
in RAM or got re-read from disk every token, at a twelfth of the speed.
Closing it was worth more than any tuning flag.

**4. The model shape.** A dense 70B reads every parameter per token and cannot
be fast on this hardware at any quantization. A mixture-of-experts 80B reads
about 3B. Total size decides what you store; active parameters decide what you
read — and only the second one costs time.

What *didn't* work is the part I keep thinking about. Speculative decoding
measured slower. Huge pages did nothing. So did prompt-lookup drafting and
every expert-placement heuristic I tried. These are the techniques everyone
recommends, and on this hardware they were flat or negative.

So I stopped guessing and wrote a tool that measures: profiles the machine,
predicts what a model will do before you download it, then sweeps real
configurations and keeps what actually wins. Every failed experiment is in the
repo next to the successful ones, because the negative results cost me the most
time and are the ones nobody publishes.

Apache 2.0, all measurements reproducible on the machine described:
https://github.com/DebasishTripathy13/mirabilis

---

## Shorter variant (if the above runs long)

I got an 80B model running at 23 tokens/sec on a 6 GB laptop — 2x what Ollama
manages on an 8B on the same machine.

The 2x didn't come from where I expected.

The GPU was never being used: the server printed "no usable GPU found" and
carried on. The CPU was downclocking to 400 MHz mid-inference, because
memory-bound work looks idle to the frequency governor. A browser was holding
the 5 GB that decided whether the model stayed in RAM.

Meanwhile the famous techniques — speculative decoding, huge pages,
prompt-lookup drafting — measured flat or negative on this hardware.

So I wrote a tool that measures instead of assuming, and published the failed
experiments alongside the wins. The negative results cost the most time and
are the ones nobody writes down.

https://github.com/DebasishTripathy13/mirabilis
