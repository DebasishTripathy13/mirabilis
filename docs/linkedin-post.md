# LinkedIn post

The first two lines are what shows before "see more", so they carry the hook.

---

I was curious whether my laptop could run a genuinely large model. Turns out
it can — an 80B running locally at ~23 tokens/sec on 6 GB of VRAM.

No cloud, no rented GPU. Just the machine I already own.

I started poking at this because I wanted to know where the actual limit was,
and I kept finding that the limit wasn't where I assumed. A few things
surprised me enough that they seem worth sharing:

**The GPU wasn't being used at all.** The server printed "no usable GPU found"
and just carried on. It never crashed, never warned loudly — it simply ran at a
third of the speed. Fixing that alone was 1.65x.

**The CPU was downclocking mid-inference.** Throughput kept climbing with reply
length: 9 tok/s for a short answer, 24 for a long one. That's not the model
warming up — it's the clock. Memory-bound work spends most of its time waiting
on RAM, the frequency governor reads that as idle, and drops to 400 MHz.

**A browser tab was holding 5 GB**, and that 5 GB decided whether the model
lived in RAM or got re-read from disk every token. Closing it helped more than
any tuning flag.

**Model shape matters more than model size.** A dense 70B has to read every
parameter for every token and can't be fast on this hardware, whatever you do.
A mixture-of-experts 80B reads about 3B per token. Total size decides what you
store; active parameters decide what you read.

The thing I found most useful, honestly, was everything that *didn't* work.
Speculative decoding measured slower. Huge pages did nothing. Same for
prompt-lookup drafting and every expert-placement trick I tried. These are the
techniques you read about everywhere, and on my hardware they were flat or
negative.

So I stopped guessing and wrote a small tool that measures instead: profiles
the machine, tells you what a model will do before you download 28 GB of it,
then tries real configurations and keeps whichever actually wins. It's tuned
for my laptop, but the point is that it re-measures on yours.

I've put the failed experiments in the repo alongside the working ones, because
those cost me the most time and nobody writes them down.

Sharing it in case it's useful to anyone else poking at the same question. It's
Apache 2.0 and I'd genuinely like to see people take it further — especially on
hardware different from mine, where I'd expect some of these findings to invert.

https://github.com/DebasishTripathy13/mirabilis

---

## Shorter variant

I was curious whether my laptop could run a genuinely large model. It can — an
80B at ~23 tokens/sec on 6 GB of VRAM, entirely local.

What surprised me was where the speed actually came from. The GPU wasn't being
used at all (the server said so and carried on anyway). The CPU was
downclocking to 400 MHz mid-inference, because memory-bound work looks idle to
the frequency governor. A browser tab was holding the 5 GB that decided whether
the model stayed in RAM.

Meanwhile the techniques everyone recommends — speculative decoding, huge
pages, prompt-lookup drafting — measured flat or slower on my hardware.

So I wrote a small tool that measures rather than assumes, and published the
failed experiments next to the working ones. Those cost the most time and
nobody writes them down.

Apache 2.0, and I'd love to see it tried on hardware unlike mine:
https://github.com/DebasishTripathy13/mirabilis
