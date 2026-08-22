"""Command line: pull, list, run, rm, serve, ps, stop, doctor."""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import estimate, hub, measure, registry, server, tune
from .gguf import read_info
from .hardware import detect

GIB = 1024**3


def _c(text: str, code: str) -> str:
    return text if not sys.stdout.isatty() else f"\033[{code}m{text}\033[0m"


def dim(t): return _c(t, "2")
def bold(t): return _c(t, "1")
def green(t): return _c(t, "32")
def yellow(t): return _c(t, "33")


# -- pull ------------------------------------------------------------------

def cmd_pull(args) -> int:
    hw = detect()
    repo = args.repo
    if "/" not in repo:
        print(f"Expected a Hugging Face repo like 'unsloth/{repo}-GGUF'.",
              file=sys.stderr)
        return 1

    print(f"Inspecting {bold(repo)} ...")
    try:
        meta = hub.repo_metadata(repo, token=args.token)
    except Exception:                              # noqa: BLE001
        meta = {}
    arch = meta.get("architecture", "")
    is_moe = estimate.is_moe_architecture(arch)
    params = meta.get("parameters", 0)
    if arch:
        shape = "MoE" if is_moe else "dense"
        size = f"{params/1e9:.0f}B " if params else ""
        print(dim(f"  {size}{shape}, architecture {arch}"))

    try:
        quants = hub.list_quants(repo, token=args.token)
    except Exception as exc:                       # noqa: BLE001 - user-facing
        print(f"Could not read repo: {exc}", file=sys.stderr)
        return 1
    if not quants:
        print("No GGUF files in that repo.", file=sys.stderr)
        return 1

    # Leave room for the parts that do not live in RAM: attention on the GPU,
    # plus headroom so the machine does not start swapping.
    budget = hw.usable_ram_gib + hw.usable_vram_gib
    if args.quant:
        wanted = args.quant.upper()
        chosen = next((q for q in quants if wanted in q.label.upper()), None)
        if chosen is None:
            print(f"No quant matching {args.quant!r}. Available:", file=sys.stderr)
            for q in quants:
                print(f"  {q.size_gib:7.1f} GiB  {q.label}", file=sys.stderr)
            return 1
    else:
        # A dense model reads everything per token, so an oversized quant is
        # slower in direct proportion. Insist on a usable rate rather than
        # maximising bits.
        chosen = hub.choose_quant(
            quants, budget,
            rate_for=lambda g: estimate.tokens_per_second(hw, g, is_moe=is_moe),
            min_rate=args.min_speed,
        )
        if chosen is None:
            smallest = quants[0]
            print(yellow(
                f"Nothing fits in {budget:.1f} GiB of RAM+VRAM. The smallest "
                f"available is {smallest.label} at {smallest.size_gib:.1f} GiB, "
                "which would be read partly from disk and be slow."))
            print("Re-run with --quant to force one anyway.")
            return 1

    print(f"\n{'quant':<28}{'size':>9}  {'fits':<6}{'speed':>10}")
    for q in quants:
        mark = green("yes   ") if q.size_gib <= budget else dim("no    ")
        speed = estimate.describe(hw, q.size_gib, is_moe=is_moe)
        star = bold("  <-- selected") if q is chosen else ""
        print(f"{q.label:<28}{q.size_gib:>7.1f} GiB  {mark}{speed:>10}{star}")

    if not is_moe and chosen is not None:
        rate = estimate.tokens_per_second(hw, chosen.size_gib, is_moe=False)
        if rate < 5:
            print(yellow(
                f"\nThis is a dense model: every parameter is read for every "
                f"token, so {chosen.size_gib:.0f} GiB crosses memory each time "
                f"and the estimate is {rate:.1f} tok/s. A smaller quant is "
                "proportionally faster; a mixture-of-experts model of similar "
                "or larger total size would be several times faster still."))

    if chosen.size_gib > hw.disk_free_gib:
        print(f"\nNot enough disk: need {chosen.size_gib:.1f} GiB, "
              f"have {hw.disk_free_gib:.0f} GiB.", file=sys.stderr)
        return 1

    print(f"\nDownloading {bold(chosen.label)} ({chosen.size_gib:.1f} GiB) ...")
    try:
        path = hub.download(repo, chosen, token=args.token)
    except Exception as exc:                       # noqa: BLE001
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    projector = ""
    try:
        found = hub.find_projector(repo, token=args.token)
        if found:
            print(dim(f"fetching vision projector {os.path.basename(found)} ..."))
            projector = hub.download_file(repo, found, token=args.token)
    except Exception:                              # noqa: BLE001
        projector = ""      # vision is optional; text still works without it

    try:
        info = read_info(path)
    except Exception:                              # noqa: BLE001
        info = None

    name = args.name or registry.default_name(repo, chosen.label)
    entry = registry.Entry(
        name=name, repo=repo, quant=chosen.label, path=path,
        size_gib=chosen.size_gib,
        architecture=info.architecture if info else "",
        layers=info.layers if info else 0,
        experts=info.experts if info else 0,
        experts_used=info.experts_used if info else 0,
        projector=projector,
    )
    registry.add(entry)
    print(f"\n{green('Installed')} as {bold(name)}"
          + (f"  ({info.describe()})" if info else "")
          + (dim("  + vision") if projector else ""))
    print(f"Run it with:  lm run {name}")
    return 0


# -- list / rm / ps --------------------------------------------------------

def cmd_list(args) -> int:
    entries = registry.load()
    if not entries:
        print("No models installed. Try:  lm pull unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF")
        return 0
    running = server.read_state()
    print(f"{'NAME':<40}{'SIZE':>9}  {'KIND':<18}{'STATUS'}")
    for name, e in sorted(entries.items()):
        status = ""
        if running and running.name == name:
            status = green("running")
        elif not e.exists:
            status = yellow("missing")
        print(f"{name:<40}{e.size_gib:>7.1f} GiB  {e.kind:<18}{status}")
    return 0


def cmd_rm(args) -> int:
    entry = registry.get(args.name)
    if entry is None:
        print(f"No model named {args.name!r}.", file=sys.stderr)
        return 1
    running = server.read_state()
    if running and running.name == entry.name:
        server.stop(running)

    freed = 0.0
    if args.purge:
        # The weights live in the shared Hugging Face cache, so deleting them
        # affects anything else using the same files. Only done on request.
        blob = os.path.realpath(entry.path)
        if os.path.exists(blob):
            freed = os.path.getsize(blob) / GIB
            os.remove(blob)
    registry.remove(entry.name)
    msg = f"Removed {entry.name}"
    if args.purge:
        msg += f" and deleted {freed:.1f} GiB of weights"
    else:
        msg += dim("  (weights kept in the Hugging Face cache; use --purge to delete)")
    print(msg)
    return 0


def cmd_ps(args) -> int:
    running = server.read_state()
    if running is None:
        print("Nothing running.")
        return 0
    print(f"{running.name}  pid {running.pid}  {running.url}")
    return 0


def cmd_stop(args) -> int:
    print("Stopped." if server.stop() else "Nothing running.")
    return 0


# -- serve / run -----------------------------------------------------------

def _plan_for(entry: registry.Entry, context: int | None):
    hw = detect()
    info = read_info(entry.path)
    p = tune.plan(hw, info, entry.size_gib, context,
                  override_ncmoe=entry.tuned_ncmoe,
                  override_threads=entry.tuned_threads,
                  override_cache_type=entry.tuned_cache_type,
                  override_gpu_layers=entry.tuned_gpu_layers,
                  override_cpu_mask=entry.tuned_cpu_mask or None)
    if entry.projector and os.path.exists(entry.projector):
        p.projector = entry.projector
        p.notes.append("Vision projector attached; image input enabled.")
    return hw, info, p


def _ensure_running(entry: registry.Entry, context: int | None,
                    verbose: bool = True) -> server.Running:
    running = server.read_state()
    if running and running.name == entry.name:
        return running
    if running:
        server.stop(running)

    hw, info, p = _plan_for(entry, context)
    if verbose:
        print(f"{bold(entry.name)}  {info.describe()}  {entry.size_gib:.1f} GiB")
        print(p.explain())
        print(dim("loading ..."), flush=True)
    started = time.time()
    running = server.start(entry.name, entry.path, p.to_args())
    if verbose:
        print(dim(f"ready in {time.time()-started:.0f}s\n"))
    return running


def cmd_serve(args) -> int:
    entry = registry.get(args.name)
    if entry is None or not entry.exists:
        print(f"No installed model named {args.name!r}.", file=sys.stderr)
        return 1
    running = _ensure_running(entry, args.context)
    print(f"Serving {bold(entry.name)} at {running.url}")
    print(dim("OpenAI-compatible endpoint: " + running.url + "/v1/chat/completions"))
    print(dim("Stop with:  lm stop"))
    return 0


def _timing_line(stats: dict, elapsed: float) -> str:
    """Format the server's own timings, falling back only if it reported none."""
    rate = stats.get("predicted_per_second")
    produced = stats.get("predicted_n") or stats.get("completion_tokens")
    prefill = stats.get("prompt_per_second")
    if rate:
        parts = [f"{rate:.1f} tok/s"]
        if produced:
            parts.append(f"{int(produced)} tokens")
        if prefill:
            parts.append(f"prefill {prefill:.0f} tok/s")
        parts.append(f"{elapsed:.1f}s")
        return "[" + ", ".join(parts) + "]"
    return f"[{elapsed:.1f}s]"


HELP = """
  /bye /exit   leave          /clear   forget the conversation
  /stats       last timing    /system  set the system prompt
"""


def cmd_run(args) -> int:
    entry = registry.get(args.name)
    if entry is None:
        print(f"No installed model named {args.name!r}. See: lm list", file=sys.stderr)
        return 1
    if not entry.exists:
        print(f"Weights for {entry.name} are missing; re-pull it.", file=sys.stderr)
        return 1

    running = _ensure_running(entry, args.context)

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        for piece in server.chat_stream(running.port, messages,
                                        temperature=args.temperature):
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()
        return 0

    try:
        import readline  # noqa: F401  -- enables line editing and history
    except ImportError:
        pass

    system = args.system or "You are a helpful assistant."
    messages: list[dict] = [{"role": "system", "content": system}]
    print(dim(f"Chatting with {entry.name}. /bye to exit, /? for commands."))

    while True:
        try:
            line = input(bold("\n>>> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/bye", "/exit", "/quit"):
            break
        if line in ("/?", "/help"):
            print(HELP)
            continue
        if line == "/clear":
            messages = [{"role": "system", "content": system}]
            print(dim("conversation cleared"))
            continue
        if line.startswith("/system"):
            system = line[len("/system"):].strip() or system
            messages = [{"role": "system", "content": system}]
            print(dim(f"system prompt set; conversation cleared"))
            continue
        if line == "/stats":
            print(dim("timing is printed after each reply"))
            continue

        messages.append({"role": "user", "content": line})
        reply: list[str] = []
        stats: dict = {}
        started = time.time()
        print()
        try:
            for piece in server.chat_stream(running.port, messages,
                                            temperature=args.temperature,
                                            stats=stats):
                reply.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print(dim("\n[interrupted]"))
            messages.pop()
            continue
        except Exception as exc:                   # noqa: BLE001
            print(f"\nerror: {exc}", file=sys.stderr)
            messages.pop()
            continue

        text = "".join(reply)
        messages.append({"role": "assistant", "content": text})
        elapsed = max(1e-6, time.time() - started)
        print(dim("\n\n" + _timing_line(stats, elapsed)))

    if not args.keep_alive:
        server.stop(running)
    return 0


def cmd_tune(args) -> int:
    entry = registry.get(args.name)
    if entry is None or not entry.exists:
        print(f"No installed model named {args.name!r}.", file=sys.stderr)
        return 1
    running = server.read_state()
    if running:
        server.stop(running)

    hw, info, base = _plan_for(entry, args.context)
    print(f"{bold(entry.name)}  {info.describe()}")
    print(dim(f"Each candidate loads the model and generates {args.tokens} "
              f"tokens {args.repeats}x, keeping the best. Takes a few minutes.\n"))
    print(f"{'placement':<44}{'tok/s':>8}")

    def show(r):
        rate = f"{r.tokens_per_second:8.2f}" if r.tokens_per_second else "  failed"
        print(f"{r.label:<44}{rate}{dim(r.spread)}", flush=True)

    results = measure.run(entry.name, entry.path, hw, info, base,
                          tokens=args.tokens, repeats=args.repeats,
                          on_result=show)
    best = max(results, key=lambda r: r.tokens_per_second, default=None)
    if best is None or not best.tokens_per_second:
        print("\nNo candidate ran successfully.", file=sys.stderr)
        return 1

    entry.tuned_ncmoe = best.ncmoe
    entry.tuned_threads = best.threads
    entry.tuned_cache_type = best.cache_type
    entry.tuned_gpu_layers = best.gpu_layers
    entry.tuned_cpu_mask = best.cpu_mask
    entry.tuned_tokens_per_second = best.tokens_per_second
    registry.add(entry)
    print(f"\n{green('Best')}: {best.label} at {best.tokens_per_second:.2f} tok/s")
    print(dim("Saved; `lm run` will use it from now on."))
    return 0


# -- doctor ----------------------------------------------------------------

def cmd_doctor(args) -> int:
    hw = detect()
    print(bold("hardware"))
    print(hw.summary())
    try:
        engine, libdir = server.find_engine()
        print(f"\nengine     {engine}")
        backend = server.find_gpu_backend(libdir)
        if backend:
            print(f"gpu backend {backend}")
        elif hw.has_gpu:
            print(yellow("gpu backend not found -- the engine will run on CPU only"))
    except FileNotFoundError as exc:
        print(f"\n{yellow('engine     not found')}\n  {exc}")

    print(f"\n{bold('what fits')}")
    budget = hw.usable_ram_gib + hw.usable_vram_gib
    print(f"  RAM+VRAM budget for weights: {budget:.1f} GiB")
    print(f"  A dense model of that size reads all of it per token.")
    print(f"  An MoE reads only its active experts, so it can be far larger.")

    if args.name:
        entry = registry.get(args.name)
        if entry and entry.exists:
            hw, info, p = _plan_for(entry, args.context)
            print(f"\n{bold('plan for ' + entry.name)}  ({info.describe()})")
            print(p.explain())
            print(dim("\n  flags: " + " ".join(p.to_args())))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lm", description="Run large language models on a laptop.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pull", help="download a model from Hugging Face")
    p.add_argument("repo", help="Hugging Face GGUF repo, e.g. unsloth/Qwen3-...-GGUF")
    p.add_argument("--quant", help="force a quantization (default: best that fits)")
    p.add_argument("--name", help="local name for the model")
    p.add_argument("--token", help="Hugging Face token for gated repos")
    p.add_argument("--min-speed", type=float, default=4.0, metavar="TOK/S",
                   help="do not pick a quant estimated below this rate "
                        "(default 4; use 0 to just maximise quality)")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("list", aliases=["ls"], help="list installed models")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("rm", aliases=["remove"], help="uninstall a model")
    p.add_argument("name")
    p.add_argument("--purge", action="store_true",
                   help="also delete the weights from the Hugging Face cache")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("run", help="chat with a model")
    p.add_argument("name")
    p.add_argument("prompt", nargs="?", help="one-shot prompt instead of a chat")
    p.add_argument("--context", type=int, help="context length")
    p.add_argument("--system", help="system prompt")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--keep-alive", action="store_true",
                   help="leave the server running after exit")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="start the server without chatting")
    p.add_argument("name")
    p.add_argument("--context", type=int)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("tune", help="measure the fastest placement for a model")
    p.add_argument("name")
    p.add_argument("--context", type=int)
    p.add_argument("--tokens", type=int, default=48,
                   help="tokens to generate per candidate")
    p.add_argument("--repeats", type=int, default=3,
                   help="runs per candidate; the best is kept, since "
                        "throughput noise only ever slows a run down")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("ps", help="show what is running")
    p.set_defaults(func=cmd_ps)

    p = sub.add_parser("stop", help="stop the running model")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("doctor", help="report hardware and placement plan")
    p.add_argument("name", nargs="?")
    p.add_argument("--context", type=int)
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
