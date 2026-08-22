"""Starting, finding, and talking to the inference server.

The engine is llama.cpp's `llama-server`, located from whichever install is
present. Its quantized CPU kernels run near memory bandwidth, which is the
whole reason a large model is usable here at all -- a pure PyTorch CPU path
measured about 0.2 GiB/s against 36 GiB/s of available bandwidth.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .registry import HOME

STATE = os.path.join(HOME, "server.json")

_SEARCH = [
    "/usr/local/lib/ollama/llama-server",
    "/usr/lib/ollama/llama-server",
    "/opt/homebrew/bin/llama-server",
    "/usr/local/bin/llama-server",
]


def find_engine() -> tuple[str, str | None]:
    """Locate llama-server and any library directory it needs."""
    override = os.environ.get("LM_LLAMA_SERVER")
    if override and os.path.exists(override):
        return override, os.path.dirname(override)
    found = shutil.which("llama-server")
    if found:
        return found, None
    for path in _SEARCH:
        if os.path.exists(path):
            return path, os.path.dirname(path)
    raise FileNotFoundError(
        "llama-server not found. Install llama.cpp or Ollama (which bundles "
        "it), or set LM_LLAMA_SERVER to its path."
    )


def find_gpu_backend(libdir: str | None) -> str | None:
    """Path to the CUDA backend shared object, if one ships alongside.

    Ollama keeps `libggml-cuda.so` in a versioned subdirectory rather than
    beside the binary, and ggml does not search subdirectories. Without being
    told where it is, the server prints "no usable GPU found", ignores
    `--gpu-layers`, and runs entirely on the CPU -- while still appearing to
    work, just slowly. `GGML_BACKEND_PATH` wants the file itself; pointing it
    at the directory fails with "Is a directory".
    """
    if not libdir:
        return None
    candidates: list[str] = []
    for entry in sorted(os.listdir(libdir), reverse=True):   # newest CUDA first
        sub = os.path.join(libdir, entry)
        if os.path.isdir(sub) and entry.startswith(("cuda", "hip", "rocm")):
            so = os.path.join(sub, "libggml-cuda.so")
            if os.path.exists(so):
                candidates.append(so)
    beside = os.path.join(libdir, "libggml-cuda.so")
    if os.path.exists(beside):
        candidates.append(beside)
    return candidates[0] if candidates else None


def engine_env(libdir: str | None) -> dict:
    """Environment that lets the engine actually find the GPU."""
    env = dict(os.environ)
    if not libdir:
        return env
    backend = find_gpu_backend(libdir)
    paths = [libdir]
    if backend:
        paths.insert(0, os.path.dirname(backend))
        env["GGML_BACKEND_PATH"] = backend
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        paths + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    return env


def free_port(preferred: int = 8099) -> int:
    for port in range(preferred, preferred + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port available")


@dataclass
class Running:
    name: str
    pid: int
    port: int
    path: str

    @property
    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def read_state() -> Running | None:
    try:
        with open(STATE) as f:
            running = Running(**json.load(f))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return running if running.alive else None


def write_state(running: Running | None) -> None:
    os.makedirs(HOME, exist_ok=True)
    if running is None:
        try:
            os.remove(STATE)
        except OSError:
            pass
        return
    with open(STATE, "w") as f:
        json.dump(running.__dict__, f)


def wait_healthy(port: int, proc: subprocess.Popen, timeout: float = 1800) -> bool:
    """Poll until the server answers, or it exits.

    Loading a large model takes a while: the file has to be paged in, which on
    a 30 GiB model is tens of seconds even from cache. The generous timeout is
    deliberate.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            # 503 "Loading model" until the weights are resident, so only a
            # 200 means the server can actually answer.
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def start(name: str, model_path: str, args: list[str], port: int | None = None,
          quiet: bool = True) -> Running:
    engine, libdir = find_engine()
    port = port or free_port()
    env = engine_env(libdir)

    cmd = [engine, "-m", model_path, "--host", "127.0.0.1", "--port", str(port),
           "--no-webui"] + args
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
        start_new_session=True,
    )
    if not wait_healthy(port, proc):
        detail = ""
        if proc.stderr is not None:
            detail = (proc.stderr.read() or b"")[-500:].decode(errors="replace")
        try:
            proc.kill()
        except OSError:
            pass
        raise RuntimeError(f"server failed to start.\n{detail.strip()}")

    running = Running(name=name, pid=proc.pid, port=port, path=model_path)
    write_state(running)
    return running


def stop(running: Running | None = None) -> bool:
    running = running or read_state()
    if running is None:
        return False
    try:
        os.killpg(os.getpgid(running.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            os.kill(running.pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(60):
        if not running.alive:
            break
        time.sleep(0.5)
    if running.alive:
        try:
            os.kill(running.pid, signal.SIGKILL)
        except OSError:
            pass
    write_state(None)
    return True


def chat_stream(port: int, messages: list[dict], max_tokens: int = 1024,
                temperature: float = 0.7, stats: dict | None = None):
    """Yield content deltas from the OpenAI-compatible streaming endpoint.

    If `stats` is given it is filled in with the server's own timings once the
    stream ends. Those are the real numbers -- tokens actually produced and the
    time spent producing them. Estimating a rate from the length of the text
    is misleading, badly so on short replies, where fixed costs dominate and
    characters-per-token varies a lot.
    """
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # The OpenAI usage extension is requested so a rate can still be
        # reported against servers implementing only that. llama.cpp's own
        # timings arrive in the final chunk without asking, and asking for
        # them per token measured 5% slower -- 23.81 against 25.04 tok/s --
        # which is instrumentation slowing the thing it measures.
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if stats is not None:
                timings = chunk.get("timings")
                if isinstance(timings, dict):
                    stats.update(timings)
                usage = chunk.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    stats.setdefault("completion_tokens", usage["completion_tokens"])
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    yield piece
