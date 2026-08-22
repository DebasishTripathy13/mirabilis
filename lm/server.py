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
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def start(name: str, model_path: str, args: list[str], port: int | None = None,
          quiet: bool = True) -> Running:
    engine, libdir = find_engine()
    port = port or free_port()
    env = dict(os.environ)
    if libdir:
        env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")

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
                temperature: float = 0.7):
    """Yield content deltas from the OpenAI-compatible streaming endpoint."""
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
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
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    yield piece
