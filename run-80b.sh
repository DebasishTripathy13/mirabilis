#!/usr/bin/env bash
# Run Qwen3-Next-80B-A3B at ~11.7 tok/s on a 6 GB / 30 GB laptop.
#
# Why this configuration, in short:
#   * 80B total but only ~3B active per token. Total size decides what must be
#     stored; active parameters decide what is read per token, and only the
#     second costs time. This is the one model shape that is both >70B and fast.
#   * UD-Q2_K_XL (28 GiB) is chosen so the working set fits in RAM. RAM reads at
#     36 GiB/s, NVMe at 2.9 GiB/s -- staying on the right side of that 12x cliff
#     matters far more than bits per weight.
#   * -ncmoe 40 keeps experts for the first 40 layers in RAM and puts the rest,
#     plus all attention, on the GPU. Measured best of the placements tried.
#   * -t 14 uses the P-cores and most E-cores. -t 20 measured *worse* (9.2 vs
#     12.3 tok/s): on memory-bound work the slow E-cores stall the fast ones.
set -euo pipefail

LIB=/usr/local/lib/ollama
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3-Next-80B-A3B-Instruct-GGUF}"
PORT="${PORT:-8099}"
CTX="${CTX:-4096}"
THREADS="${THREADS:-14}"
NCMOE="${NCMOE:-40}"

GGUF="$(readlink -f "$MODEL_DIR"/snapshots/*/*.gguf 2>/dev/null | head -1 || true)"
if [[ -z "$GGUF" || ! -f "$GGUF" ]]; then
  echo "Model not found under $MODEL_DIR" >&2
  echo "Fetch it with:" >&2
  echo "  huggingface-cli download unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF \\" >&2
  echo "      --include '*UD-Q2_K_XL*'" >&2
  exit 1
fi

echo "model   : $(du -h "$GGUF" | cut -f1)  $(basename "$GGUF")"
echo "threads : $THREADS   experts-on-CPU layers: $NCMOE   ctx: $CTX"
echo "serving : http://127.0.0.1:$PORT"

exec env LD_LIBRARY_PATH="$LIB" "$LIB/llama-server" \
  -m "$GGUF" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" \
  -ngl 999 \
  -ncmoe "$NCMOE" \
  -t "$THREADS" \
  -fa on \
  "$@"
