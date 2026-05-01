#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BIN="runtime/llama-server.exe"
MODEL="models/Qwen3.5-9B-UD-Q6_K_XL.gguf"
MODEL_ALIASES_DEFAULT='gpt-3.5-turbo,Qwen3.5-9B-UD-Q6_K_XL.gguf,models\Qwen3.5-9B-UD-Q6_K_XL,models/Qwen3.5-9B-UD-Q6_K_XL'
MODEL_ALIASES="${LLAMA_MODEL_ALIAS:-$MODEL_ALIASES_DEFAULT}"
DEFAULT_API_KEY="llama.cpp"

# ── Hardware fijo: RTX 3060 12 GB ─────────────────────────────────────────────
CTX_SIZE="${LLAMA_CTX_SIZE:-32768}"
PARALLEL="${LLAMA_PARALLEL:-1}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"    # todas las capas en GPU

# KV cache comprimido al máximo para que quepa en VRAM
CACHE_TYPE_K="${LLAMA_CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${LLAMA_CACHE_TYPE_V:-q4_0}"

# Batch pequeño = menos pico de VRAM durante el prefill
BATCH_SIZE="${LLAMA_BATCH_SIZE:-256}"
UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-128}"

# Flash attention: imprescindible para 32k, reduce VRAM del attention
FLASH_ATTN="on"

# Dejar que llama.cpp use RAM si la VRAM se llena (comportamiento por defecto)
# --no-mmap evita que el modelo compita con el KV cache por VRAM al inicio
EXTRA_FLAGS="--no-mmap"
# ──────────────────────────────────────────────────────────────────────────────

if [[ ! -f "$BIN" ]]; then
  echo "ERROR: no se encontro $BIN"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: no se encontro $MODEL"
  exit 1
fi

echo "========================================"
echo " Qwen3.5-9B  →  http://127.0.0.1:8080"
echo "========================================"
echo " GPU        : RTX 3060 12 GB"
echo " Contexto   : $CTX_SIZE tokens"
echo " KV cache   : q4_0/q4_0  (~1.8 GB)"
echo " Batch      : $BATCH_SIZE / $UBATCH_SIZE"
echo " Flash attn : $FLASH_ATTN"
echo " RAM overflow: activo (automático)"
echo "========================================"
echo

API_KEY="${LLAMA_API_KEY:-$DEFAULT_API_KEY}"

exec "$BIN" \
  -m "$MODEL" \
  -ngl "$GPU_LAYERS" \
  -c "$CTX_SIZE" \
  -np "$PARALLEL" \
  --host 127.0.0.1 \
  --port 8080 \
  -a "$MODEL_ALIASES" \
  -fa "$FLASH_ATTN" \
  -b "$BATCH_SIZE" \
  -ub "$UBATCH_SIZE" \
  -ctk "$CACHE_TYPE_K" \
  -ctv "$CACHE_TYPE_V" \
  --n-predict 16384 \
  --reasoning-budget 0 \
  --api-key "$API_KEY" \
  $EXTRA_FLAGS \
  "$@"