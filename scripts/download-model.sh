#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="models"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B-UD-Q6_K_XL.gguf}"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
PART_PATH="$MODEL_PATH.part"
MODEL_URL="${MODEL_URL:-}"
EXPECTED_SHA256="${MODEL_SHA256:-}"

if [[ -f "$MODEL_PATH" ]]; then
  echo "Modelo ya disponible en: $MODEL_PATH"
  exit 0
fi

if [[ -z "$MODEL_URL" ]]; then
  echo "ERROR: define MODEL_URL antes de ejecutar este script."
  echo "Ejemplo:"
  echo "  export MODEL_URL=\"https://huggingface.co/ORG/REPO/resolve/main/$MODEL_NAME?download=true\""
  exit 1
fi

mkdir -p "$MODEL_DIR"

echo "Descargando modelo a: $MODEL_PATH"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 --retry-delay 2 -C - -o "$PART_PATH" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -c -O "$PART_PATH" "$MODEL_URL"
else
  echo "ERROR: no se encontro ni curl ni wget en PATH."
  exit 1
fi

mv "$PART_PATH" "$MODEL_PATH"

if [[ -n "$EXPECTED_SHA256" ]]; then
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: MODEL_SHA256 fue definido pero sha256sum no esta disponible."
    exit 1
  fi

  ACTUAL_SHA256="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
  if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: checksum SHA-256 invalido."
    echo "Esperado: $EXPECTED_SHA256"
    echo "Actual:   $ACTUAL_SHA256"
    exit 1
  fi
fi

echo "Listo: $MODEL_PATH"
