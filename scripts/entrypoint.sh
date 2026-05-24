#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${WHISPER_MODEL_PATH:-/models/ggml-large-v3.bin}"
MODEL_URL="${WHISPER_MODEL_URL:-https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin}"

mkdir -p "$(dirname "$MODEL_PATH")"
mkdir -p "${DATA_DIR:-/data}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[entrypoint] model missing at $MODEL_PATH"
    echo "[entrypoint] downloading from $MODEL_URL (~3GB, one-time)"
    tmp="${MODEL_PATH}.part"
    curl -L --fail --retry 3 --retry-delay 5 -o "$tmp" "$MODEL_URL"
    mv "$tmp" "$MODEL_PATH"
    echo "[entrypoint] model ready"
else
    echo "[entrypoint] model present at $MODEL_PATH"
fi

echo "[entrypoint] GPU visibility:"
nvidia-smi -L || echo "[entrypoint] WARNING: nvidia-smi unavailable - GPU may not be passed through"

exec /usr/local/bin/rust-whisper-mcp
