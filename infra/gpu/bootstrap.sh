#!/usr/bin/env bash
# Runs inside the Runpod pod: starts vLLM (port 8000) and the speech server (port 8001).
set -euo pipefail

: "${SELFHOST_API_KEY:?SELFHOST_API_KEY must be set}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d /workspace/speech-venv ]; then
  uv venv /workspace/speech-venv --python 3.12
  uv pip install --python /workspace/speech-venv/bin/python \
    -r /workspace/repo/infra/gpu/speech-requirements.txt
fi

mkdir -p /workspace/logs

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key "$SELFHOST_API_KEY" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-prefix-caching \
  --max-model-len 8192 --gpu-memory-utilization 0.60 \
  > /workspace/logs/vllm.log 2>&1 &

cd /workspace/repo
/workspace/speech-venv/bin/uvicorn infra.selfhost.speech_server:app \
  --host 0.0.0.0 --port 8001 \
  > /workspace/logs/speech.log 2>&1 &

wait -n
exit 1
