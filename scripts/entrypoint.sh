#!/usr/bin/env bash
set -euo pipefail

ENGINE_CACHE_DIR="${CTX_ENGINE_CACHE_DIR:-/app/models/engines}"
PT_MODEL_PATH="${CTX_PT_MODEL_PATH:-/app/models/fine_tuning/best.pt}"
ENGINE_MODEL_PATH="${CTX_ENGINE_MODEL_PATH:-/app/models/engines/best.engine}"
MODEL_IMAGE_SIZE="${CTX_MODEL_IMAGE_SIZE:-640}"
RUNTIME_BACKEND="${CTX_RUNTIME_BACKEND:-engine}"
BACKEND_FILE="${ENGINE_CACHE_DIR}/runtime_backend.env"
mkdir -p "${ENGINE_CACHE_DIR}"

echo "CTX_RUNTIME_BACKEND=${RUNTIME_BACKEND}" > "${BACKEND_FILE}"

if [ "${RUNTIME_BACKEND}" = "engine" ]; then
  if [ ! -f "${ENGINE_MODEL_PATH}" ]; then
    echo "[entrypoint] ERROR: missing TensorRT engine: ${ENGINE_MODEL_PATH}" >&2
    echo "[entrypoint] Build the engine separately from ${PT_MODEL_PATH} before starting control-api." >&2
    exit 1
  fi
fi

exec "$@"
