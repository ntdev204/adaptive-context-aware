#!/usr/bin/env bash
set -euo pipefail

ENGINE_CACHE_DIR="${CTX_ENGINE_CACHE_DIR:-/app/models/engines}"
MODEL_NAME="${CTX_MODEL_NAME:-yolov8s}"
MODEL_IMAGE_SIZE="${CTX_MODEL_IMAGE_SIZE:-640}"
BACKEND_FILE="${ENGINE_CACHE_DIR}/runtime_backend.env"
mkdir -p "${ENGINE_CACHE_DIR}"

echo "CTX_RUNTIME_BACKEND=engine" > "${BACKEND_FILE}"

python /app/scripts/bootstrap_engine.py \
  --engine-dir "${ENGINE_CACHE_DIR}" \
  --model-name "${MODEL_NAME}" \
  --image-size "${MODEL_IMAGE_SIZE}"

exec "$@"
