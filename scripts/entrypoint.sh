#!/usr/bin/env bash
set -euo pipefail

ENGINE_CACHE_DIR="${CTX_ENGINE_CACHE_DIR:-/app/models/engines}"
PT_MODEL_PATH="${CTX_PT_MODEL_PATH:-/app/models/fine_tuning/best.pt}"
ENGINE_MODEL_PATH="${CTX_ENGINE_MODEL_PATH:-/app/models/engines/best.engine}"
MODEL_IMAGE_SIZE="${CTX_MODEL_IMAGE_SIZE:-640}"
BACKEND_FILE="${ENGINE_CACHE_DIR}/runtime_backend.env"
mkdir -p "${ENGINE_CACHE_DIR}"

echo "CTX_RUNTIME_BACKEND=engine" > "${BACKEND_FILE}"

python /app/scripts/bootstrap_engine.py \
  --engine-dir "${ENGINE_CACHE_DIR}" \
  --model-path "${PT_MODEL_PATH}" \
  --engine-output "${ENGINE_MODEL_PATH}" \
  --image-size "${MODEL_IMAGE_SIZE}"

exec "$@"
