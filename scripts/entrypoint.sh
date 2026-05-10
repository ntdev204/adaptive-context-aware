#!/usr/bin/env bash
set -euo pipefail

ENGINE_CACHE_DIR="${CTX_ENGINE_CACHE_DIR:-/app/models/engines}"
ONNX_DIR="${CTX_ONNX_DIR:-/app/models/onnx}"
BACKEND_FILE="${ENGINE_CACHE_DIR}/runtime_backend.env"
mkdir -p "${ENGINE_CACHE_DIR}" "${ONNX_DIR}"

echo "CTX_RUNTIME_BACKEND=onnxruntime" > "${BACKEND_FILE}"

build_engine() {
  local onnx_path="$1"
  local name
  local sha
  local engine_path
  local hash_path

  name="$(basename "${onnx_path}" .onnx)"
  sha="$(sha256sum "${onnx_path}" | awk '{print $1}')"
  engine_path="${ENGINE_CACHE_DIR}/${name}.engine"
  hash_path="${ENGINE_CACHE_DIR}/${name}.sha256"

  if [[ -f "${engine_path}" && -f "${hash_path}" ]] && [[ "$(cat "${hash_path}")" == "${sha}" ]]; then
    echo "engine cache hit for ${name}"
    return 0
  fi

  if command -v trtexec >/dev/null 2>&1; then
    echo "building TensorRT engine for ${name}"
    if trtexec --onnx="${onnx_path}" --saveEngine="${engine_path}" >/tmp/trtexec-"${name}".log 2>&1; then
      echo "${sha}" > "${hash_path}"
      echo "CTX_RUNTIME_BACKEND=tensorrt" > "${BACKEND_FILE}"
      return 0
    fi
    echo "WARNING: trtexec failed for ${name}, falling back to ONNX Runtime"
  else
    echo "WARNING: trtexec not available, falling back to ONNX Runtime"
  fi

  rm -f "${engine_path}"
  echo "${sha}" > "${hash_path}"
}

shopt -s nullglob
for onnx in "${ONNX_DIR}"/*.onnx; do
  build_engine "${onnx}"
done
shopt -u nullglob

exec "$@"
