#!/usr/bin/env bash
set -euo pipefail

ENGINE_CACHE_DIR="${CTX_ENGINE_CACHE_DIR:-/app/models/engines}"
PT_MODEL_PATH="${CTX_PT_MODEL_PATH:-/app/models/fine_tuning/best.pt}"
ENGINE_MODEL_PATH="${CTX_ENGINE_MODEL_PATH:-/app/models/engines/best.engine}"
MODEL_IMAGE_SIZE="${CTX_MODEL_IMAGE_SIZE:-640}"
RUNTIME_BACKEND="${CTX_RUNTIME_BACKEND:-engine}"
REQUIRE_ENGINE_AT_BOOT="${CTX_REQUIRE_ENGINE_AT_BOOT:-false}"
MIN_FREE_DISK_MB="${CTX_MIN_FREE_DISK_MB:-512}"
BACKEND_FILE="${ENGINE_CACHE_DIR}/runtime_backend.env"
mkdir -p "${ENGINE_CACHE_DIR}"

log() {
  printf '[entrypoint] %s: %s\n' "$1" "$2" >&2
}

log_df() {
  for target in /app /app/models "${ENGINE_CACHE_DIR}"; do
    if [ -e "${target}" ]; then
      df -h "${target}" 2>/dev/null | awk 'NR == 2 {print "[entrypoint] INFO: disk " $6 " size=" $2 " used=" $3 " avail=" $4 " use=" $5}' >&2 || true
    fi
  done
}

check_free_disk() {
  free_kb="$(df -Pk "${ENGINE_CACHE_DIR}" 2>/dev/null | awk 'NR == 2 {print $4}')"
  if [ -n "${free_kb}" ]; then
    free_mb=$((free_kb / 1024))
    if [ "${free_mb}" -lt "${MIN_FREE_DISK_MB}" ]; then
      log WARN "low free disk at ${ENGINE_CACHE_DIR}: ${free_mb}MB available; minimum expected ${MIN_FREE_DISK_MB}MB"
    else
      log INFO "free disk at ${ENGINE_CACHE_DIR}: ${free_mb}MB"
    fi
  else
    log WARN "could not determine free disk for ${ENGINE_CACHE_DIR}"
  fi
}

log INFO "runtime_backend=${RUNTIME_BACKEND} camera_source=${CTX_CAMERA_SOURCE:-device} camera_backend=${CTX_CAMERA_BACKEND:-openni}"
log INFO "engine_model_path=${ENGINE_MODEL_PATH}"
log INFO "pt_model_path=${PT_MODEL_PATH}"
log INFO "openni sdk=${OPENNI_SDK_ROOT:-/opt/orbbec/openni} redist=${OPENNI2_REDIST:-unset} drivers=${OPENNI2_DRIVERS_PATH:-unset}"
if [ -d /dev/bus/usb ]; then
  log INFO "/dev/bus/usb is mounted"
else
  log WARN "/dev/bus/usb is not mounted; Astra S OpenNI camera cannot be discovered"
fi
if command -v lsusb >/dev/null 2>&1; then
  log INFO "USB devices visible to container:"
  lsusb 2>&1 | sed 's/^/[entrypoint] INFO: usb /' >&2 || true
fi
log_df
check_free_disk

echo "CTX_RUNTIME_BACKEND=${RUNTIME_BACKEND}" > "${BACKEND_FILE}"

if [ "${RUNTIME_BACKEND}" = "engine" ]; then
  if [ ! -f "${ENGINE_MODEL_PATH}" ]; then
    log WARN "missing TensorRT engine: ${ENGINE_MODEL_PATH}"
    log WARN "Runtime will start degraded so camera/sensor diagnostics remain available"
    log WARN "build the engine separately from ${PT_MODEL_PATH} to enable ready=true"
    case "${REQUIRE_ENGINE_AT_BOOT}" in
      1|true|TRUE|yes|YES|on|ON)
        log ERROR "CTX_REQUIRE_ENGINE_AT_BOOT=${REQUIRE_ENGINE_AT_BOOT}; refusing to start"
        exit 1
        ;;
    esac
  else
    engine_size="$(du -h "${ENGINE_MODEL_PATH}" 2>/dev/null | awk '{print $1}')"
    log INFO "TensorRT engine found: ${ENGINE_MODEL_PATH} size=${engine_size:-unknown}"
  fi
fi

if [ ! -f "${PT_MODEL_PATH}" ]; then
  log WARN "detector weights not found: ${PT_MODEL_PATH}"
else
  pt_size="$(du -h "${PT_MODEL_PATH}" 2>/dev/null | awk '{print $1}')"
  log INFO "detector weights found: ${PT_MODEL_PATH} size=${pt_size:-unknown}"
fi

log INFO "starting command: $*"

exec "$@"
