#!/usr/bin/env bash
set -Eeuo pipefail

# Jetson cleanup helper:
# - Reclaims reclaimable memory (page cache/slab/swap)
# - Deletes old Docker artifacts (images/build cache/containers/volumes/networks)
#
# Usage:
#   bash scripts/jetson_cleanup.sh --dry-run
#   sudo bash scripts/jetson_cleanup.sh --force

DRY_RUN=0
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: jetson_cleanup.sh [--dry-run] [--force]

Options:
  --dry-run   Print commands but do not execute destructive cleanup.
  --force     Skip confirmation prompt.
  -h, --help  Show this help message.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

print_mem() {
  echo "Memory snapshot:"
  free -h || true
  if command -v swapon >/dev/null 2>&1; then
    echo
    swapon --show || true
  fi
  echo
}

need_cmd docker
need_cmd free

echo "This will remove old Docker data and reclaim RAM caches."
echo "Targets:"
echo "  - Stopped containers"
echo "  - Dangling/unused images"
echo "  - Docker build cache"
echo "  - Unused networks"
echo "  - Unused volumes"
echo "  - Linux page cache + slab cache"
echo

if [[ "$FORCE" -ne 1 ]]; then
  read -r -p "Continue? [y/N]: " answer
  case "${answer:-N}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

echo
echo "Before cleanup"
print_mem

echo "Docker disk usage before:"
run "docker system df"
echo

echo "Stopping running containers..."
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] docker ps -q | xargs -r docker stop"
else
  docker ps -q | xargs -r docker stop
fi
echo

echo "Pruning Docker artifacts..."
run "docker container prune -f"
run "docker image prune -a -f"
run "docker builder prune -a -f"
run "docker network prune -f"
run "docker volume prune -f"
run "docker system prune -a -f --volumes"
echo

echo "Reclaiming Linux caches..."
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] sync"
  echo "[dry-run] echo 3 > /proc/sys/vm/drop_caches"
  echo "[dry-run] swapoff -a && swapon -a"
else
  sync
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches
  else
    echo "Warning: cannot write /proc/sys/vm/drop_caches (try sudo)." >&2
  fi
  if command -v swapoff >/dev/null 2>&1 && command -v swapon >/dev/null 2>&1; then
    swapoff -a || true
    swapon -a || true
  fi
fi
echo

echo "After cleanup"
print_mem

echo "Docker disk usage after:"
run "docker system df"
echo

echo "Cleanup completed."
