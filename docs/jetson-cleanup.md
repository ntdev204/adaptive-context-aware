# Jetson Cleanup: RAM + Docker Build/Image Garbage

Use this when your Jetson has been rebuilding images repeatedly and starts running out of RAM/disk.

Script: `scripts/jetson_cleanup.sh`

## What it does

1. Stops running Docker containers.
2. Removes old Docker artifacts:
   - stopped containers
   - unused/dangling images
   - build cache
   - unused networks
   - unused volumes
3. Reclaims Linux memory cache:
   - `sync`
   - drop page/slab cache
   - reset swap (`swapoff -a && swapon -a`)
4. Shows memory and Docker disk usage before/after.

## Important warning

This is destructive for old Docker build state:

- Removed images will need to be rebuilt/pulled again.
- Removed volumes may delete persisted local dev data that is not mounted from host.

Run it only when you accept rebuild cost.

## Usage

Preview only (safe):

```bash
bash scripts/jetson_cleanup.sh --dry-run
```

Run real cleanup:

```bash
sudo bash scripts/jetson_cleanup.sh --force
```

Interactive mode (asks confirmation):

```bash
sudo bash scripts/jetson_cleanup.sh
```

## Recommended flow on Jetson

1. Stop your app workload first.
2. Run dry-run to verify actions.
3. Run real cleanup with sudo.
4. Rebuild only the image you need next.

## Notes

- `drop_caches` needs elevated permissions.
- If swap is not configured, swap reset step is skipped safely.
- This script is intended for Linux/Jetson environments.
