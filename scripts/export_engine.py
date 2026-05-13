"""export_engine.py — Export best.pt to TensorRT .engine file.

Called by docker/Dockerfile.engine as the container CMD.
All settings are read from environment variables so they can be
overridden at ``docker run`` time without rebuilding the image.

Environment variables:
    CTX_PT_MODEL_PATH   : path to the .pt weight file  (default: /app/models/fine_tuning/best.pt)
    CTX_ENGINE_OUTPUT   : destination .engine path      (default: /app/models/engines/yolo11s.engine)
    ENGINE_IMG_SIZE     : inference image size           (default: 640)
    ENGINE_HALF         : use FP16 half precision        (default: True)
    ENGINE_WORKSPACE_GB : TensorRT workspace in GB       (default: 4)
    ENGINE_BATCH        : static batch size              (default: 1)

Usage (inside container):
    python scripts/export_engine.py

Usage (override at runtime):
    docker run --gpus all --rm \\
        -v "$(pwd)/models:/app/models" \\
        -e ENGINE_HALF=False \\
        -e ENGINE_IMG_SIZE=480 \\
        ctx-aware:engine
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, str(default)).strip().lower()
    return val in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_engine() -> Path:
    pt_path = Path(os.environ.get("CTX_PT_MODEL_PATH", "/app/models/fine_tuning/best.pt"))
    engine_out = Path(os.environ.get("CTX_ENGINE_OUTPUT", "/app/models/engines/yolo11s.engine"))
    img_size = _env_int("ENGINE_IMG_SIZE", 640)
    half = _env_bool("ENGINE_HALF", True)
    workspace_gb = _env_int("ENGINE_WORKSPACE_GB", 4)
    batch = _env_int("ENGINE_BATCH", 1)

    print(f"[export_engine] Source   : {pt_path}")
    print(f"[export_engine] Output   : {engine_out}")
    print(f"[export_engine] img_size : {img_size}  half: {half}  workspace: {workspace_gb}GB  batch: {batch}")

    if not pt_path.exists():
        print(f"[export_engine] ERROR: .pt file not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        print("[export_engine] ERROR: ultralytics not installed.", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(pt_path))

    # Export — ultralytics writes the .engine next to the .pt by default
    exported_path = Path(
        model.export(
            format="engine",
            imgsz=img_size,
            half=half,
            workspace=workspace_gb,
            batch=batch,
            device=0,  # GPU 0 (required for TensorRT export)
        )
    )

    # Move to the configured output location
    engine_out.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != engine_out.resolve():
        exported_path.replace(engine_out)

    engine_sha = sha256_file(engine_out)
    meta = {
        "source_pt": str(pt_path),
        "engine_path": str(engine_out),
        "engine_sha256": engine_sha,
        "img_size": img_size,
        "half": half,
        "workspace_gb": workspace_gb,
        "batch": batch,
        "format": "TensorRT engine",
    }
    meta_path = engine_out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[export_engine] ✓ Engine saved  : {engine_out}")
    print(f"[export_engine] ✓ Metadata saved: {meta_path}")
    print(f"[export_engine] ✓ SHA-256       : {engine_sha}")
    return engine_out


if __name__ == "__main__":
    export_engine()
