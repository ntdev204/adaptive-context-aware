"""export_engine.py — Export one or many .pt files to TensorRT .engine files.

Called by docker/Dockerfile.engine as the container CMD.
All settings are read from environment variables so they can be
overridden at ``docker run`` time without rebuilding the image.

Environment variables:
    CTX_PT_MODEL_PATH   : optional single .pt file path
    CTX_PT_MODEL_ROOT   : root directory to scan for .pt files (default: /app/models)
    CTX_ENGINE_OUTPUT   : destination .engine path for single-file mode
    CTX_ENGINE_ROOT     : destination directory for multi-file mode (default: /app/models/engines)
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


def export_single_engine(pt_path: Path, engine_out: Path) -> Path:
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


def _iter_pt_files(model_root: Path, engine_root: Path) -> list[Path]:
    pt_files = []
    for path in sorted(model_root.rglob("*.pt")):
        try:
            path.relative_to(engine_root)
            continue
        except ValueError:
            pass
        pt_files.append(path)
    return pt_files


def _resolve_multi_engine_outputs(pt_files: list[Path], engine_root: Path) -> dict[Path, Path]:
    outputs: dict[str, Path] = {}
    mapping: dict[Path, Path] = {}
    collisions: dict[str, list[str]] = {}

    for pt_path in pt_files:
        engine_name = f"{pt_path.stem}.engine"
        engine_out = engine_root / engine_name
        if engine_name in outputs:
            collisions.setdefault(engine_name, [str(outputs[engine_name])]).append(str(pt_path))
            continue
        outputs[engine_name] = pt_path
        mapping[pt_path] = engine_out

    if collisions:
        lines = []
        for engine_name, paths in sorted(collisions.items()):
            joined = ", ".join(paths)
            lines.append(f"{engine_name}: {joined}")
        print("[export_engine] ERROR: duplicate .pt stems would overwrite engine outputs:", file=sys.stderr)
        for line in lines:
            print(f"  - {line}", file=sys.stderr)
        sys.exit(1)

    return mapping


def export_engine() -> list[Path]:
    pt_model_path_raw = os.environ.get("CTX_PT_MODEL_PATH", "").strip()
    pt_model_root = Path(os.environ.get("CTX_PT_MODEL_ROOT", "/app/models"))
    engine_root = Path(os.environ.get("CTX_ENGINE_ROOT", "/app/models/engines"))

    if pt_model_path_raw:
        pt_path = Path(pt_model_path_raw)
        engine_out = Path(os.environ.get("CTX_ENGINE_OUTPUT", str(engine_root / f"{pt_path.stem}.engine")))
        return [export_single_engine(pt_path, engine_out)]

    if not pt_model_root.exists():
        print(f"[export_engine] ERROR: model root not found: {pt_model_root}", file=sys.stderr)
        sys.exit(1)

    pt_files = _iter_pt_files(pt_model_root, engine_root)
    if not pt_files:
        print(f"[export_engine] ERROR: no .pt files found under {pt_model_root}", file=sys.stderr)
        sys.exit(1)

    exported_paths: list[Path] = []
    output_map = _resolve_multi_engine_outputs(pt_files, engine_root)
    for pt_path in pt_files:
        exported_paths.append(export_single_engine(pt_path, output_map[pt_path]))
    return exported_paths


if __name__ == "__main__":
    export_engine()
