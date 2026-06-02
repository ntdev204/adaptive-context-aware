"""Export YOLO .pt models to TensorRT .engine using Ultralytics built-in export.

Uses the Ultralytics TensorRT export flow for every `.pt` found under the
configured model root. Any exporter side-products are removed after each build
so only `.engine` artifacts are retained.

Usage:
    python3 scripts/export_engine.py
    python3 scripts/export_engine.py --root /app/models --output-dir /app/models/engines
    python3 scripts/export_engine.py --root /app/models \
        --output-dir /tmp/engines \
        --codebase-output-dir /workspace/models/engines
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("export_engine")

DEFAULT_MODEL_ROOT = os.environ.get("CTX_PT_MODEL_ROOT", "/app/models")
DEFAULT_OUTPUT_DIR = os.environ.get("CTX_ENGINE_ROOT", "/app/models/engines")
DEFAULT_CODEBASE_OUTPUT_DIR = os.environ.get("CTX_CODEBASE_ENGINE_ROOT")
DEFAULT_WORKSPACE_GB = int(os.environ.get("ENGINE_WORKSPACE_GB", "2"))
DEFAULT_BATCH = int(os.environ.get("ENGINE_BATCH", "2"))
DEFAULT_DYNAMIC = os.environ.get("ENGINE_DYNAMIC", "true").lower() in {"1", "true", "yes", "on"}
DEFAULT_FP16 = os.environ.get("ENGINE_FP16", "true").lower() in {"1", "true", "yes", "on"}
DEFAULT_INT8 = os.environ.get("ENGINE_INT8", "false").lower() in {"1", "true", "yes", "on"}
DEFAULT_DATA = os.environ.get("ENGINE_DATA")
DEFAULT_IMGSZ = [480, 640]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all YOLO .pt -> TensorRT .engine")
    try:
        default_imgsz = _parse_imgsz_env(os.environ.get("ENGINE_IMGSZ"), DEFAULT_IMGSZ)
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument(
        "--root",
        default=DEFAULT_MODEL_ROOT,
        help="Root directory to scan for .pt files (default: CTX_PT_MODEL_ROOT or /app/models)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write .engine files into (default: CTX_ENGINE_ROOT or /app/models/engines)",
    )
    parser.add_argument(
        "--codebase-output-dir",
        default=DEFAULT_CODEBASE_OUTPUT_DIR,
        help="Optional host-mounted codebase directory that receives a copy of every exported .engine",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=DEFAULT_FP16,
        help="FP16 precision (default: ENGINE_FP16 or true; ignored when --int8 is enabled)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16 precision",
    )
    parser.add_argument(
        "--int8",
        dest="int8",
        action="store_true",
        default=DEFAULT_INT8,
        help="INT8 TensorRT export (default: ENGINE_INT8 or false)",
    )
    parser.add_argument(
        "--no-int8",
        dest="int8",
        action="store_false",
        help="Disable INT8 TensorRT export",
    )
    parser.add_argument(
        "--dynamic",
        dest="dynamic",
        action="store_true",
        default=DEFAULT_DYNAMIC,
        help="Enable dynamic TensorRT input shapes (default: ENGINE_DYNAMIC or true)",
    )
    parser.add_argument(
        "--static",
        dest="dynamic",
        action="store_false",
        help="Disable dynamic TensorRT input shapes",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Maximum export batch size when dynamic=True (default: ENGINE_BATCH or 2)",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Dataset YAML for INT8 calibration (default: ENGINE_DATA)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=default_imgsz,
        help="Input size as H W or a square side length (default: ENGINE_IMGSZ or 480 640)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=DEFAULT_WORKSPACE_GB,
        help="TensorRT workspace in GB (default: 2 or ENGINE_WORKSPACE_GB)",
    )
    args = parser.parse_args()

    model_root = Path(args.root)
    output_dir = Path(args.output_dir)
    codebase_output_dir = Path(args.codebase_output_dir) if args.codebase_output_dir else None
    if not model_root.exists():
        log.error("Model root not found: %s", model_root)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    if codebase_output_dir is not None:
        codebase_output_dir.mkdir(parents=True, exist_ok=True)
    pt_files = _iter_pt_files(model_root, output_dir)
    if not pt_files:
        log.error("No .pt files found under %s", model_root)
        raise SystemExit(1)

    os.environ.setdefault("YOLO_OFFLINE", "True")
    os.environ.setdefault("ULTRALYTICS_TELEMETRY", "0")

    import logging as _logging

    _logging.getLogger("ultralytics").setLevel(_logging.ERROR)

    output_map = _resolve_outputs(pt_files, output_dir)
    imgsz = args.imgsz if len(args.imgsz) > 1 else args.imgsz[0]
    failures = 0
    for pt_path in pt_files:
        engine_path = output_map[pt_path]
        codebase_engine_path = (
            codebase_output_dir / engine_path.name if codebase_output_dir is not None else None
        )
        if not export_model(
            pt_path=pt_path,
            engine_path=engine_path,
            codebase_engine_path=codebase_engine_path,
            fp16=args.fp16,
            int8=args.int8,
            dynamic=args.dynamic,
            batch=args.batch,
            data=args.data,
            imgsz=imgsz,
            workspace=args.workspace,
        ):
            failures += 1

    os._exit(1 if failures else 0)


def export_model(
    *,
    pt_path: Path,
    engine_path: Path,
    codebase_engine_path: Path | None = None,
    fp16: bool,
    int8: bool,
    dynamic: bool,
    batch: int,
    data: str | None,
    imgsz: list[int] | int,
    workspace: int,
) -> bool:
    if not pt_path.exists():
        log.error("Model not found: %s", pt_path)
        return False

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "Exporting %s -> %s  [dynamic=%s, batch=%d, FP16=%s, INT8=%s, data=%s, workspace=%dGB, imgsz=%s]",
        pt_path,
        engine_path,
        dynamic,
        batch,
        fp16,
        int8,
        data or "-",
        workspace,
        imgsz,
    )

    from ultralytics import YOLO

    model = YOLO(str(pt_path))

    log.info("Running TensorRT export (this takes 3-10 min on Jetson Orin)...")
    export_kwargs = {
        "format": "engine",
        "device": 0,
        "dynamic": dynamic,
        "batch": batch,
        "half": fp16 and not int8,
        "int8": int8,
        "imgsz": imgsz,
        "workspace": workspace,
        "simplify": False,
        "verbose": False,
    }
    if data:
        export_kwargs["data"] = data

    try:
        exported = Path(model.export(**export_kwargs))
    finally:
        _cleanup_intermediate_exports(pt_path)

    if exported.exists() and exported.resolve() != engine_path.resolve():
        exported.replace(engine_path)

    if engine_path.exists():
        if codebase_engine_path is not None:
            _copy_engine_artifact(engine_path, codebase_engine_path)
        size_mb = engine_path.stat().st_size / (1024**2)
        log.info("Engine saved: %s (%.1f MB)", engine_path, size_mb)
        return True

    log.error("Export failed -- engine not found at %s", engine_path)
    return False


def _cleanup_intermediate_exports(pt_path: Path) -> None:
    for path in (pt_path.with_suffix(".onnx"),):
        if path.exists():
            path.unlink()
            log.info("Removed intermediate export artifact: %s", path)


def _iter_pt_files(model_root: Path, output_dir: Path) -> list[Path]:
    pt_files: list[Path] = []
    for path in sorted(model_root.rglob("*.pt")):
        try:
            path.relative_to(output_dir)
            continue
        except ValueError:
            pass
        pt_files.append(path)
    return pt_files


def _resolve_outputs(pt_files: list[Path], output_dir: Path) -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    seen: dict[str, Path] = {}

    for pt_path in pt_files:
        engine_name = f"{pt_path.stem}.engine"
        if engine_name in seen:
            other = seen[engine_name]
            log.error("Duplicate model stem for engine output: %s and %s", other, pt_path)
            raise SystemExit(1)
        seen[engine_name] = pt_path
        mapping[pt_path] = output_dir / engine_name

    return mapping


def _parse_imgsz_env(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or not raw.strip():
        return default
    normalized = raw.replace(",", " ")
    values = [int(part) for part in normalized.split()]
    if len(values) not in {1, 2}:
        raise ValueError("ENGINE_IMGSZ must be one integer or H W")
    return values


def _copy_engine_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if destination.exists() and source.samefile(destination):
            return
    except OSError:
        pass
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)
    log.info("Engine copied to codebase: %s", destination)


if __name__ == "__main__":
    main()
