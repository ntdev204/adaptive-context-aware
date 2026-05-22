"""Export YOLO .pt models to TensorRT .engine using Ultralytics built-in export.

Uses the same per-model export flow as the validated single-model script, then
applies it to every `.pt` found under the configured model root.

Usage:
    python3 scripts/export_engine.py
    python3 scripts/export_engine.py --root /app/models --output-dir /app/models/engines
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("export_engine")

DEFAULT_MODEL_ROOT = os.environ.get("CTX_PT_MODEL_ROOT", "/app/models")
DEFAULT_OUTPUT_DIR = os.environ.get("CTX_ENGINE_ROOT", "/app/models/engines")
DEFAULT_WORKSPACE_GB = int(os.environ.get("ENGINE_WORKSPACE_GB", "2"))
DEFAULT_IMGSZ = [480, 640]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all YOLO .pt -> TensorRT .engine")
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
        "--fp16",
        dest="fp16",
        action="store_true",
        default=True,
        help="FP16 precision (default: True)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=DEFAULT_IMGSZ,
        help="Input size as H W (default: 480 640 for landscape camera)",
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
    if not model_root.exists():
        log.error("Model root not found: %s", model_root)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
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
        if not export_model(
            pt_path=pt_path,
            engine_path=engine_path,
            fp16=args.fp16,
            imgsz=imgsz,
            workspace=args.workspace,
        ):
            failures += 1

    os._exit(1 if failures else 0)


def export_model(
    *,
    pt_path: Path,
    engine_path: Path,
    fp16: bool,
    imgsz: list[int] | int,
    workspace: int,
) -> bool:
    if not pt_path.exists():
        log.error("Model not found: %s", pt_path)
        return False

    try:
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "Exporting %s -> %s  [FP16=%s, workspace=%dGB, imgsz=%s]",
            pt_path,
            engine_path,
            fp16,
            workspace,
            imgsz,
        )

        from ultralytics import YOLO

        model = YOLO(str(pt_path))

        log.info("Running TensorRT export (this takes 3-10 min on Jetson Orin)...")
        exported = Path(
            model.export(
                format="engine",
                device=0,
                half=fp16,
                imgsz=imgsz,
                workspace=workspace,
                simplify=False,
                verbose=False,
            )
        )
        _finalize_exported_engine(exported, engine_path=engine_path, fp16=fp16, workspace_gb=workspace)
    except Exception as exc:
        log.error("Export failed for %s: %s", pt_path, exc)
        return False

    if not _is_valid_tensorrt_engine(engine_path):
        log.error("Export produced an invalid TensorRT engine: %s", engine_path)
        return False

    size_mb = engine_path.stat().st_size / (1024**2)
    log.info("Engine saved: %s (%.1f MB)", engine_path, size_mb)
    return True


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


def _finalize_exported_engine(exported: Path, *, engine_path: Path, fp16: bool, workspace_gb: int) -> Path:
    del fp16, workspace_gb
    if not exported.exists():
        raise FileNotFoundError(f"Ultralytics export did not produce an artifact: {exported}")

    direct_engine = exported if exported.suffix == ".engine" else exported.with_suffix(".engine")
    if direct_engine.exists():
        return _move_engine_artifact(direct_engine, engine_path)

    raise RuntimeError(
        f"Ultralytics returned {exported.name} but did not leave a sibling .engine artifact. "
        "Expected direct TensorRT export; refusing to rename or consume a non-engine file."
    )


def _move_engine_artifact(source: Path, target: Path) -> Path:
    if source.resolve() != target.resolve():
        source.replace(target)
    return target


def _is_valid_tensorrt_engine(engine_path: Path) -> bool:
    if not engine_path.exists():
        return False
    try:
        import tensorrt as trt
    except ImportError:
        return True

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    return engine is not None


if __name__ == "__main__":
    main()
