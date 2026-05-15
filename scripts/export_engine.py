"""Export all YOLO .pt models under the models tree to TensorRT .engine files.

Uses Ultralytics' built-in TensorRT export entrypoint and scans the configured
model root for every `.pt` file.

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
        default=[480, 640],
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

    output_map = _resolve_outputs(pt_files, output_dir)
    imgsz = args.imgsz if len(args.imgsz) > 1 else args.imgsz[0]

    os.environ.setdefault("YOLO_OFFLINE", "True")
    os.environ.setdefault("ULTRALYTICS_TELEMETRY", "0")

    import logging as _logging

    _logging.getLogger("ultralytics").setLevel(_logging.ERROR)

    from ultralytics import YOLO

    failures = 0
    for pt_path in pt_files:
        engine_path = output_map[pt_path]
        log.info(
            "Exporting %s -> %s [FP16=%s, workspace=%dGB, imgsz=%s]",
            pt_path,
            engine_path,
            args.fp16,
            args.workspace,
            imgsz,
        )
        model = YOLO(str(pt_path))
        exported = Path(
            model.export(
                format="engine",
                device=0,
                half=args.fp16,
                imgsz=imgsz,
                workspace=args.workspace,
                simplify=False,
                verbose=False,
            )
        )
        if exported.exists() and exported.resolve() != engine_path.resolve():
            exported.replace(engine_path)

        if engine_path.exists():
            size_mb = engine_path.stat().st_size / (1024**2)
            log.info("Engine saved: %s (%.1f MB)", engine_path, size_mb)
        else:
            log.error("Export failed -- engine not found at %s", engine_path)
            failures += 1

    os._exit(1 if failures else 0)


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


if __name__ == "__main__":
    main()
