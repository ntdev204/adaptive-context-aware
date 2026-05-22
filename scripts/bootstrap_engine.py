from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

DEFAULT_MODEL_PATH = "/app/models/fine_tuning/best.pt"
DEFAULT_ENGINE_OUTPUT = "/app/models/engines/best.engine"
DEFAULT_IMAGE_SIZE = 640


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_engine(
    engine_dir: Path,
    *,
    model_path: Path,
    engine_output: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Path:
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_output
    meta_path = engine_output.with_suffix(".json")

    if engine_path.exists() and meta_path.exists() and _is_valid_tensorrt_engine(engine_path):
        return engine_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"missing detector weights: {model_path}. Mount your fine-tuned models into /app/models/fine_tuning."
        )

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    exported = Path(model.export(format="engine", imgsz=image_size, half=True, device=0, simplify=False))
    _finalize_exported_engine(exported, engine_path=engine_path, fp16=True, workspace_gb=2)
    if not _is_valid_tensorrt_engine(engine_path):
        raise RuntimeError(f"generated TensorRT engine is invalid: {engine_path}")
    engine_sha = sha256_file(engine_path)

    metadata = {
        "model_name": model_path.stem,
        "source_weights": str(model_path),
        "engine_path": str(engine_path),
        "engine_sha256": engine_sha,
        "image_size": image_size,
        "format": "TensorRT engine",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return engine_path


def _finalize_exported_engine(exported: Path, *, engine_path: Path, fp16: bool, workspace_gb: int) -> Path:
    del fp16, workspace_gb
    if not exported.exists():
        raise FileNotFoundError(f"Ultralytics export did not produce an artifact: {exported}")

    direct_engine = exported if exported.suffix == ".engine" else exported.with_suffix(".engine")
    if direct_engine.exists():
        if direct_engine.resolve() != engine_path.resolve():
            direct_engine.replace(engine_path)
        return engine_path

    raise RuntimeError(
        f"Ultralytics returned {exported.name} but did not leave a sibling .engine artifact. "
        "Expected direct TensorRT export; refusing to rename or consume a non-engine file."
    )


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--model-path", default=os.environ.get("CTX_PT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--engine-output", default=os.environ.get("CTX_ENGINE_MODEL_PATH", DEFAULT_ENGINE_OUTPUT))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    args = parser.parse_args()
    ensure_engine(
        Path(args.engine_dir),
        model_path=Path(args.model_path),
        engine_output=Path(args.engine_output),
        image_size=args.image_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
