from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

DEFAULT_MODEL_PATH = "/app/models/fine_tuning/best.pt"
DEFAULT_ENGINE_OUTPUT = "/app/models/engines/best.engine"
DEFAULT_CODEBASE_ENGINE_OUTPUT = os.environ.get("CTX_CODEBASE_ENGINE_MODEL_PATH")
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
    codebase_engine_output: Path | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Path:
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_output
    meta_path = engine_output.with_suffix(".json")

    if engine_path.exists() and meta_path.exists():
        if codebase_engine_output is not None:
            _copy_artifact(engine_path, codebase_engine_output)
            _copy_artifact(meta_path, codebase_engine_output.with_suffix(".json"))
        return engine_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"missing detector weights: {model_path}. Mount your fine-tuned models into /app/models/fine_tuning."
        )

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    exported = Path(model.export(format="engine", imgsz=image_size, half=True))
    exported.replace(engine_path)
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
    if codebase_engine_output is not None:
        _copy_artifact(engine_path, codebase_engine_output)
        _copy_artifact(meta_path, codebase_engine_output.with_suffix(".json"))
    return engine_path


def _copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if destination.exists() and source.samefile(destination):
            return
    except OSError:
        pass
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--model-path", default=os.environ.get("CTX_PT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--engine-output", default=os.environ.get("CTX_ENGINE_MODEL_PATH", DEFAULT_ENGINE_OUTPUT))
    parser.add_argument(
        "--codebase-engine-output",
        default=DEFAULT_CODEBASE_ENGINE_OUTPUT,
        help="Optional host-mounted codebase path that receives a copy of the engine and metadata",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    args = parser.parse_args()
    ensure_engine(
        Path(args.engine_dir),
        model_path=Path(args.model_path),
        engine_output=Path(args.engine_output),
        codebase_engine_output=Path(args.codebase_engine_output) if args.codebase_engine_output else None,
        image_size=args.image_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
