from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODEL_NAME = "yolov8s"
DEFAULT_IMAGE_SIZE = 640


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_engine(engine_dir: Path, model_name: str = DEFAULT_MODEL_NAME, image_size: int = DEFAULT_IMAGE_SIZE) -> Path:
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{model_name}.engine"
    meta_path = engine_dir / f"{model_name}.json"
    temp_onnx_path = engine_dir / f"{model_name}.bootstrap.onnx"

    if engine_path.exists() and meta_path.exists():
        return engine_path

    model = YOLO(f"{model_name}.pt")
    exported = Path(model.export(format="onnx", imgsz=image_size, simplify=False))
    exported.replace(temp_onnx_path)
    onnx_sha = sha256_file(temp_onnx_path)

    subprocess.run(
        ["trtexec", f"--onnx={temp_onnx_path}", f"--saveEngine={engine_path}"],
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = {
        "model_name": model_name,
        "source_weights": f"{model_name}.pt",
        "engine_path": str(engine_path),
        "onnx_sha256": onnx_sha,
        "image_size": image_size,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp_onnx_path.unlink(missing_ok=True)
    return engine_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    args = parser.parse_args()
    ensure_engine(Path(args.engine_dir), model_name=args.model_name, image_size=args.image_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
