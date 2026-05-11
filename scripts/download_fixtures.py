from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    root = ROOT
    _extract_cctv_person_subset(
        source_dir=root / "data" / "fine_tuning" / "cctv_person",
        output_dir=root / "tests" / "fixtures" / "cctv_person_subset",
        max_images=20,
    )
    _extract_mot20_subset(
        source_dir=root / "data" / "fine_tuning" / "mot20",
        output_dir=root / "tests" / "fixtures" / "mot20_subset",
        max_frames_per_sequence=30,
    )


def _extract_cctv_person_subset(source_dir: Path, output_dir: Path, *, max_images: int) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    for split in ("train", "valid", "test"):
        images_dir = source_dir / split / "images"
        labels_dir = source_dir / split / "labels"
        if not images_dir.exists():
            continue
        for image_path in sorted(images_dir.glob("*")):
            if copied >= max_images:
                break
            label_path = labels_dir / f"{image_path.stem}.txt"
            shutil.copy2(image_path, output_dir / "images" / image_path.name)
            if label_path.exists():
                shutil.copy2(label_path, output_dir / "labels" / label_path.name)
            copied += 1
        if copied >= max_images:
            break

    (output_dir / "README.txt").write_text(
        "Bootstrap subset extracted locally from data/fine_tuning/cctv_person.\n",
        encoding="utf-8",
    )


def _extract_mot20_subset(source_dir: Path, output_dir: Path, *, max_frames_per_sequence: int) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        split_dir = source_dir / split
        if not split_dir.exists():
            continue
        for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir())[:2]:
            target_sequence = output_dir / sequence_dir.name
            (target_sequence / "img1").mkdir(parents=True, exist_ok=True)
            gt_dir = sequence_dir / "gt"
            if gt_dir.exists():
                (target_sequence / "gt").mkdir(parents=True, exist_ok=True)
                gt_path = gt_dir / "gt.txt"
                if gt_path.exists():
                    lines = gt_path.read_text(encoding="utf-8").splitlines()
                    kept = [line for line in lines if int(line.split(",")[0]) <= max_frames_per_sequence]
                    (target_sequence / "gt" / "gt.txt").write_text("\n".join(kept), encoding="utf-8")
            seqinfo_path = sequence_dir / "seqinfo.ini"
            if seqinfo_path.exists():
                shutil.copy2(seqinfo_path, target_sequence / "seqinfo.ini")
            for image_path in sorted((sequence_dir / "img1").glob("*.jpg"))[:max_frames_per_sequence]:
                shutil.copy2(image_path, target_sequence / "img1" / image_path.name)

    (output_dir / "README.txt").write_text(
        "Bootstrap subset extracted locally from data/fine_tuning/mot20.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
