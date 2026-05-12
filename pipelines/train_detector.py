from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class DetectorTrainingConfig:
    base_model: str = "yolo11s.pt"
    pretrain_dataset: Path = Path("data/fine_tuning/cctv_person/data.yaml")
    custom_datasets: tuple[Path, ...] = ()
    work_dir: Path = Path("artifacts/detector")
    image_size: int = 640
    pretrain_epochs: int = 30
    finetune_epochs: int = 50
    batch_size: int = 16
    device: str = "cpu"
    workers: int = 2
    project_name: str = "detector"
    run_name_pretrain: str = "pretrain-cctv-person"
    run_name_finetune: str = "finetune-school-multiclass"


@dataclass(frozen=True, slots=True)
class DetectorTrainingPlan:
    pretrain_weights: Path
    finetune_weights: Path | None
    merged_dataset_yaml: Path | None

    @property
    def final_weights(self) -> Path:
        return self.finetune_weights or self.pretrain_weights


DEFAULT_DETECTOR_TRAINING_CONFIG = DetectorTrainingConfig()


def load_yolo_dataset_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def build_person_only_dataset(
    source_dataset_dir: str | Path,
    output_root: str | Path,
    *,
    person_class_name: str = "nguoi",
) -> Path:
    source_dir = Path(source_dataset_dir)
    output_dir = Path(output_root) / source_dir.name
    config = load_yolo_dataset_config(source_dir / "data.yaml")
    names = list(config["names"])
    if person_class_name not in names:
        raise ValueError(f"{person_class_name!r} not found in {source_dir / 'data.yaml'}")
    person_class_index = names.index(person_class_name)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        image_src = source_dir / split / "images"
        label_src = source_dir / split / "labels"
        image_dst = output_dir / split / "images"
        label_dst = output_dir / split / "labels"
        image_dst.mkdir(parents=True, exist_ok=True)
        label_dst.mkdir(parents=True, exist_ok=True)

        if image_src.exists():
            for image_path in image_src.iterdir():
                if image_path.is_file():
                    shutil.copy2(image_path, image_dst / image_path.name)

        if label_src.exists():
            for label_path in label_src.glob("*.txt"):
                converted_lines = _filter_person_only_annotations(label_path, person_class_index)
                (label_dst / label_path.name).write_text("\n".join(converted_lines), encoding="utf-8")

    dataset_yaml = output_dir / "data.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": 1,
                "names": ["person"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def build_merged_dataset(dataset_yamls: list[Path], output_root: str | Path) -> Path:
    if not dataset_yamls:
        raise ValueError("at least one dataset yaml is required")

    output_dir = Path(output_root)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = _load_dataset_names(dataset_yamls[0])
    for dataset_yaml in dataset_yamls[1:]:
        if _load_dataset_names(dataset_yaml) != names:
            raise ValueError("all custom datasets must share the same class ordering")

    for split in ("train", "valid", "test"):
        split_images = output_dir / split / "images"
        split_labels = output_dir / split / "labels"
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        for dataset_yaml in dataset_yamls:
            dataset_dir = dataset_yaml.parent
            source_images = dataset_dir / split / "images"
            source_labels = dataset_dir / split / "labels"
            prefix = dataset_dir.name
            for image_path in source_images.iterdir():
                target_name = f"{prefix}-{image_path.name}"
                shutil.copy2(image_path, split_images / target_name)
            for label_path in source_labels.glob("*.txt"):
                target_name = f"{prefix}-{label_path.name}"
                shutil.copy2(label_path, split_labels / target_name)

    merged_yaml = output_dir / "data.yaml"
    merged_yaml.write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return merged_yaml


def plan_detector_training(config: DetectorTrainingConfig = DEFAULT_DETECTOR_TRAINING_CONFIG) -> DetectorTrainingPlan:
    pretrain_weights = config.work_dir / config.project_name / config.run_name_pretrain / "weights" / "best.pt"
    merged_dataset_yaml = None
    finetune_weights = None
    if config.custom_datasets:
        merged_dataset_yaml = build_merged_dataset(
            [dataset_dir / "data.yaml" for dataset_dir in config.custom_datasets],
            config.work_dir / "merged_custom_dataset",
        )
        finetune_weights = config.work_dir / config.project_name / config.run_name_finetune / "weights" / "best.pt"

    return DetectorTrainingPlan(
        pretrain_weights=pretrain_weights,
        finetune_weights=finetune_weights,
        merged_dataset_yaml=merged_dataset_yaml,
    )


def train_detector(config: DetectorTrainingConfig = DEFAULT_DETECTOR_TRAINING_CONFIG) -> DetectorTrainingPlan:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to train the YOLO11 detector") from exc

    plan = plan_detector_training(config)
    model = YOLO(config.base_model)
    pretrain_results = model.train(
        data=str(config.pretrain_dataset),
        epochs=config.pretrain_epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        device=config.device,
        workers=config.workers,
        project=str(config.work_dir / config.project_name),
        name=config.run_name_pretrain,
    )
    pretrain_weights = _resolve_best_weights(pretrain_results, fallback=plan.pretrain_weights)

    if plan.merged_dataset_yaml is not None:
        finetune_model = YOLO(str(pretrain_weights))
        finetune_results = finetune_model.train(
            data=str(plan.merged_dataset_yaml),
            epochs=config.finetune_epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            device=config.device,
            workers=config.workers,
            project=str(config.work_dir / config.project_name),
            name=config.run_name_finetune,
        )
        finetune_weights = _resolve_best_weights(finetune_results, fallback=plan.finetune_weights)
        return DetectorTrainingPlan(
            pretrain_weights=pretrain_weights,
            finetune_weights=finetune_weights,
            merged_dataset_yaml=plan.merged_dataset_yaml,
        )

    return DetectorTrainingPlan(
        pretrain_weights=pretrain_weights,
        finetune_weights=None,
        merged_dataset_yaml=None,
    )


def _filter_person_only_annotations(label_path: Path, person_class_index: int) -> list[str]:
    converted_lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        class_id = int(float(parts[0]))
        if class_id != person_class_index:
            continue
        converted_lines.append(" ".join(["0", *parts[1:]]))
    return converted_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the YOLO11 detector from the CCTV person dataset.")
    parser.add_argument("--base-model", default=DEFAULT_DETECTOR_TRAINING_CONFIG.base_model)
    parser.add_argument("--pretrain-dataset", type=Path, default=DEFAULT_DETECTOR_TRAINING_CONFIG.pretrain_dataset)
    parser.add_argument("--custom-dataset", action="append", type=Path, dest="custom_datasets")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_DETECTOR_TRAINING_CONFIG.work_dir)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_DETECTOR_TRAINING_CONFIG.image_size)
    parser.add_argument("--pretrain-epochs", type=int, default=DEFAULT_DETECTOR_TRAINING_CONFIG.pretrain_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_DETECTOR_TRAINING_CONFIG.finetune_epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_DETECTOR_TRAINING_CONFIG.batch_size)
    parser.add_argument("--device", default=DEFAULT_DETECTOR_TRAINING_CONFIG.device)
    parser.add_argument("--workers", type=int, default=DEFAULT_DETECTOR_TRAINING_CONFIG.workers)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    config = DetectorTrainingConfig(
        base_model=args.base_model,
        pretrain_dataset=args.pretrain_dataset,
        custom_datasets=tuple(args.custom_datasets or ()),
        work_dir=args.work_dir,
        image_size=args.imgsz,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
    )

    plan = plan_detector_training(config) if args.plan_only else train_detector(config)
    print(f"pretrain_best={plan.pretrain_weights}")
    if plan.finetune_weights is not None:
        print(f"finetune_best={plan.finetune_weights}")
    print(f"detector_best={plan.final_weights}")
    if plan.merged_dataset_yaml is not None:
        print(f"merged_dataset={plan.merged_dataset_yaml}")


def _load_dataset_names(dataset_yaml: Path) -> list[str]:
    payload = load_yolo_dataset_config(dataset_yaml)
    return list(payload["names"])


def _resolve_best_weights(results: object, *, fallback: Path | None) -> Path:
    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        candidate = Path(save_dir) / "weights" / "best.pt"
        if candidate.exists():
            return candidate
    if fallback is None:
        raise FileNotFoundError("unable to resolve best.pt from Ultralytics results")
    return fallback


if __name__ == "__main__":
    main()
