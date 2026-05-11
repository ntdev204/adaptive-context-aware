from __future__ import annotations

from pathlib import Path

from pipelines.train_detector import (
    build_merged_person_dataset,
    build_person_only_dataset,
    plan_detector_training,
)


def _write_dataset(root: Path, *, name: str, class_names: list[str]) -> Path:
    dataset_dir = root / name
    for split in ("train", "valid", "test"):
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "images" / f"{split}.jpg").write_bytes(b"fake")
        (dataset_dir / split / "labels" / f"{split}.txt").write_text(
            "4 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n",
            encoding="utf-8",
        )
    (dataset_dir / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n",
        encoding="utf-8",
    )
    return dataset_dir


def test_build_person_only_dataset_filters_non_person_classes(tmp_path) -> None:
    dataset_dir = _write_dataset(tmp_path, name="custom_1", class_names=["a", "b", "c", "d", "nguoi", "f"])

    dataset_yaml = build_person_only_dataset(dataset_dir, tmp_path / "out")

    assert dataset_yaml.exists()
    label_text = (dataset_yaml.parent / "train" / "labels" / "train.txt").read_text(encoding="utf-8")
    assert label_text.strip() == "0 0.5 0.5 0.2 0.2"


def test_build_merged_person_dataset_combines_custom_sets(tmp_path) -> None:
    dataset_a = _write_dataset(tmp_path, name="custom_1", class_names=["a", "b", "c", "d", "nguoi", "f"])
    dataset_b = _write_dataset(tmp_path, name="custom_2", class_names=["a", "b", "c", "d", "nguoi", "f"])
    yaml_a = build_person_only_dataset(dataset_a, tmp_path / "converted")
    yaml_b = build_person_only_dataset(dataset_b, tmp_path / "converted")

    merged_yaml = build_merged_person_dataset([yaml_a, yaml_b], tmp_path / "merged")

    train_images = list((merged_yaml.parent / "train" / "images").glob("*.jpg"))
    assert len(train_images) == 2
    assert merged_yaml.exists()


def test_plan_detector_training_generates_expected_outputs(tmp_path) -> None:
    pretrain_dataset = _write_dataset(tmp_path, name="cctv_person", class_names=["person"]) / "data.yaml"
    custom_1 = _write_dataset(tmp_path, name="custom_1", class_names=["a", "b", "c", "d", "nguoi", "f"])
    custom_2 = _write_dataset(tmp_path, name="custom_2", class_names=["a", "b", "c", "d", "nguoi", "f"])

    plan = plan_detector_training(
        config=type(
            "Config",
            (),
            {
                "base_model": "yolo11s.pt",
                "pretrain_dataset": pretrain_dataset,
                "custom_datasets": (custom_1, custom_2),
                "work_dir": tmp_path / "artifacts",
                "image_size": 640,
                "pretrain_epochs": 1,
                "finetune_epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "workers": 0,
                "project_name": "detector",
                "run_name_pretrain": "pretrain",
                "run_name_finetune": "finetune",
                "person_class_name": "nguoi",
            },
        )()
    )

    assert plan.merged_dataset_yaml.exists()
    assert "best.pt" in str(plan.pretrain_weights)
    assert "best.pt" in str(plan.finetune_weights)
