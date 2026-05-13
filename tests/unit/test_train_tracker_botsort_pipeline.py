from __future__ import annotations

from pathlib import Path

from pipelines.train_tracker_botsort import build_tracker, iter_hallway_sequences, iter_mot20_sequences, tune_botsort


def _write_mot_sequence(root: Path, *, name: str, frames: int = 8) -> Path:
    sequence_dir = root / name
    (sequence_dir / "img1").mkdir(parents=True, exist_ok=True)
    (sequence_dir / "det").mkdir(parents=True, exist_ok=True)
    (sequence_dir / "gt").mkdir(parents=True, exist_ok=True)
    (sequence_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=25\nseqLength={frames}\nimWidth=640\nimHeight=480\nimExt=.jpg\n",
        encoding="utf-8",
    )
    for frame_id in range(1, frames + 1):
        (sequence_dir / "img1" / f"{frame_id:06d}.jpg").write_bytes(b"")
    det_lines = [f"{frame_id},-1,100,120,80,160,0.9,-1,-1,-1" for frame_id in range(1, frames + 1)]
    gt_lines = [f"{frame_id},1,102,122,80,160,1,1,0.9" for frame_id in range(1, frames + 1)]
    (sequence_dir / "det" / "det.txt").write_text("\n".join(det_lines), encoding="utf-8")
    (sequence_dir / "gt" / "gt.txt").write_text("\n".join(gt_lines), encoding="utf-8")
    return sequence_dir


def _write_hallway_sequence(
    root: Path,
    *,
    scene_name: str = "epfl_corridor",
    clip_name: str = "clip_01",
    frames: int = 8,
) -> Path:
    scene_dir = root / scene_name
    clip_dir = scene_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in range(frames):
        (clip_dir / f"rgb{frame_id:06d}.png").write_bytes(b"")
        (clip_dir / f"depth{frame_id:06d}.png").write_bytes(b"")
    gt = {clip_name: {frame_id: {0: [100, 120, 80, 160]} for frame_id in range(frames)}}
    import yaml

    (scene_dir / "ground_truth_image_plane.yaml").write_text(yaml.safe_dump(gt), encoding="utf-8")
    return scene_dir


def test_iter_mot20_sequences_finds_valid_sequences(tmp_path) -> None:
    train_root = tmp_path / "mot20" / "train"
    _write_mot_sequence(train_root, name="MOT20-01")

    sequences = list(iter_mot20_sequences(tmp_path / "mot20"))

    assert len(sequences) == 1
    assert sequences[0]["name"] == "MOT20-01"


def test_tune_botsort_writes_config(tmp_path) -> None:
    train_root = tmp_path / "mot20" / "train"
    _write_mot_sequence(train_root, name="MOT20-01")
    _write_hallway_sequence(tmp_path / "hallway_rgbdt")
    result = tune_botsort(
        config=type(
            "Config",
            (),
            {
                "mot20_dir": tmp_path / "mot20",
                "hallway_rgbdt_dir": tmp_path / "hallway_rgbdt",
                "output_path": tmp_path / "botsort_tuned.json",
                "max_mot20_sequences": 1,
                "max_hallway_sequences": 1,
                "max_frames_per_sequence": 8,
            },
        )()
    )

    assert result.best_config_path.exists()
    assert result.best_score > 0.0
    assert result.evaluated_sequences == 2


def test_build_tracker_loads_saved_config(tmp_path) -> None:
    train_root = tmp_path / "mot20" / "train"
    _write_mot_sequence(train_root, name="MOT20-01")
    _write_hallway_sequence(tmp_path / "hallway_rgbdt")
    result = tune_botsort(
        config=type(
            "Config",
            (),
            {
                "mot20_dir": tmp_path / "mot20",
                "hallway_rgbdt_dir": tmp_path / "hallway_rgbdt",
                "output_path": tmp_path / "botsort_tuned.json",
                "max_mot20_sequences": 1,
                "max_hallway_sequences": 1,
                "max_frames_per_sequence": 8,
            },
        )()
    )

    tracker = build_tracker(result.best_config_path)

    assert tracker.iou_threshold > 0.0
    assert tracker.max_missed_frames >= 2


def test_iter_hallway_sequences_finds_valid_sequences(tmp_path) -> None:
    _write_hallway_sequence(tmp_path / "hallway_rgbdt")

    sequences = list(iter_hallway_sequences(tmp_path / "hallway_rgbdt"))

    assert len(sequences) == 1
    assert sequences[0]["source"] == "hallway_rgbdt"
