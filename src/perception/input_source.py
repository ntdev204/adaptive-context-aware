"""input_source.py — Unified frame-provider abstraction.

Supports three input modes:
  - image  : single image file (yields one frame then stops)
  - video  : video file (yields frames sequentially)
  - camera : live webcam capture

Usage:
    source = InputSource.from_image("photo.jpg")
    source = InputSource.from_video("clip.mp4")
    source = InputSource.from_camera(device_index=0)

    with source:
        for frame_bgr, frame_id, timestamp_us in source:
            ...
"""

from __future__ import annotations

import time
from enum import Enum, auto
from pathlib import Path
from typing import Generator, Iterator

import cv2
import numpy as np

from src.utils.constants import FRAME_HEIGHT, FRAME_WIDTH


class InputMode(Enum):
    IMAGE = auto()
    VIDEO = auto()
    CAMERA = auto()


# Frame tuple: (BGR ndarray, frame_id, timestamp_us)
FrameTuple = tuple[np.ndarray, int, int]


def _resize_to_target(frame: np.ndarray) -> np.ndarray:
    """Resize frame to the model's expected input resolution (640×480)."""
    if frame.shape[:2] == (FRAME_HEIGHT, FRAME_WIDTH):
        return frame
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)


class InputSource:
    """Unified iterable frame source for image, video, or camera input."""

    def __init__(self, mode: InputMode, source: str | int) -> None:
        self.mode = mode
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self._image: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_image(cls, path: str | Path) -> "InputSource":
        """Load a single image file (JPEG, PNG, BMP, …)."""
        return cls(InputMode.IMAGE, str(path))

    @classmethod
    def from_video(cls, path: str | Path) -> "InputSource":
        """Load a video file (MP4, AVI, MOV, …)."""
        return cls(InputMode.VIDEO, str(path))

    @classmethod
    def from_camera(cls, device_index: int = 0) -> "InputSource":
        """Open a live webcam by device index."""
        return cls(InputMode.CAMERA, device_index)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "InputSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # open / close
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self.mode == InputMode.IMAGE:
            img = cv2.imread(str(self._source))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {self._source}")
            self._image = _resize_to_target(img)

        elif self.mode in (InputMode.VIDEO, InputMode.CAMERA):
            self._cap = cv2.VideoCapture(self._source)
            if not self._cap.isOpened():
                raise OSError(f"Cannot open {'camera' if self.mode == InputMode.CAMERA else 'video'}: {self._source}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._image = None

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[FrameTuple]:
        return self._frame_generator()

    def _frame_generator(self) -> Generator[FrameTuple, None, None]:
        if self.mode == InputMode.IMAGE:
            if self._image is None:
                raise RuntimeError("InputSource not opened. Use `open()` or a `with` block.")
            yield self._image, 0, _now_us()
            return

        if self._cap is None:
            raise RuntimeError("InputSource not opened. Use `open()` or a `with` block.")

        frame_id = 0
        while True:
            ok, raw = self._cap.read()
            if not ok:
                break
            yield _resize_to_target(raw), frame_id, _now_us()
            frame_id += 1

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Return reported FPS for video/camera (0 for image mode)."""
        if self._cap is not None:
            return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return 0.0

    @property
    def frame_count(self) -> int:
        """Return total frame count for video (-1 for camera/image)."""
        if self.mode == InputMode.VIDEO and self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    def __repr__(self) -> str:
        return f"InputSource(mode={self.mode.name}, source={self._source!r})"


def _now_us() -> int:
    return int(time.monotonic() * 1_000_000)
