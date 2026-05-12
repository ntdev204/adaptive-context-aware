from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AstraSCameraConfig:
    rgb_device: str = "/dev/video0"
    depth_device: str = "/dev/video1"
    width: int = 640
    height: int = 480
    fps: int = 30


class CameraUnavailableError(RuntimeError):
    pass


class AstraSCameraRuntime:
    def __init__(self, config: AstraSCameraConfig | None = None) -> None:
        self.config = config or AstraSCameraConfig()

    def assert_available(self) -> None:
        missing = [
            device
            for device in (self.config.rgb_device, self.config.depth_device)
            if not Path(device).exists()
        ]
        if missing:
            raise CameraUnavailableError(f"AstraS camera device(s) not available: {', '.join(missing)}")
