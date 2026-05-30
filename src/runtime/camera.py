from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AstraSCameraConfig:
    backend: str = "openni"
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
        backend = self.config.backend.strip().lower()
        if backend == "openni":
            self._assert_openni_available()
            return

        self._assert_video_devices_available()

    def _assert_video_devices_available(self) -> None:
        missing = [device for device in (self.config.rgb_device, self.config.depth_device) if not Path(device).exists()]
        if missing:
            raise CameraUnavailableError(f"AstraS camera device(s) not available: {', '.join(missing)}")

    def _assert_openni_available(self) -> None:
        usb_bus = Path("/dev/bus/usb")
        if not usb_bus.exists():
            raise CameraUnavailableError("OpenNI backend requires /dev/bus/usb to be mounted into the container")

        sdk_root = Path(os.environ.get("OPENNI_SDK_ROOT", "/opt/orbbec/openni"))
        candidate_libs = (
            sdk_root / "lib" / "libOpenNI2.so",
            sdk_root / "Redist" / "libOpenNI2.so",
            Path("/usr/lib/libOpenNI2.so"),
            Path("/usr/local/lib/libOpenNI2.so"),
        )
        library_path = next((path for path in candidate_libs if path.exists()), None)
        if library_path is None:
            raise CameraUnavailableError(
                "OpenNI backend selected, but libOpenNI2.so was not found in the container runtime"
            )

        try:
            ctypes.CDLL(str(library_path))
        except OSError as exc:
            raise CameraUnavailableError(f"failed to load OpenNI runtime library: {exc}") from exc
