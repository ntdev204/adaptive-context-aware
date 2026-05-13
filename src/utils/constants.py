"""Project-wide constants shared across subsystems."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Frame resolution (model input / depth map)
# ---------------------------------------------------------------------------
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FRAME_SHAPE_HWC: tuple[int, int, int] = (FRAME_HEIGHT, FRAME_WIDTH, 3)
DEPTH_SHAPE_HW: tuple[int, int] = (FRAME_HEIGHT, FRAME_WIDTH)

# ---------------------------------------------------------------------------
# Unified reasoning output dimension (shared by fusion, intent, anomaly)
# ---------------------------------------------------------------------------
UNIFIED_REASONING_DIM: int = 256
