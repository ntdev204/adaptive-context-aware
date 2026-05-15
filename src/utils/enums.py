from __future__ import annotations

from enum import IntEnum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 on Jetson
    from strenum import StrEnum


class Activity(StrEnum):
    """String-based activity labels for serialization.

    .. deprecated::
        Prefer :class:`src.decision.intent_predictor.ActivityClass` (IntEnum)
        for inference code. This StrEnum remains for config / logging only.
    """

    WALKING = "WALKING"
    RUNNING = "RUNNING"
    STANDING = "STANDING"
    SITTING = "SITTING"
    INTERACTING = "INTERACTING"
    FALLING = "FALLING"
    LOITERING = "LOITERING"
    FIGHTING = "FIGHTING"
    OTHER = "OTHER"


class SceneContext(StrEnum):
    CORRIDOR = "CORRIDOR"
    LOBBY = "LOBBY"
    GATE_AREA = "GATE_AREA"
    RESTAURANT = "RESTAURANT"
    OPEN_SPACE = "OPEN_SPACE"
    UNKNOWN = "UNKNOWN"


class IntentDirection(StrEnum):
    """String-based direction labels for serialization.

    .. deprecated::
        Prefer :class:`src.decision.intent_predictor.IntentDirection` (IntEnum)
        for inference code. This StrEnum remains for config / logging only.
    """

    NORTH = "NORTH"
    NE = "NE"
    EAST = "EAST"
    SE = "SE"
    SOUTH = "SOUTH"
    SW = "SW"
    WEST = "WEST"
    NW = "NW"
    STATIONARY = "STATIONARY"


class SafetyState(IntEnum):
    NORMAL = 0
    DEGRADED = 1
    ESTOP = 2
    RECOVERY = 3


class StatusChangeReason(IntEnum):
    GPU_OVERHEAT = 0x01
    CAMERA_FAIL = 0x02
    SOH_LOW = 0x03
    CONDITION_RESOLVED = 0x04
    SELF_TEST_PASS = 0x05
    SELF_TEST_FAIL = 0x06


class EStopReason(IntEnum):
    HEARTBEAT_TIMEOUT = 0x01
    ANOMALY_CRITICAL = 0x02
    MANUAL_BUTTON = 0x03
    SYSTEM_FAULT = 0x04
    THERMAL_CRITICAL = 0x05


class EStopSource(IntEnum):
    JETSON_AI = 0x01
    RPI_WATCHDOG = 0x02
    HARDWARE = 0x03
    OPERATOR = 0x04
