from __future__ import annotations

from dataclasses import dataclass

from src.utils.enums import EStopReason, EStopSource

from .protocol import MsgType, encode_packet, pack_estop


@dataclass(slots=True)
class EStopClientMessage:
    reason: EStopReason
    source: EStopSource

    def to_packet(self, seq: int) -> bytes:
        return encode_packet(
            MsgType.ESTOP,
            seq=seq,
            payload=pack_estop(int(self.reason), int(self.source)),
        )
