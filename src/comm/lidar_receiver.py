from __future__ import annotations

import socket
from dataclasses import dataclass, field
from queue import Queue

from .protocol import MsgType, decode_packet, read_packet, unpack_lidar_scan
from .runtime import MultiClientTCPServer


def decode_lidar_packet(raw: bytes) -> dict[str, object]:
    packet = decode_packet(raw)
    if packet.msg_type != MsgType.LIDAR_SCAN:
        raise ValueError("unexpected message type")
    return unpack_lidar_scan(packet.payload)


@dataclass(slots=True)
class LidarReceiverServer:
    host: str = "0.0.0.0"
    port: int = 9090
    backlog: int = 1

    def receive_once(self) -> dict[str, object]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(self.backlog)
            conn, _ = server.accept()
            with conn:
                packet = read_packet(conn.makefile("rb"))
                if packet.msg_type != MsgType.LIDAR_SCAN:
                    raise ValueError("unexpected message type")
                return unpack_lidar_scan(packet.payload)


@dataclass(slots=True)
class LidarReceiverDaemon:
    host: str = "0.0.0.0"
    port: int = 9090
    backlog: int = 8
    messages: Queue[dict[str, object]] = field(default_factory=Queue, init=False)
    _server: MultiClientTCPServer = field(init=False)

    def __post_init__(self) -> None:
        self._server = MultiClientTCPServer(self.host, self.port, self._handle_packet, backlog=self.backlog)

    def _handle_packet(self, packet, _conn: socket.socket) -> None:
        if packet.msg_type != MsgType.LIDAR_SCAN:
            return
        self.messages.put(unpack_lidar_scan(packet.payload))

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.stop()
