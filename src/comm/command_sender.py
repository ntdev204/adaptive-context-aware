from __future__ import annotations

import socket
from dataclasses import dataclass, field
from queue import Queue
from typing import Callable

from .protocol import MsgType, encode_packet, pack_nav_cmd
from .runtime import MultiClientTCPServer, PersistentTCPClient


def build_nav_command_packet(seq: int, vx: float, vy: float, omega: float, cmd_seq: int, flags: int = 0) -> bytes:
    payload = pack_nav_cmd(vx=vx, vy=vy, omega=omega, cmd_seq=cmd_seq, flags=flags)
    return encode_packet(MsgType.NAV_CMD, seq=seq, payload=payload)


@dataclass(slots=True)
class CommandSenderClient:
    host: str
    port: int = 9091
    timeout_s: float = 2.0

    def send_nav_command(self, seq: int, vx: float, vy: float, omega: float, cmd_seq: int, flags: int = 0) -> bytes:
        packet = build_nav_command_packet(seq=seq, vx=vx, vy=vy, omega=omega, cmd_seq=cmd_seq, flags=flags)
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.sendall(packet)
        return packet


@dataclass(slots=True)
class CommandSenderDaemon:
    host: str
    port: int = 9091
    timeout_s: float = 2.0
    client: PersistentTCPClient = field(init=False)

    def __post_init__(self) -> None:
        self.client = PersistentTCPClient(self.host, self.port, timeout_s=self.timeout_s)

    def start(self) -> None:
        self.client.start()

    def stop(self) -> None:
        self.client.stop()

    def send_nav_command(self, seq: int, vx: float, vy: float, omega: float, cmd_seq: int, flags: int = 0) -> bytes:
        packet = build_nav_command_packet(seq=seq, vx=vx, vy=vy, omega=omega, cmd_seq=cmd_seq, flags=flags)
        self.client.send(packet)
        return packet


@dataclass(slots=True)
class NavCommandServer:
    host: str = "0.0.0.0"
    port: int = 9091
    backlog: int = 8
    messages: Queue[dict[str, float | int]] = field(default_factory=Queue, init=False)
    _unpack_nav_cmd: Callable[[bytes], dict[str, float | int]] = field(init=False)
    _server: MultiClientTCPServer = field(init=False)

    def __post_init__(self) -> None:
        from .protocol import unpack_nav_cmd

        self._unpack_nav_cmd = unpack_nav_cmd
        self._server = MultiClientTCPServer(self.host, self.port, self._handle_packet, backlog=self.backlog)

    def _handle_packet(self, packet, _conn: socket.socket) -> None:
        if packet.msg_type != MsgType.NAV_CMD:
            return
        self.messages.put(self._unpack_nav_cmd(packet.payload))

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.stop()
