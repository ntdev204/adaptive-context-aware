from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, field
from queue import Queue
from time import monotonic
from typing import Callable

from src.utils.enums import EStopReason, EStopSource, SafetyState

from .protocol import (
    MsgType,
    Packet,
    encode_packet,
    pack_heartbeat,
    pack_soh,
    pack_status_update,
    read_packet,
    unpack_heartbeat,
    unpack_soh,
    unpack_status_update,
)
from .runtime import AckTracker, MultiClientTCPServer, PersistentTCPClient, parse_ack, send_ack


@dataclass(slots=True)
class HeartbeatWatchdog:
    timeout_ms: int = 2000
    check_interval_ms: int = 100
    now_ms: Callable[[], int] | None = None
    on_estop: Callable[[EStopReason, EStopSource], None] | None = None
    last_heartbeat_ms: int | None = field(default=None, init=False)
    state: SafetyState = field(default=SafetyState.NORMAL, init=False)
    estop_triggered: bool = field(default=False, init=False)

    def _clock(self) -> int:
        if self.now_ms is None:
            return int(monotonic() * 1000)
        return self.now_ms()

    def record_heartbeat(self, at_ms: int | None = None) -> None:
        self.last_heartbeat_ms = self._clock() if at_ms is None else at_ms
        if self.state != SafetyState.ESTOP:
            self.state = SafetyState.NORMAL

    def check(self, at_ms: int | None = None) -> bool:
        now = self._clock() if at_ms is None else at_ms
        if self.last_heartbeat_ms is None:
            self.last_heartbeat_ms = now
            return False
        if now - self.last_heartbeat_ms > self.timeout_ms and not self.estop_triggered:
            self.estop_triggered = True
            self.state = SafetyState.ESTOP
            if self.on_estop is not None:
                self.on_estop(EStopReason.HEARTBEAT_TIMEOUT, EStopSource.RPI_WATCHDOG)
            return True
        return False


@dataclass(slots=True)
class HeartbeatClient:
    host: str
    port: int = 9093
    timeout_s: float = 2.0

    def send_once(self, seq: int, state: SafetyState, pipeline_fps: float, gpu_temp_c: int) -> bytes:
        payload = pack_heartbeat(int(state), pipeline_fps, gpu_temp_c)
        packet = encode_packet(MsgType.HEARTBEAT, seq=seq, payload=payload)
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.sendall(packet)
        return packet


@dataclass(slots=True)
class HeartbeatServer:
    host: str = "0.0.0.0"
    port: int = 9093
    backlog: int = 1
    watchdog: HeartbeatWatchdog | None = None

    def receive_once(self) -> dict[str, float | int]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(self.backlog)
            conn, _ = server.accept()
            with conn:
                packet = read_packet(conn.makefile("rb"))
                if packet.msg_type != MsgType.HEARTBEAT:
                    raise ValueError("unexpected message type")
                heartbeat = unpack_heartbeat(packet.payload)
                if self.watchdog is not None:
                    self.watchdog.record_heartbeat()
                return heartbeat


@dataclass(slots=True)
class SoHTelemetryReceiver:
    host: str = "0.0.0.0"
    port: int = 9092
    buffer_size: int = 4096

    def receive_once(self) -> dict[str, float | int]:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.host, self.port))
            raw, _ = sock.recvfrom(self.buffer_size)
        packet = read_packet(_DatagramReader(raw))
        if packet.msg_type != MsgType.SOH_TELEMETRY:
            raise ValueError("unexpected message type")
        return unpack_soh(packet.payload)


@dataclass(slots=True)
class SoHTelemetrySender:
    host: str
    port: int = 9092

    def send_once(
        self,
        seq: int,
        cpu_temp_c: float,
        cpu_util_pct: float,
        ram_used_mb: float,
        battery_v: float,
        motor_current_a: float,
        camera_ok: int,
        motor_ok: int,
        uptime_s: int,
        reserved: int = 0,
        reserved2: int = 0,
    ) -> bytes:
        payload = pack_soh(
            cpu_temp_c=cpu_temp_c,
            cpu_util_pct=cpu_util_pct,
            ram_used_mb=ram_used_mb,
            battery_v=battery_v,
            motor_current_a=motor_current_a,
            camera_ok=camera_ok,
            motor_ok=motor_ok,
            reserved=reserved,
            uptime_s=uptime_s,
            reserved2=reserved2,
        )
        packet = encode_packet(MsgType.SOH_TELEMETRY, seq=seq, payload=payload)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(packet, (self.host, self.port))
        return packet


@dataclass(slots=True)
class SoHTelemetryDaemon:
    host: str
    port: int = 9092
    interval_s: float = 1.0
    payload_factory: Callable[[], dict[str, float | int]] | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        sender = SoHTelemetrySender(self.host, self.port)
        seq = 0
        while not self._stop_event.is_set():
            payload = self.payload_factory() if self.payload_factory is not None else {
                "cpu_temp_c": 0.0,
                "cpu_util_pct": 0.0,
                "ram_used_mb": 0.0,
                "battery_v": 0.0,
                "motor_current_a": 0.0,
                "camera_ok": 1,
                "motor_ok": 1,
                "uptime_s": 0,
            }
            sender.send_once(seq=seq, **payload)
            seq += 1
            self._stop_event.wait(self.interval_s)


@dataclass(slots=True)
class SoHTelemetryReceiverDaemon:
    host: str = "0.0.0.0"
    port: int = 9092
    buffer_size: int = 4096
    messages: Queue[dict[str, float | int]] = field(default_factory=Queue, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.host, self.port))
            sock.settimeout(0.2)
            while not self._stop_event.is_set():
                try:
                    raw, _ = sock.recvfrom(self.buffer_size)
                except socket.timeout:
                    continue
                packet = read_packet(_DatagramReader(raw))
                if packet.msg_type == MsgType.SOH_TELEMETRY:
                    self.messages.put(unpack_soh(packet.payload))


@dataclass(slots=True)
class HeartbeatSessionState:
    heartbeats: Queue[dict[str, float | int]] = field(default_factory=Queue)
    status_updates: Queue[dict[str, int]] = field(default_factory=Queue)
    estops: Queue[dict[str, int]] = field(default_factory=Queue)


@dataclass(slots=True)
class HeartbeatServerDaemon:
    host: str = "0.0.0.0"
    port: int = 9093
    backlog: int = 8
    watchdog: HeartbeatWatchdog | None = None
    state: HeartbeatSessionState = field(default_factory=HeartbeatSessionState, init=False)
    _ack_seq: int = field(default=0, init=False)
    _server: MultiClientTCPServer = field(init=False)
    _watchdog_thread: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)

    def __post_init__(self) -> None:
        self._server = MultiClientTCPServer(self.host, self.port, self._handle_packet, backlog=self.backlog)

    def _handle_packet(self, packet: Packet, conn: socket.socket) -> None:
        from .protocol import unpack_estop

        if packet.msg_type == MsgType.HEARTBEAT:
            heartbeat = unpack_heartbeat(packet.payload)
            self.state.heartbeats.put(heartbeat)
            if self.watchdog is not None:
                self.watchdog.record_heartbeat()
            send_ack(conn, seq=self._next_ack_seq(), ack_msg_type=MsgType.HEARTBEAT, ack_seq=packet.seq)
            return
        if packet.msg_type == MsgType.STATUS_UPDATE:
            self.state.status_updates.put(unpack_status_update(packet.payload))
            send_ack(conn, seq=self._next_ack_seq(), ack_msg_type=MsgType.STATUS_UPDATE, ack_seq=packet.seq)
            return
        if packet.msg_type == MsgType.ESTOP:
            self.state.estops.put(unpack_estop(packet.payload))
            send_ack(conn, seq=self._next_ack_seq(), ack_msg_type=MsgType.ESTOP, ack_seq=packet.seq)

    def _next_ack_seq(self) -> int:
        value = self._ack_seq
        self._ack_seq += 1
        return value

    def start(self) -> None:
        self._server.start()
        if self.watchdog is not None:
            self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)
        self._server.stop()

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.watchdog is not None:
                self.watchdog.check()
                self._stop_event.wait(self.watchdog.check_interval_ms / 1000)


@dataclass(slots=True)
class HeartbeatClientDaemon:
    host: str
    port: int = 9093
    timeout_s: float = 2.0
    interval_s: float = 0.5
    heartbeat_payload_factory: Callable[[], dict[str, float | int]] | None = None
    acks: Queue[dict[str, int]] = field(default_factory=Queue, init=False)
    _ack_tracker: AckTracker = field(default_factory=AckTracker, init=False)
    _client: PersistentTCPClient = field(init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._client = PersistentTCPClient(self.host, self.port, on_packet=self._on_packet, timeout_s=self.timeout_s)

    def start(self) -> None:
        self._client.start()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._client.stop()

    def _on_packet(self, packet: Packet) -> None:
        if packet.msg_type != MsgType.ACK:
            return
        ack = parse_ack(packet)
        self._ack_tracker.resolve(ack)
        self.acks.put(
            {
                "ack_msg_type": ack.ack_msg_type,
                "ack_seq": ack.ack_seq,
                "status": ack.status,
            }
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            payload = self.heartbeat_payload_factory() if self.heartbeat_payload_factory is not None else {
                "state": int(SafetyState.NORMAL),
                "pipeline_fps": 0.0,
                "gpu_temp_c": 0,
            }
            self.send_heartbeat(
                state=SafetyState(int(payload["state"])),
                pipeline_fps=float(payload["pipeline_fps"]),
                gpu_temp_c=int(payload["gpu_temp_c"]),
            )
            self._stop_event.wait(self.interval_s)

    def send_heartbeat(self, state: SafetyState, pipeline_fps: float, gpu_temp_c: int) -> int:
        seq = self._seq
        self._seq += 1
        packet = encode_packet(MsgType.HEARTBEAT, seq=seq, payload=pack_heartbeat(int(state), pipeline_fps, gpu_temp_c))
        self._ack_tracker.register(MsgType.HEARTBEAT, seq)
        self._client.send(packet)
        return seq

    def send_status_update(self, new_state: SafetyState, reason: int) -> int:
        seq = self._seq
        self._seq += 1
        packet = encode_packet(MsgType.STATUS_UPDATE, seq=seq, payload=pack_status_update(int(new_state), reason))
        self._ack_tracker.register(MsgType.STATUS_UPDATE, seq)
        self._client.send(packet)
        return seq

    def send_estop(self, reason: int, source: int) -> int:
        from .protocol import pack_estop

        seq = self._seq
        self._seq += 1
        packet = encode_packet(MsgType.ESTOP, seq=seq, payload=pack_estop(reason, source))
        self._ack_tracker.register(MsgType.ESTOP, seq)
        self._client.send(packet)
        return seq


class _DatagramReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk
