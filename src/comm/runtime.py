from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable

from .protocol import (
    MsgType,
    Packet,
    encode_packet,
    pack_ack,
    read_packet,
    unpack_ack,
)

PacketHandler = Callable[[Packet, socket.socket], None]


class StoppableService:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def stop(self) -> None:
        self._stop_event.set()
        self._on_stop()
        for thread in self._threads:
            thread.join(timeout=2.0)

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def _register(self, thread: threading.Thread) -> None:
        self._threads.append(thread)

    def _on_stop(self) -> None:
        return None


@dataclass(slots=True)
class AckResult:
    ack_msg_type: int
    ack_seq: int
    status: int
    reserved: int


class AckTracker:
    def __init__(self) -> None:
        self._pending: dict[tuple[int, int], Queue[AckResult]] = {}
        self._lock = threading.Lock()

    def register(self, msg_type: int, seq: int) -> Queue[AckResult]:
        queue: Queue[AckResult] = Queue(maxsize=1)
        with self._lock:
            self._pending[(msg_type, seq)] = queue
        return queue

    def resolve(self, ack: AckResult) -> None:
        with self._lock:
            queue = self._pending.pop((ack.ack_msg_type, ack.ack_seq), None)
        if queue is not None:
            queue.put(ack)


def send_ack(sock: socket.socket, seq: int, ack_msg_type: MsgType, ack_seq: int, status: int = 0) -> None:
    payload = pack_ack(int(ack_msg_type), ack_seq, status)
    sock.sendall(encode_packet(MsgType.ACK, seq=seq, payload=payload))


def parse_ack(packet: Packet) -> AckResult:
    if packet.msg_type != MsgType.ACK:
        raise ValueError("packet is not an ACK")
    payload = unpack_ack(packet.payload)
    return AckResult(
        ack_msg_type=int(payload["ack_msg_type"]),
        ack_seq=int(payload["ack_seq"]),
        status=int(payload["status"]),
        reserved=int(payload["reserved"]),
    )


class MultiClientTCPServer(StoppableService):
    def __init__(self, host: str, port: int, handler: PacketHandler, backlog: int = 8) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.backlog = backlog
        self.handler = handler
        self._server_socket: socket.socket | None = None

    def start(self) -> None:
        thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._register(thread)
        thread.start()

    def _serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self._server_socket = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(self.backlog)
            server.settimeout(0.2)
            while not self.stopping:
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                self._register(thread)
                thread.start()

    def _handle_client(self, conn: socket.socket) -> None:
        with conn:
            stream = conn.makefile("rb")
            while not self.stopping:
                try:
                    packet = read_packet(stream)
                except (ConnectionError, OSError):
                    break
                self.handler(packet, conn)

    def stop(self) -> None:
        super().stop()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass


class PersistentTCPClient(StoppableService):
    def __init__(
        self,
        host: str,
        port: int,
        on_packet: Callable[[Packet], None] | None = None,
        timeout_s: float = 2.0,
        backoff_schedule_s: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8, 1.0),
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.on_packet = on_packet
        self.backoff_schedule_s = backoff_schedule_s
        self._outbox: Queue[bytes] = Queue()
        self._socket: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True)
        self._register(thread)
        thread.start()

    def send(self, payload: bytes) -> None:
        self._outbox.put(payload)

    def _run(self) -> None:
        backoff_index = 0
        while not self.stopping:
            try:
                with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
                    sock.settimeout(0.2)
                    self._socket = sock
                    backoff_index = 0
                    self._connected_loop(sock)
            except OSError:
                delay = self.backoff_schedule_s[min(backoff_index, len(self.backoff_schedule_s) - 1)]
                backoff_index += 1
                time.sleep(delay)
            finally:
                self._socket = None

    def _connected_loop(self, sock: socket.socket) -> None:
        if self.on_packet is not None:
            self._reader_thread = threading.Thread(target=self._read_loop, args=(sock,), daemon=True)
            self._reader_thread.start()
        while not self.stopping:
            try:
                payload = self._outbox.get(timeout=0.05)
                sock.sendall(payload)
            except Empty:
                pass
            except OSError:
                break

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.5)
            self._reader_thread = None

    def _read_loop(self, sock: socket.socket) -> None:
        stream = sock.makefile("rb")
        while not self.stopping:
            try:
                packet = read_packet(stream)
            except socket.timeout:
                continue
            except (ConnectionError, OSError):
                break
            if self.on_packet is not None:
                self.on_packet(packet)

    def _on_stop(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
