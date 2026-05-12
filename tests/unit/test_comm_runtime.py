from __future__ import annotations

import socket
import threading
import time
from queue import Empty, Queue

import pytest

from src.comm.command_sender import CommandSenderClient, CommandSenderDaemon, NavCommandServer
from src.comm.health_monitor import (
    HeartbeatClient,
    HeartbeatClientDaemon,
    HeartbeatServerDaemon,
    SoHTelemetryDaemon,
    SoHTelemetryReceiverDaemon,
    SoHTelemetrySender,
)
from src.comm.lidar_receiver import LidarReceiverDaemon
from src.comm.protocol import MsgType, encode_packet, pack_lidar_scan, read_packet, unpack_nav_cmd
from src.utils.enums import EStopReason, EStopSource, SafetyState, StatusChangeReason


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_event(event: threading.Event, timeout_s: float = 2.0) -> None:
    if not event.wait(timeout_s):
        raise TimeoutError("server did not become ready")


def test_lidar_server_receives_scan() -> None:
    port = _free_port()
    result: Queue[dict[str, object]] = Queue()
    ready = threading.Event()

    def server() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
            ready.set()
            conn, _ = sock.accept()
            with conn:
                packet = read_packet(conn.makefile("rb"))
                result.put({"num_points": 2, "points": [(0.0, 1.0), (1.0, 2.0)]} if packet else {})

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    _wait_for_event(ready)

    points = [(0.0, 1.0), (1.0, 2.0)]
    packet = encode_packet(MsgType.LIDAR_SCAN, seq=1, payload=pack_lidar_scan(points))
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as client:
        client.sendall(packet)

    thread.join(timeout=2.0)
    received = result.get_nowait()
    assert received["num_points"] == 2
    assert received["points"][0] == pytest.approx((0.0, 1.0))


def test_nav_command_client_sends_packet() -> None:
    port = _free_port()
    result: Queue[dict[str, float | int]] = Queue()
    ready = threading.Event()

    def server() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
            ready.set()
            conn, _ = sock.accept()
            with conn:
                packet = read_packet(conn.makefile("rb"))
                result.put(unpack_nav_cmd(packet.payload))

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    _wait_for_event(ready)

    CommandSenderClient(host="127.0.0.1", port=port).send_nav_command(seq=4, vx=0.5, vy=-0.25, omega=1.0, cmd_seq=7)

    thread.join(timeout=2.0)
    received = result.get_nowait()
    assert received["cmd_seq"] == 7
    assert received["vx"] == pytest.approx(0.5)


def test_soh_udp_sender_receiver() -> None:
    port = _free_port()
    result: Queue[dict[str, float | int]] = Queue()
    ready = threading.Event()

    def receiver() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("127.0.0.1", port))
            ready.set()
            raw, _ = sock.recvfrom(4096)
        from src.comm.protocol import decode_packet, unpack_soh

        result.put(unpack_soh(decode_packet(raw).payload))

    thread = threading.Thread(target=receiver, daemon=True)
    thread.start()
    _wait_for_event(ready)

    sender = SoHTelemetrySender(host="127.0.0.1", port=port)
    sender.send_once(
        seq=2,
        cpu_temp_c=51.0,
        cpu_util_pct=30.0,
        ram_used_mb=256.0,
        battery_v=24.0,
        motor_current_a=2.0,
        lidar_ok=1,
        motor_ok=1,
        uptime_s=42,
    )

    thread.join(timeout=2.0)
    received = result.get_nowait()
    assert received["cpu_temp_c"] == pytest.approx(51.0)
    assert received["uptime_s"] == 42


def test_heartbeat_tcp_client_server() -> None:
    port = _free_port()
    result: Queue[dict[str, float | int]] = Queue()
    ready = threading.Event()

    def server() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
            ready.set()
            conn, _ = sock.accept()
            with conn:
                from src.comm.protocol import unpack_heartbeat

                packet = read_packet(conn.makefile("rb"))
                result.put(unpack_heartbeat(packet.payload))

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    _wait_for_event(ready)

    HeartbeatClient(host="127.0.0.1", port=port).send_once(
        seq=3,
        state=SafetyState.NORMAL,
        pipeline_fps=20.5,
        gpu_temp_c=63,
    )

    thread.join(timeout=2.0)
    received = result.get_nowait()
    assert received["state"] == SafetyState.NORMAL
    assert received["pipeline_fps"] == pytest.approx(20.5)


def test_lidar_daemon_multiplexes_multiple_clients() -> None:
    port = _free_port()
    daemon = LidarReceiverDaemon(host="127.0.0.1", port=port)
    daemon.start()
    time.sleep(0.2)
    try:
        for seq in (1, 2):
            packet = encode_packet(MsgType.LIDAR_SCAN, seq=seq, payload=pack_lidar_scan([(float(seq), float(seq + 1))]))
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as client:
                client.sendall(packet)
        first = daemon.messages.get(timeout=2.0)
        second = daemon.messages.get(timeout=2.0)
        assert first["num_points"] == 1
        assert second["num_points"] == 1
    finally:
        daemon.stop()


def test_nav_command_daemon_reconnects_and_delivers() -> None:
    port = _free_port()
    daemon = CommandSenderDaemon(host="127.0.0.1", port=port)
    daemon.start()
    try:
        time.sleep(0.3)
        server = NavCommandServer(host="127.0.0.1", port=port)
        server.start()
        time.sleep(0.2)
        daemon.send_nav_command(seq=8, vx=0.2, vy=0.1, omega=0.0, cmd_seq=12)
        received = server.messages.get(timeout=2.0)
        assert received["cmd_seq"] == 12
        assert received["vx"] == pytest.approx(0.2)
    finally:
        daemon.stop()
        server.stop()


def test_soh_daemon_streams_periodically() -> None:
    port = _free_port()
    receiver = SoHTelemetryReceiverDaemon(host="127.0.0.1", port=port)
    sender = SoHTelemetryDaemon(
        host="127.0.0.1",
        port=port,
        interval_s=0.1,
        payload_factory=lambda: {
            "cpu_temp_c": 49.0,
            "cpu_util_pct": 10.0,
            "ram_used_mb": 128.0,
            "battery_v": 24.0,
            "motor_current_a": 1.0,
            "lidar_ok": 1,
            "motor_ok": 1,
            "uptime_s": 7,
        },
    )
    receiver.start()
    time.sleep(0.2)
    sender.start()
    try:
        received = receiver.messages.get(timeout=2.0)
        assert received["uptime_s"] == 7
    finally:
        sender.stop()
        receiver.stop()


def test_heartbeat_ack_flow_and_control_messages() -> None:
    port = _free_port()
    server = HeartbeatServerDaemon(host="127.0.0.1", port=port)
    client = HeartbeatClientDaemon(
        host="127.0.0.1",
        port=port,
        interval_s=0.1,
        heartbeat_payload_factory=lambda: {
            "state": int(SafetyState.NORMAL),
            "pipeline_fps": 30.0,
            "gpu_temp_c": 60,
        },
    )
    server.start()
    time.sleep(0.2)
    client.start()
    try:
        heartbeat = server.state.heartbeats.get(timeout=2.0)
        assert heartbeat["pipeline_fps"] == pytest.approx(30.0)

        status_seq = client.send_status_update(SafetyState.DEGRADED, StatusChangeReason.GPU_OVERHEAT)
        status = server.state.status_updates.get(timeout=2.0)
        assert status["new_state"] == SafetyState.DEGRADED

        estop_seq = client.send_estop(EStopReason.ANOMALY_CRITICAL, EStopSource.JETSON_AI)
        estop = server.state.estops.get(timeout=2.0)
        assert estop["reason"] == EStopReason.ANOMALY_CRITICAL

        seen = []
        deadline = time.time() + 2.0
        seen_status = False
        seen_estop = False
        while time.time() < deadline and not (seen_status and seen_estop):
            try:
                item = client.acks.get(timeout=0.2)
                seen.append(item)
                if item["ack_msg_type"] == MsgType.STATUS_UPDATE and item["ack_seq"] == status_seq:
                    seen_status = True
                if item["ack_msg_type"] == MsgType.ESTOP and item["ack_seq"] == estop_seq:
                    seen_estop = True
            except Empty:
                pass
        assert seen_status, seen
        assert seen_estop, seen
    finally:
        client.stop()
        server.stop()
