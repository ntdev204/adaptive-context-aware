from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import BinaryIO, Iterable

MAGIC = b"\xca\xfe"
HEADER_FORMAT = "!2s B I Q I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
CRC_FORMAT = "!H"
CRC_SIZE = struct.calcsize(CRC_FORMAT)

NAV_CMD_FORMAT = "!f f f I H H"
ESTOP_FORMAT = "!B B H"
SOH_FORMAT = "!f f f f f B B H I I"
HEARTBEAT_FORMAT = "!B f B H"
STATUS_FORMAT = "!B B H"
ACK_FORMAT = "!B I B H"


class MsgType(IntEnum):
    LIDAR_SCAN = 0x01
    NAV_CMD = 0x02
    HEARTBEAT = 0x03
    ESTOP = 0x04
    SOH_TELEMETRY = 0x05
    ACK = 0x06
    STATUS_UPDATE = 0x07


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


@dataclass(slots=True)
class Packet:
    msg_type: MsgType
    seq: int
    timestamp: int
    payload: bytes


def now_micros() -> int:
    return int(time.time() * 1_000_000)


def encode_packet(msg_type: MsgType, seq: int, payload: bytes, timestamp: int | None = None) -> bytes:
    packet_timestamp = now_micros() if timestamp is None else timestamp
    header = struct.pack(HEADER_FORMAT, MAGIC, int(msg_type), seq, packet_timestamp, len(payload))
    crc = crc16_ccitt(header + payload)
    return header + payload + struct.pack(CRC_FORMAT, crc)


def decode_packet(raw: bytes) -> Packet:
    if len(raw) < HEADER_SIZE + CRC_SIZE:
        raise ValueError("packet too short")
    magic, msg_type_raw, seq, timestamp, payload_len = struct.unpack(HEADER_FORMAT, raw[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError("invalid magic")
    expected_size = HEADER_SIZE + payload_len + CRC_SIZE
    if len(raw) != expected_size:
        raise ValueError("packet length mismatch")
    payload = raw[HEADER_SIZE : HEADER_SIZE + payload_len]
    (crc_received,) = struct.unpack(CRC_FORMAT, raw[-CRC_SIZE:])
    crc_expected = crc16_ccitt(raw[:-CRC_SIZE])
    if crc_received != crc_expected:
        raise ValueError("crc mismatch")
    return Packet(msg_type=MsgType(msg_type_raw), seq=seq, timestamp=timestamp, payload=payload)


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise ConnectionError("stream closed before expected bytes were received")
        chunks.extend(chunk)
    return bytes(chunks)


def read_packet(stream: BinaryIO) -> Packet:
    header = read_exact(stream, HEADER_SIZE)
    _, _, _, _, payload_len = struct.unpack(HEADER_FORMAT, header)
    payload_and_crc = read_exact(stream, payload_len + CRC_SIZE)
    return decode_packet(header + payload_and_crc)


def pack_nav_cmd(vx: float, vy: float, omega: float, cmd_seq: int, flags: int = 0, reserved: int = 0) -> bytes:
    return struct.pack(NAV_CMD_FORMAT, vx, vy, omega, cmd_seq, flags, reserved)


def unpack_nav_cmd(payload: bytes) -> dict[str, float | int]:
    vx, vy, omega, cmd_seq, flags, reserved = struct.unpack(NAV_CMD_FORMAT, payload)
    return {"vx": vx, "vy": vy, "omega": omega, "cmd_seq": cmd_seq, "flags": flags, "reserved": reserved}


def pack_estop(reason: int, source: int, reserved: int = 0) -> bytes:
    return struct.pack(ESTOP_FORMAT, reason, source, reserved)


def unpack_estop(payload: bytes) -> dict[str, int]:
    reason, source, reserved = struct.unpack(ESTOP_FORMAT, payload)
    return {"reason": reason, "source": source, "reserved": reserved}


def pack_heartbeat(state: int, pipeline_fps: float, gpu_temp_c: int, reserved: int = 0) -> bytes:
    return struct.pack(HEARTBEAT_FORMAT, state, pipeline_fps, gpu_temp_c, reserved)


def unpack_heartbeat(payload: bytes) -> dict[str, float | int]:
    state, pipeline_fps, gpu_temp_c, reserved = struct.unpack(HEARTBEAT_FORMAT, payload)
    return {"state": state, "pipeline_fps": pipeline_fps, "gpu_temp_c": gpu_temp_c, "reserved": reserved}


def pack_soh(
    cpu_temp_c: float,
    cpu_util_pct: float,
    ram_used_mb: float,
    battery_v: float,
    motor_current_a: float,
    lidar_ok: int,
    motor_ok: int,
    reserved: int,
    uptime_s: int,
    reserved2: int = 0,
) -> bytes:
    return struct.pack(
        SOH_FORMAT,
        cpu_temp_c,
        cpu_util_pct,
        ram_used_mb,
        battery_v,
        motor_current_a,
        lidar_ok,
        motor_ok,
        reserved,
        uptime_s,
        reserved2,
    )


def unpack_soh(payload: bytes) -> dict[str, float | int]:
    values = struct.unpack(SOH_FORMAT, payload)
    keys = [
        "cpu_temp_c",
        "cpu_util_pct",
        "ram_used_mb",
        "battery_v",
        "motor_current_a",
        "lidar_ok",
        "motor_ok",
        "reserved",
        "uptime_s",
        "reserved2",
    ]
    return dict(zip(keys, values, strict=True))


def pack_status_update(new_state: int, reason: int, reserved: int = 0) -> bytes:
    return struct.pack(STATUS_FORMAT, new_state, reason, reserved)


def unpack_status_update(payload: bytes) -> dict[str, int]:
    new_state, reason, reserved = struct.unpack(STATUS_FORMAT, payload)
    return {"new_state": new_state, "reason": reason, "reserved": reserved}


def pack_ack(ack_msg_type: int, ack_seq: int, status: int, reserved: int = 0) -> bytes:
    return struct.pack(ACK_FORMAT, ack_msg_type, ack_seq, status, reserved)


def unpack_ack(payload: bytes) -> dict[str, int]:
    ack_msg_type, ack_seq, status, reserved = struct.unpack(ACK_FORMAT, payload)
    return {"ack_msg_type": ack_msg_type, "ack_seq": ack_seq, "status": status, "reserved": reserved}


def pack_lidar_scan(points: Iterable[tuple[float, float]]) -> bytes:
    point_list = list(points)
    payload = struct.pack("!I", len(point_list))
    for angle_rad, distance_m in point_list:
        payload += struct.pack("!f f", angle_rad, distance_m)
    return payload


def unpack_lidar_scan(payload: bytes) -> dict[str, object]:
    if len(payload) < 4:
        raise ValueError("lidar payload too short")
    (num_points,) = struct.unpack("!I", payload[:4])
    expected = 4 + num_points * 8
    if len(payload) != expected:
        raise ValueError("lidar payload length mismatch")
    points: list[tuple[float, float]] = []
    offset = 4
    for _ in range(num_points):
        angle_rad, distance_m = struct.unpack("!f f", payload[offset : offset + 8])
        points.append((angle_rad, distance_m))
        offset += 8
    return {"num_points": num_points, "points": points}
