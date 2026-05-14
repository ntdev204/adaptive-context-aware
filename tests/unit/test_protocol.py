from __future__ import annotations

import math

import pytest

from src.comm.protocol import (
    MsgType,
    decode_packet,
    encode_packet,
    pack_heartbeat,
    pack_lidar_scan,
    pack_nav_cmd,
    pack_soh,
    unpack_heartbeat,
    unpack_lidar_scan,
    unpack_nav_cmd,
    unpack_soh,
)


def test_nav_cmd_roundtrip() -> None:
    payload = pack_nav_cmd(0.5, -0.25, 1.0, cmd_seq=7, flags=3)
    packet = encode_packet(MsgType.NAV_CMD, seq=11, payload=payload, timestamp=123456)
    decoded = decode_packet(packet)
    assert decoded.msg_type == MsgType.NAV_CMD
    assert decoded.seq == 11
    assert decoded.timestamp == 123456
    values = unpack_nav_cmd(decoded.payload)
    assert values["cmd_seq"] == 7
    assert math.isclose(values["vx"], 0.5, rel_tol=1e-6)
    assert math.isclose(values["vy"], -0.25, rel_tol=1e-6)
    assert math.isclose(values["omega"], 1.0, rel_tol=1e-6)


def test_lidar_roundtrip() -> None:
    payload = pack_lidar_scan([(0.0, 1.2), (1.57, 2.4)])
    packet = encode_packet(MsgType.LIDAR_SCAN, seq=1, payload=payload)
    decoded = decode_packet(packet)
    values = unpack_lidar_scan(decoded.payload)
    assert values["num_points"] == 2
    points = values["points"]
    assert points[0] == pytest.approx((0.0, 1.2))
    assert points[1] == pytest.approx((1.57, 2.4))


def test_crc_rejects_corruption() -> None:
    packet = bytearray(encode_packet(MsgType.HEARTBEAT, seq=1, payload=pack_heartbeat(0, 21.0, 64)))
    packet[10] ^= 0xFF
    with pytest.raises(ValueError, match="crc mismatch"):
        decode_packet(bytes(packet))


def test_heartbeat_roundtrip() -> None:
    payload = pack_heartbeat(1, 28.5, 70)
    values = unpack_heartbeat(payload)
    assert values == {"state": 1, "pipeline_fps": pytest.approx(28.5), "gpu_temp_c": 70, "reserved": 0}


def test_soh_roundtrip() -> None:
    payload = pack_soh(55.0, 12.0, 1024.0, 24.0, 3.0, 1, 1, 0, 99)
    values = unpack_soh(payload)
    assert values["lidar_ok"] == 1
    assert values["motor_ok"] == 1
    assert values["uptime_s"] == 99
