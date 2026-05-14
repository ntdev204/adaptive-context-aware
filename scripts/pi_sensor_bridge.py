from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.transport.messages import ImuSampleMessage, LidarScanMessage, PiStatusMessage
from src.transport.zmq_sensor_client import ZmqSensorClient, ZmqSensorClientConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge real Raspberry Pi sensor JSONL into Jetson ZMQ ingest.")
    parser.add_argument("--jetson-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--source-id", default="raspi-101")
    args = parser.parse_args()

    client = ZmqSensorClient(ZmqSensorClientConfig(jetson_host=args.jetson_host, jetson_port=args.port))
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            client.send(_parse_message(line, args.source_id))
    finally:
        client.close()
    return 0


def _parse_message(line: str, fallback_source_id: str) -> LidarScanMessage | ImuSampleMessage | PiStatusMessage:
    payload = json.loads(line)
    source_id = str(payload.get("source_id", fallback_source_id))
    sequence = int(payload["sequence"])
    timestamp_us = int(payload["timestamp_us"])
    kind = str(payload["kind"])
    if kind == "lidar_scan":
        angles = np.asarray(payload["angle_rad"], dtype=np.float32)
        ranges = np.asarray(payload["range_m"], dtype=np.float32)
        if angles.shape != ranges.shape:
            raise ValueError("lidar angle_rad and range_m must have the same length")
        return LidarScanMessage(
            source_id=source_id,
            sequence=sequence,
            timestamp_us=timestamp_us,
            scan_points=np.stack((angles, ranges), axis=1).astype(np.float32),
        )
    if kind == "imu_sample":
        return ImuSampleMessage(
            source_id=source_id,
            sequence=sequence,
            timestamp_us=timestamp_us,
            accel_xyz_mps2=np.asarray(payload["accel_xyz_mps2"], dtype=np.float32),
            quat_xyzw=np.asarray(payload["quat_xyzw"], dtype=np.float32),
        )
    if kind == "pi_status":
        return PiStatusMessage(
            source_id=source_id,
            sequence=sequence,
            timestamp_us=timestamp_us,
            state=str(payload["state"]),
            cpu_temp_c=float(payload["cpu_temp_c"]),
            cpu_load_pct=float(payload["cpu_load_pct"]),
        )
    raise ValueError(f"unsupported sensor message kind: {kind}")


if __name__ == "__main__":
    raise SystemExit(main())
