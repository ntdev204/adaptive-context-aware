from __future__ import annotations

import time

import numpy as np
import zmq
from zmq.error import Again

from src.runtime.sensor_store import SensorStore
from src.transport.messages import ImuSampleMessage, LidarScanMessage
from src.transport.zmq_sensor_client import ZmqSensorClient, ZmqSensorClientConfig
from src.transport.zmq_sensor_ingest import ZmqIngestConfig, ZmqSensorIngest


def test_zmq_sensor_ingest_receives_lidar_and_imu() -> None:
    context = zmq.Context()
    store = SensorStore()
    ingest = ZmqSensorIngest(
        ZmqIngestConfig(bind_host="127.0.0.1", bind_port=5599, recv_timeout_ms=10),
        handler=store.update,
        context=context,
    )
    ingest.start()
    client = ZmqSensorClient(
        ZmqSensorClientConfig(jetson_host="127.0.0.1", jetson_port=5599),
        context=context,
    )
    try:
        time.sleep(0.05)
        _send_with_retry(
            client,
            LidarScanMessage(
                source_id="pi-101",
                sequence=1,
                timestamp_us=100,
                scan_points=np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]], dtype=np.float32),
            )
        )
        _send_with_retry(
            client,
            ImuSampleMessage(
                source_id="pi-101",
                sequence=2,
                timestamp_us=101,
                accel_xyz_mps2=np.array([0.0, 0.0, 9.8], dtype=np.float32),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
        )

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            snapshot = store.snapshot()
            if snapshot.has_lidar and snapshot.has_imu:
                break
            time.sleep(0.01)

        snapshot = store.snapshot()
        assert snapshot.has_lidar
        assert snapshot.has_imu
        assert ingest.stats().messages_received == 2
    finally:
        client.close()
        ingest.stop()
        context.term()


def _send_with_retry(client: ZmqSensorClient, message: LidarScanMessage | ImuSampleMessage) -> None:
    deadline = time.monotonic() + 1.0
    while True:
        try:
            client.send(message)
            return
        except Again:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
