from __future__ import annotations

import time

import zmq
from zmq.error import Again

from src.runtime.sensor_store import SensorStore
from src.transport.messages import PiStatusMessage
from src.transport.zmq_sensor_client import ZmqSensorClient, ZmqSensorClientConfig
from src.transport.zmq_sensor_ingest import ZmqIngestConfig, ZmqSensorIngest


def test_zmq_sensor_ingest_receives_pi_status() -> None:
    context = zmq.Context()
    store = SensorStore()
    ingest = ZmqSensorIngest(
        ZmqIngestConfig(bind_host="127.0.0.1", bind_port=5599, recv_timeout_ms=10),
        handler=store.update,
        context=context,
    )
    ingest.start()
    client = ZmqSensorClient(
        ZmqSensorClientConfig(adaptive_host="127.0.0.1", adaptive_port=5599),
        context=context,
    )
    try:
        time.sleep(0.05)
        _send_with_retry(
            client,
            PiStatusMessage(
                source_id="pi-101",
                sequence=1,
                timestamp_us=100,
                state="NORMAL",
                cpu_temp_c=42.0,
                cpu_load_pct=8.5,
            ),
        )

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            snapshot = store.snapshot()
            if snapshot.pi_state is not None:
                break
            time.sleep(0.01)

        snapshot = store.snapshot()
        assert snapshot.pi_state == "NORMAL"
        assert ingest.stats().messages_received == 1
    finally:
        client.close()
        ingest.stop()
        context.term()


def _send_with_retry(client: ZmqSensorClient, message: PiStatusMessage) -> None:
    deadline = time.monotonic() + 1.0
    while True:
        try:
            client.send(message)
            return
        except Again:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
