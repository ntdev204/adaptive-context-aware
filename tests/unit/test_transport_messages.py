from __future__ import annotations

from src.transport.messages import PiStatusMessage, SensorMessageCodec


def test_pi_status_message_roundtrip() -> None:
    message = PiStatusMessage(
        source_id="pi-101",
        sequence=1,
        timestamp_us=123456,
        state="NORMAL",
        cpu_temp_c=45.0,
        cpu_load_pct=12.5,
    )

    raw = SensorMessageCodec.encode(message)
    decoded = SensorMessageCodec.decode(raw)

    assert isinstance(raw, bytes)
    assert isinstance(decoded, PiStatusMessage)
    assert decoded.source_id == "pi-101"
    assert decoded.state == "NORMAL"
    assert abs(decoded.cpu_temp_c - 45.0) < 1e-4
    assert abs(decoded.cpu_load_pct - 12.5) < 1e-4
