from __future__ import annotations

import json

from src.runtime.tensorrt_engine import _deserialize_engine_blob, _strip_ultralytics_metadata_prefix


class FakeRuntime:
    def __init__(self, expected_blob: bytes) -> None:
        self.expected_blob = expected_blob
        self.calls: list[bytes] = []

    def deserialize_cuda_engine(self, blob: bytes):
        self.calls.append(blob)
        if blob == self.expected_blob:
            return object()
        return None


def test_strip_ultralytics_metadata_prefix_returns_raw_engine_bytes() -> None:
    engine_bytes = b"TRT_ENGINE_BYTES"
    metadata_bytes = json.dumps({"description": "Ultralytics YOLO"}).encode("utf-8")
    blob = len(metadata_bytes).to_bytes(4, byteorder="little", signed=True) + metadata_bytes + engine_bytes

    assert _strip_ultralytics_metadata_prefix(blob) == engine_bytes


def test_deserialize_engine_blob_falls_back_to_ultralytics_prefixed_engine() -> None:
    engine_bytes = b"TRT_ENGINE_BYTES"
    metadata_bytes = json.dumps({"description": "Ultralytics YOLO"}).encode("utf-8")
    blob = len(metadata_bytes).to_bytes(4, byteorder="little", signed=True) + metadata_bytes + engine_bytes
    runtime = FakeRuntime(engine_bytes)

    engine = _deserialize_engine_blob(runtime, blob)

    assert engine is not None
    assert runtime.calls == [engine_bytes]


def test_deserialize_engine_blob_returns_raw_engine_immediately() -> None:
    engine_bytes = b"TRT_ENGINE_BYTES"
    runtime = FakeRuntime(engine_bytes)

    engine = _deserialize_engine_blob(runtime, engine_bytes)

    assert engine is not None
    assert runtime.calls == [engine_bytes]
