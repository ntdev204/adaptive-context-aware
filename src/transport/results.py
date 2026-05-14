from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from src.utils.validation import validate_ndarray

from ._proto_helpers import add_proto_field


@dataclass(frozen=True, slots=True)
class TrackedEntityMessage:
    track_id: int
    bbox_xywh: np.ndarray
    position_xyz_m: np.ndarray
    velocity_xyz_mps: np.ndarray
    heading_rad: float
    confidence: float
    nearest_obstacle_distance_m: float | None = None


@dataclass(frozen=True, slots=True)
class RuntimeMetricsMessage:
    total_latency_ms: float
    camera_latency_ms: float
    detector_latency_ms: float
    fusion_latency_ms: float
    fps: float


@dataclass(frozen=True, slots=True)
class PerceptionResultMessage:
    source_id: str
    sequence: int
    timestamp_us: int
    entities: list[TrackedEntityMessage]
    metrics: RuntimeMetricsMessage


class PerceptionResultCodec:
    _classes: dict[str, type] | None = None

    @staticmethod
    def encode(message: PerceptionResultMessage) -> bytes:
        envelope = _get_result_classes()["PerceptionResultEnvelope"]()
        envelope.source_id = message.source_id
        envelope.sequence = message.sequence
        envelope.timestamp_us = message.timestamp_us

        for entity in message.entities:
            _encode_entity(envelope.entities.add(), entity)

        _encode_metrics(envelope.metrics, message.metrics)
        return envelope.SerializeToString()

    @staticmethod
    def decode(raw: bytes) -> PerceptionResultMessage:
        envelope = _get_result_classes()["PerceptionResultEnvelope"]()
        envelope.ParseFromString(raw)
        return PerceptionResultMessage(
            source_id=str(envelope.source_id),
            sequence=int(envelope.sequence),
            timestamp_us=int(envelope.timestamp_us),
            entities=[_decode_entity(e) for e in envelope.entities],
            metrics=_decode_metrics(envelope.metrics),
        )


# ------------------------------------------------------------------
# Encode helpers (SRP)
# ------------------------------------------------------------------


def _encode_entity(proto: object, entity: TrackedEntityMessage) -> None:
    validate_ndarray(entity.bbox_xywh, expected_shape=(4,), name="bbox_xywh")
    validate_ndarray(entity.position_xyz_m, expected_shape=(3,), name="position_xyz_m")
    validate_ndarray(entity.velocity_xyz_mps, expected_shape=(3,), name="velocity_xyz_mps")
    proto.track_id = entity.track_id
    proto.bbox_xywh.extend(entity.bbox_xywh.astype(float).tolist())
    proto.position_xyz_m.extend(entity.position_xyz_m.astype(float).tolist())
    proto.velocity_xyz_mps.extend(entity.velocity_xyz_mps.astype(float).tolist())
    proto.heading_rad = entity.heading_rad
    proto.confidence = entity.confidence
    if entity.nearest_obstacle_distance_m is not None:
        proto.nearest_obstacle_distance_m = entity.nearest_obstacle_distance_m


def _encode_metrics(proto: object, metrics: RuntimeMetricsMessage) -> None:
    proto.total_latency_ms = metrics.total_latency_ms
    proto.camera_latency_ms = metrics.camera_latency_ms
    proto.detector_latency_ms = metrics.detector_latency_ms
    proto.fusion_latency_ms = metrics.fusion_latency_ms
    proto.fps = metrics.fps


# ------------------------------------------------------------------
# Decode helpers (SRP)
# ------------------------------------------------------------------


def _decode_entity(proto: object) -> TrackedEntityMessage:
    return TrackedEntityMessage(
        track_id=int(proto.track_id),
        bbox_xywh=np.asarray(proto.bbox_xywh, dtype=np.float32),
        position_xyz_m=np.asarray(proto.position_xyz_m, dtype=np.float32),
        velocity_xyz_mps=np.asarray(proto.velocity_xyz_mps, dtype=np.float32),
        heading_rad=float(proto.heading_rad),
        confidence=float(proto.confidence),
        nearest_obstacle_distance_m=(
            float(proto.nearest_obstacle_distance_m) if proto.HasField("nearest_obstacle_distance_m") else None
        ),
    )


def _decode_metrics(proto: object) -> RuntimeMetricsMessage:
    return RuntimeMetricsMessage(
        total_latency_ms=float(proto.total_latency_ms),
        camera_latency_ms=float(proto.camera_latency_ms),
        detector_latency_ms=float(proto.detector_latency_ms),
        fusion_latency_ms=float(proto.fusion_latency_ms),
        fps=float(proto.fps),
    )


# ------------------------------------------------------------------
# Protobuf schema registration (lazy, cached)
# ------------------------------------------------------------------

_PROTO_PACKAGE = "adaptive.context.v1"


def _get_result_classes() -> dict[str, type]:
    if PerceptionResultCodec._classes is None:
        file_desc = descriptor_pb2.FileDescriptorProto()
        file_desc.name = "adaptive/context/v1/perception.proto"
        file_desc.package = _PROTO_PACKAGE
        _register_tracked_entity(file_desc)
        _register_runtime_metrics(file_desc)
        _register_result_envelope(file_desc)

        pool = descriptor_pool.DescriptorPool()
        pool.Add(file_desc)
        PerceptionResultCodec._classes = {
            name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"{_PROTO_PACKAGE}.{name}"))
            for name in ("PerceptionResultEnvelope", "TrackedEntity", "RuntimeMetrics")
        }
    return PerceptionResultCodec._classes


def _register_tracked_entity(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    msg = file_desc.message_type.add()
    msg.name = "TrackedEntity"
    add_proto_field(msg, "track_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add_proto_field(msg, "bbox_xywh", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)
    add_proto_field(msg, "position_xyz_m", 3, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)
    add_proto_field(msg, "velocity_xyz_mps", 4, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)
    add_proto_field(msg, "heading_rad", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    add_proto_field(msg, "confidence", 6, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    add_proto_field(
        msg,
        "nearest_obstacle_distance_m",
        7,
        descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
        proto3_optional=True,
    )


def _register_runtime_metrics(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    msg = file_desc.message_type.add()
    msg.name = "RuntimeMetrics"
    _METRIC_FIELDS = [
        "total_latency_ms",
        "camera_latency_ms",
        "detector_latency_ms",
        "fusion_latency_ms",
        "fps",
    ]
    for idx, name in enumerate(_METRIC_FIELDS, start=1):
        add_proto_field(msg, name, idx, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)


def _register_result_envelope(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    msg = file_desc.message_type.add()
    msg.name = "PerceptionResultEnvelope"
    add_proto_field(msg, "source_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    add_proto_field(msg, "sequence", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add_proto_field(msg, "timestamp_us", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add_proto_field(
        msg,
        "entities",
        10,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        repeated=True,
        type_name=f".{_PROTO_PACKAGE}.TrackedEntity",
    )
    add_proto_field(
        msg,
        "metrics",
        11,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PROTO_PACKAGE}.RuntimeMetrics",
    )
