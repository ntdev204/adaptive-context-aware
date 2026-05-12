from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


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
    _classes = None

    @staticmethod
    def encode(message: PerceptionResultMessage) -> bytes:
        classes = _result_message_classes()
        envelope = classes["PerceptionResultEnvelope"]()
        envelope.source_id = message.source_id
        envelope.sequence = message.sequence
        envelope.timestamp_us = message.timestamp_us
        for entity in message.entities:
            _validate_vector(entity.bbox_xywh, (4,), "bbox_xywh")
            _validate_vector(entity.position_xyz_m, (3,), "position_xyz_m")
            _validate_vector(entity.velocity_xyz_mps, (3,), "velocity_xyz_mps")
            entity_proto = envelope.entities.add()
            entity_proto.track_id = entity.track_id
            entity_proto.bbox_xywh.extend(entity.bbox_xywh.astype(float).tolist())
            entity_proto.position_xyz_m.extend(entity.position_xyz_m.astype(float).tolist())
            entity_proto.velocity_xyz_mps.extend(entity.velocity_xyz_mps.astype(float).tolist())
            entity_proto.heading_rad = entity.heading_rad
            entity_proto.confidence = entity.confidence
            if entity.nearest_obstacle_distance_m is not None:
                entity_proto.nearest_obstacle_distance_m = entity.nearest_obstacle_distance_m
        envelope.metrics.total_latency_ms = message.metrics.total_latency_ms
        envelope.metrics.camera_latency_ms = message.metrics.camera_latency_ms
        envelope.metrics.detector_latency_ms = message.metrics.detector_latency_ms
        envelope.metrics.fusion_latency_ms = message.metrics.fusion_latency_ms
        envelope.metrics.fps = message.metrics.fps
        return envelope.SerializeToString()

    @staticmethod
    def decode(raw: bytes) -> PerceptionResultMessage:
        classes = _result_message_classes()
        envelope = classes["PerceptionResultEnvelope"]()
        envelope.ParseFromString(raw)
        return PerceptionResultMessage(
            source_id=str(envelope.source_id),
            sequence=int(envelope.sequence),
            timestamp_us=int(envelope.timestamp_us),
            entities=[
                TrackedEntityMessage(
                    track_id=int(entity.track_id),
                    bbox_xywh=np.asarray(entity.bbox_xywh, dtype=np.float32),
                    position_xyz_m=np.asarray(entity.position_xyz_m, dtype=np.float32),
                    velocity_xyz_mps=np.asarray(entity.velocity_xyz_mps, dtype=np.float32),
                    heading_rad=float(entity.heading_rad),
                    confidence=float(entity.confidence),
                    nearest_obstacle_distance_m=(
                        float(entity.nearest_obstacle_distance_m)
                        if entity.HasField("nearest_obstacle_distance_m")
                        else None
                    ),
                )
                for entity in envelope.entities
            ],
            metrics=RuntimeMetricsMessage(
                total_latency_ms=float(envelope.metrics.total_latency_ms),
                camera_latency_ms=float(envelope.metrics.camera_latency_ms),
                detector_latency_ms=float(envelope.metrics.detector_latency_ms),
                fusion_latency_ms=float(envelope.metrics.fusion_latency_ms),
                fps=float(envelope.metrics.fps),
            ),
        )


def _validate_vector(vector: np.ndarray, expected_shape: tuple[int, ...], name: str) -> None:
    if vector.shape != expected_shape:
        raise ValueError(f"expected {name} shape {expected_shape}")
    if vector.dtype != np.float32:
        raise ValueError(f"expected {name} dtype float32")


def _result_message_classes():
    if PerceptionResultCodec._classes is None:
        file_desc = descriptor_pb2.FileDescriptorProto()
        file_desc.name = "adaptive/context/v1/perception.proto"
        file_desc.package = "adaptive.context.v1"
        _add_tracked_entity(file_desc)
        _add_runtime_metrics(file_desc)
        _add_result_envelope(file_desc)

        pool = descriptor_pool.DescriptorPool()
        pool.Add(file_desc)
        PerceptionResultCodec._classes = {
            name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"adaptive.context.v1.{name}"))
            for name in ("PerceptionResultEnvelope", "TrackedEntity", "RuntimeMetrics")
        }
    return PerceptionResultCodec._classes


def _add_tracked_entity(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    message_desc = file_desc.message_type.add()
    message_desc.name = "TrackedEntity"
    _add_field(message_desc, "track_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    _add_field(message_desc, "bbox_xywh", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)
    _add_field(message_desc, "position_xyz_m", 3, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)
    _add_field(message_desc, "velocity_xyz_mps", 4, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)
    _add_field(message_desc, "heading_rad", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "confidence", 6, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(
        message_desc,
        "nearest_obstacle_distance_m",
        7,
        descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
        proto3_optional=True,
    )


def _add_runtime_metrics(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    message_desc = file_desc.message_type.add()
    message_desc.name = "RuntimeMetrics"
    _add_field(message_desc, "total_latency_ms", 1, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "camera_latency_ms", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "detector_latency_ms", 3, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "fusion_latency_ms", 4, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "fps", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)


def _add_result_envelope(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    message_desc = file_desc.message_type.add()
    message_desc.name = "PerceptionResultEnvelope"
    _add_field(message_desc, "source_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(message_desc, "sequence", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    _add_field(message_desc, "timestamp_us", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    _add_field(
        message_desc,
        "entities",
        10,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        repeated=True,
        type_name=".adaptive.context.v1.TrackedEntity",
    )
    _add_field(
        message_desc,
        "metrics",
        11,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".adaptive.context.v1.RuntimeMetrics",
    )


def _add_field(
    message_desc: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
    *,
    repeated: bool = False,
    type_name: str | None = None,
    proto3_optional: bool = False,
) -> None:
    field_desc = message_desc.field.add()
    field_desc.name = name
    field_desc.number = number
    field_desc.label = (
        descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        if repeated
        else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    )
    field_desc.type = field_type
    if type_name is not None:
        field_desc.type_name = type_name
    if proto3_optional:
        oneof_desc = message_desc.oneof_decl.add()
        oneof_desc.name = f"_{name}"
        field_desc.oneof_index = len(message_desc.oneof_decl) - 1
        field_desc.proto3_optional = True
