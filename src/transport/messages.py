from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


class SensorMessageKind(StrEnum):
    PI_STATUS = "pi_status"


@dataclass(frozen=True, slots=True)
class PiStatusMessage:
    source_id: str
    sequence: int
    timestamp_us: int
    state: str
    cpu_temp_c: float
    cpu_load_pct: float


SensorMessage = PiStatusMessage


class SensorMessageCodec:
    """Binary data-plane codec backed by protobuf wire format."""

    _classes = None

    @staticmethod
    def encode(message: SensorMessage) -> bytes:
        classes = _sensor_message_classes()
        envelope = classes["SensorEnvelope"]()
        envelope.source_id = message.source_id
        envelope.sequence = message.sequence
        envelope.timestamp_us = message.timestamp_us
        envelope.pi_status.state = message.state
        envelope.pi_status.cpu_temp_c = message.cpu_temp_c
        envelope.pi_status.cpu_load_pct = message.cpu_load_pct
        return envelope.SerializeToString()

    @staticmethod
    def decode(raw: bytes) -> SensorMessage:
        classes = _sensor_message_classes()
        envelope = classes["SensorEnvelope"]()
        envelope.ParseFromString(raw)
        payload_kind = envelope.WhichOneof("payload")
        if payload_kind != SensorMessageKind.PI_STATUS.value:
            raise ValueError("sensor envelope does not contain a supported payload")
        return PiStatusMessage(
            source_id=str(envelope.source_id),
            sequence=int(envelope.sequence),
            timestamp_us=int(envelope.timestamp_us),
            state=str(envelope.pi_status.state),
            cpu_temp_c=float(envelope.pi_status.cpu_temp_c),
            cpu_load_pct=float(envelope.pi_status.cpu_load_pct),
        )


def _validate_vector(vector: np.ndarray, expected_shape: tuple[int, ...], name: str) -> None:
    if vector.shape != expected_shape:
        raise ValueError(f"expected {name} shape {expected_shape}")
    if vector.dtype != np.float32:
        raise ValueError(f"expected {name} dtype float32")


def _sensor_message_classes():
    if SensorMessageCodec._classes is None:
        file_desc = descriptor_pb2.FileDescriptorProto()
        file_desc.name = "adaptive/context/v1/sensors.proto"
        file_desc.package = "adaptive.context.v1"
        _add_pi_status(file_desc)
        _add_sensor_envelope(file_desc)

        pool = descriptor_pool.DescriptorPool()
        pool.Add(file_desc)
        SensorMessageCodec._classes = {
            name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"adaptive.context.v1.{name}"))
            for name in ("SensorEnvelope", "PiStatus")
        }
    return SensorMessageCodec._classes


def _add_pi_status(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    message_desc = file_desc.message_type.add()
    message_desc.name = "PiStatus"
    _add_field(message_desc, "state", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(message_desc, "cpu_temp_c", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "cpu_load_pct", 3, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)


def _add_sensor_envelope(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    message_desc = file_desc.message_type.add()
    message_desc.name = "SensorEnvelope"
    _add_field(message_desc, "source_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(message_desc, "sequence", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    _add_field(message_desc, "timestamp_us", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    oneof_desc = message_desc.oneof_decl.add()
    oneof_desc.name = "payload"
    _add_field(
        message_desc,
        "pi_status",
        12,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".adaptive.context.v1.PiStatus",
        oneof_index=0,
    )


def _add_field(
    message_desc: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
    *,
    repeated: bool = False,
    type_name: str | None = None,
    oneof_index: int | None = None,
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
    if oneof_index is not None:
        field_desc.oneof_index = oneof_index
