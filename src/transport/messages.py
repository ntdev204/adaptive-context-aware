from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from src.utils.validation import validate_ndarray

from ._proto_helpers import add_proto_field


class SensorMessageKind(StrEnum):
    LIDAR_SCAN = "lidar_scan"
    IMU_SAMPLE = "imu_sample"
    PI_STATUS = "pi_status"


@dataclass(frozen=True, slots=True)
class LidarScanMessage:
    source_id: str
    sequence: int
    timestamp_us: int
    scan_points: np.ndarray


@dataclass(frozen=True, slots=True)
class ImuSampleMessage:
    source_id: str
    sequence: int
    timestamp_us: int
    accel_xyz_mps2: np.ndarray
    quat_xyzw: np.ndarray


@dataclass(frozen=True, slots=True)
class PiStatusMessage:
    source_id: str
    sequence: int
    timestamp_us: int
    state: str
    cpu_temp_c: float
    cpu_load_pct: float


SensorMessage = LidarScanMessage | ImuSampleMessage | PiStatusMessage


class SensorMessageCodec:
    """Binary data-plane codec backed by protobuf wire format."""

    _classes: dict[str, type] | None = None

    @staticmethod
    def encode(message: SensorMessage) -> bytes:
        envelope = _get_sensor_classes()["SensorEnvelope"]()
        envelope.source_id = message.source_id
        envelope.sequence = message.sequence
        envelope.timestamp_us = message.timestamp_us

        if isinstance(message, LidarScanMessage):
            _encode_lidar(envelope, message)
        elif isinstance(message, ImuSampleMessage):
            _encode_imu(envelope, message)
        else:
            _encode_pi_status(envelope, message)

        return envelope.SerializeToString()

    @staticmethod
    def decode(raw: bytes) -> SensorMessage:
        envelope = _get_sensor_classes()["SensorEnvelope"]()
        envelope.ParseFromString(raw)
        payload_kind = envelope.WhichOneof("payload")

        if payload_kind == SensorMessageKind.LIDAR_SCAN.value:
            return _decode_lidar(envelope)
        if payload_kind == SensorMessageKind.IMU_SAMPLE.value:
            return _decode_imu(envelope)
        if payload_kind != SensorMessageKind.PI_STATUS.value:
            raise ValueError("sensor envelope does not contain a supported payload")
        return _decode_pi_status(envelope)


# ------------------------------------------------------------------
# Encode helpers (SRP — one function per message type)
# ------------------------------------------------------------------


def _encode_lidar(envelope: object, message: LidarScanMessage) -> None:
    validate_ndarray(message.scan_points, expected_dtype=np.float32, name="scan_points")
    if message.scan_points.ndim != 2 or message.scan_points.shape[1] != 2:
        raise ValueError("expected lidar scan shape [N, 2]")
    envelope.lidar_scan.angle_rad.extend(message.scan_points[:, 0].astype(float).tolist())
    envelope.lidar_scan.range_m.extend(message.scan_points[:, 1].astype(float).tolist())


def _encode_imu(envelope: object, message: ImuSampleMessage) -> None:
    validate_ndarray(message.accel_xyz_mps2, expected_shape=(3,), name="accel_xyz_mps2")
    validate_ndarray(message.quat_xyzw, expected_shape=(4,), name="quat_xyzw")
    imu = envelope.imu_sample
    imu.accel_x_mps2 = float(message.accel_xyz_mps2[0])
    imu.accel_y_mps2 = float(message.accel_xyz_mps2[1])
    imu.accel_z_mps2 = float(message.accel_xyz_mps2[2])
    imu.quat_x = float(message.quat_xyzw[0])
    imu.quat_y = float(message.quat_xyzw[1])
    imu.quat_z = float(message.quat_xyzw[2])
    imu.quat_w = float(message.quat_xyzw[3])


def _encode_pi_status(envelope: object, message: PiStatusMessage) -> None:
    envelope.pi_status.state = message.state
    envelope.pi_status.cpu_temp_c = message.cpu_temp_c
    envelope.pi_status.cpu_load_pct = message.cpu_load_pct


# ------------------------------------------------------------------
# Decode helpers (SRP — one function per message type)
# ------------------------------------------------------------------


def _decode_lidar(envelope: object) -> LidarScanMessage:
    angles = np.asarray(envelope.lidar_scan.angle_rad, dtype=np.float32)
    ranges = np.asarray(envelope.lidar_scan.range_m, dtype=np.float32)
    if angles.shape != ranges.shape:
        raise ValueError("lidar angle/range arrays must have the same length")
    return LidarScanMessage(
        source_id=str(envelope.source_id),
        sequence=int(envelope.sequence),
        timestamp_us=int(envelope.timestamp_us),
        scan_points=np.stack((angles, ranges), axis=1).astype(np.float32),
    )


def _decode_imu(envelope: object) -> ImuSampleMessage:
    imu = envelope.imu_sample
    return ImuSampleMessage(
        source_id=str(envelope.source_id),
        sequence=int(envelope.sequence),
        timestamp_us=int(envelope.timestamp_us),
        accel_xyz_mps2=np.array([imu.accel_x_mps2, imu.accel_y_mps2, imu.accel_z_mps2], dtype=np.float32),
        quat_xyzw=np.array([imu.quat_x, imu.quat_y, imu.quat_z, imu.quat_w], dtype=np.float32),
    )


def _decode_pi_status(envelope: object) -> PiStatusMessage:
    return PiStatusMessage(
        source_id=str(envelope.source_id),
        sequence=int(envelope.sequence),
        timestamp_us=int(envelope.timestamp_us),
        state=str(envelope.pi_status.state),
        cpu_temp_c=float(envelope.pi_status.cpu_temp_c),
        cpu_load_pct=float(envelope.pi_status.cpu_load_pct),
    )


# ------------------------------------------------------------------
# Protobuf schema registration (lazy, cached)
# ------------------------------------------------------------------

_PROTO_PACKAGE = "adaptive.context.v1"


def _get_sensor_classes() -> dict[str, type]:
    if SensorMessageCodec._classes is None:
        file_desc = descriptor_pb2.FileDescriptorProto()
        file_desc.name = "adaptive/context/v1/sensors.proto"
        file_desc.package = _PROTO_PACKAGE
        _register_lidar_scan(file_desc)
        _register_imu_sample(file_desc)
        _register_pi_status(file_desc)
        _register_sensor_envelope(file_desc)

        pool = descriptor_pool.DescriptorPool()
        pool.Add(file_desc)
        SensorMessageCodec._classes = {
            name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"{_PROTO_PACKAGE}.{name}"))
            for name in ("SensorEnvelope", "LidarScan", "ImuSample", "PiStatus")
        }
    return SensorMessageCodec._classes


def _register_lidar_scan(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    msg = file_desc.message_type.add()
    msg.name = "LidarScan"
    add_proto_field(msg, "angle_rad", 1, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)
    add_proto_field(msg, "range_m", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)


def _register_imu_sample(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    msg = file_desc.message_type.add()
    msg.name = "ImuSample"
    _IMU_FIELDS = [
        "accel_x_mps2",
        "accel_y_mps2",
        "accel_z_mps2",
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
    ]
    for idx, name in enumerate(_IMU_FIELDS, start=1):
        add_proto_field(msg, name, idx, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)


def _register_pi_status(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    msg = file_desc.message_type.add()
    msg.name = "PiStatus"
    add_proto_field(msg, "state", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    add_proto_field(msg, "cpu_temp_c", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    add_proto_field(msg, "cpu_load_pct", 3, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)


def _register_sensor_envelope(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    msg = file_desc.message_type.add()
    msg.name = "SensorEnvelope"
    add_proto_field(msg, "source_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    add_proto_field(msg, "sequence", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add_proto_field(msg, "timestamp_us", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    oneof_desc = msg.oneof_decl.add()
    oneof_desc.name = "payload"
    _PAYLOAD_FIELDS = [
        ("lidar_scan", 10, "LidarScan"),
        ("imu_sample", 11, "ImuSample"),
        ("pi_status", 12, "PiStatus"),
    ]
    for name, number, type_suffix in _PAYLOAD_FIELDS:
        type_name = f".{_PROTO_PACKAGE}.{type_suffix}"
        add_proto_field(
            msg,
            name,
            number,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=type_name,
            oneof_index=0,
        )
