from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


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

    _classes = None

    @staticmethod
    def encode(message: SensorMessage) -> bytes:
        classes = _sensor_message_classes()
        envelope = classes["SensorEnvelope"]()
        envelope.source_id = message.source_id
        envelope.sequence = message.sequence
        envelope.timestamp_us = message.timestamp_us
        if isinstance(message, LidarScanMessage):
            _validate_scan_points(message.scan_points)
            envelope.lidar_scan.angle_rad.extend(message.scan_points[:, 0].astype(float).tolist())
            envelope.lidar_scan.range_m.extend(message.scan_points[:, 1].astype(float).tolist())
        elif isinstance(message, ImuSampleMessage):
            _validate_vector(message.accel_xyz_mps2, expected_shape=(3,), name="accel_xyz_mps2")
            _validate_vector(message.quat_xyzw, expected_shape=(4,), name="quat_xyzw")
            envelope.imu_sample.accel_x_mps2 = float(message.accel_xyz_mps2[0])
            envelope.imu_sample.accel_y_mps2 = float(message.accel_xyz_mps2[1])
            envelope.imu_sample.accel_z_mps2 = float(message.accel_xyz_mps2[2])
            envelope.imu_sample.quat_x = float(message.quat_xyzw[0])
            envelope.imu_sample.quat_y = float(message.quat_xyzw[1])
            envelope.imu_sample.quat_z = float(message.quat_xyzw[2])
            envelope.imu_sample.quat_w = float(message.quat_xyzw[3])
        else:
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
        if payload_kind == SensorMessageKind.LIDAR_SCAN.value:
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
        if payload_kind == SensorMessageKind.IMU_SAMPLE.value:
            return ImuSampleMessage(
                source_id=str(envelope.source_id),
                sequence=int(envelope.sequence),
                timestamp_us=int(envelope.timestamp_us),
                accel_xyz_mps2=np.asarray(
                    [
                        envelope.imu_sample.accel_x_mps2,
                        envelope.imu_sample.accel_y_mps2,
                        envelope.imu_sample.accel_z_mps2,
                    ],
                    dtype=np.float32,
                ),
                quat_xyzw=np.asarray(
                    [
                        envelope.imu_sample.quat_x,
                        envelope.imu_sample.quat_y,
                        envelope.imu_sample.quat_z,
                        envelope.imu_sample.quat_w,
                    ],
                    dtype=np.float32,
                ),
            )
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


def _validate_scan_points(scan_points: np.ndarray) -> None:
    if scan_points.ndim != 2 or scan_points.shape[1] != 2:
        raise ValueError("expected lidar scan shape [N, 2]")
    if scan_points.dtype != np.float32:
        raise ValueError("expected lidar scan dtype float32")


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
        _add_lidar_scan(file_desc)
        _add_imu_sample(file_desc)
        _add_pi_status(file_desc)
        _add_sensor_envelope(file_desc)

        pool = descriptor_pool.DescriptorPool()
        pool.Add(file_desc)
        SensorMessageCodec._classes = {
            name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"adaptive.context.v1.{name}"))
            for name in ("SensorEnvelope", "LidarScan", "ImuSample", "PiStatus")
        }
    return SensorMessageCodec._classes


def _add_lidar_scan(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    message_desc = file_desc.message_type.add()
    message_desc.name = "LidarScan"
    _add_field(message_desc, "angle_rad", 1, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)
    _add_field(message_desc, "range_m", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, repeated=True)


def _add_imu_sample(file_desc: descriptor_pb2.FileDescriptorProto) -> None:
    message_desc = file_desc.message_type.add()
    message_desc.name = "ImuSample"
    _add_field(message_desc, "accel_x_mps2", 1, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "accel_y_mps2", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "accel_z_mps2", 3, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "quat_x", 4, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "quat_y", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "quat_z", 6, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    _add_field(message_desc, "quat_w", 7, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)


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
        "lidar_scan",
        10,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".adaptive.context.v1.LidarScan",
        oneof_index=0,
    )
    _add_field(
        message_desc,
        "imu_sample",
        11,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".adaptive.context.v1.ImuSample",
        oneof_index=0,
    )
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
