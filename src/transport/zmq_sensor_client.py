from __future__ import annotations

from dataclasses import dataclass

import zmq

from .messages import SensorMessage, SensorMessageCodec


@dataclass(frozen=True, slots=True)
class ZmqSensorClientConfig:
    jetson_host: str = "25.12.4.100"
    jetson_port: int = 5555
    high_water_mark: int = 1

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.jetson_host}:{self.jetson_port}"


class ZmqSensorClient:
    """Raspberry Pi-side PUSH client for LiDAR/IMU packets."""

    def __init__(self, config: ZmqSensorClientConfig, context: zmq.Context | None = None) -> None:
        self.config = config
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, self.config.high_water_mark)
        self._socket.connect(self.config.endpoint)

    def send(self, message: SensorMessage) -> None:
        self._socket.send(SensorMessageCodec.encode(message), flags=zmq.NOBLOCK)

    def close(self) -> None:
        self._socket.close(linger=0)
