from __future__ import annotations

from dataclasses import dataclass

import zmq

from .messages import SensorMessage, SensorMessageCodec


@dataclass(frozen=True, slots=True)
class ZmqSensorClientConfig:
    adaptive_host: str = "127.0.0.1"
    adaptive_port: int = 5555
    high_water_mark: int = 1
    jetson_host: str | None = None
    jetson_port: int | None = None

    @property
    def endpoint(self) -> str:
        host = self.jetson_host or self.adaptive_host
        port = self.jetson_port or self.adaptive_port
        return f"tcp://{host}:{port}"


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
