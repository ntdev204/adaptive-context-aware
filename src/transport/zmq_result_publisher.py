from __future__ import annotations

from dataclasses import dataclass

import zmq

from .results import PerceptionResultCodec, PerceptionResultMessage


@dataclass(frozen=True, slots=True)
class ZmqResultPublisherConfig:
    bind_host: str = "25.12.4.100"
    bind_port: int = 5556
    high_water_mark: int = 1

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.bind_host}:{self.bind_port}"


class ZmqResultPublisher:
    def __init__(self, config: ZmqResultPublisherConfig, context: zmq.Context | None = None) -> None:
        self.config = config
        self._context = context or zmq.Context.instance()
        self._socket: zmq.Socket | None = None

    def start(self) -> None:
        if self._socket is not None:
            return
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, self.config.high_water_mark)
        self._socket.bind(self.config.endpoint)

    def publish(self, message: PerceptionResultMessage) -> None:
        if self._socket is None:
            raise RuntimeError("result publisher is not started")
        self._socket.send(PerceptionResultCodec.encode(message), flags=zmq.NOBLOCK)

    def stop(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
