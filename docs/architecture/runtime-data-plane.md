# Runtime Data Plane

Production runs on Jetson Orin Nano in Docker. FastAPI is the control plane only; camera frames and sensor streams do not travel through HTTP.

## Control Plane

- `GET /health`
- `GET /ready`
- `GET /config`
- `GET /metrics`
- `POST /control/start`
- `POST /control/stop`

Default control API port: `8080`.

## Data Plane

- Jetson: `25.12.4.100`
- Raspberry Pi: `25.12.4.101`
- Sensor ingest: `tcp://25.12.4.100:5555`
- Result publish: `tcp://25.12.4.100:5556`
- Sensor transport: ZMQ `PULL` on Jetson, ZMQ `PUSH` on Pi
- Result transport: ZMQ `PUB` on Jetson
- Payload: protobuf wire format matching `proto/sensors.proto`

LiDAR and IMU data must come from real device drivers on the Pi. Synthetic sensor packets are reserved for tests and benchmarks, not production runtime.

## Camera

The AstraS RGB-D camera is local to the Jetson. The production pipeline must read RGB and depth frames from the Jetson container runtime with the required device mounts. The control API does not accept HTTP frame uploads for the realtime path.

## Production Rule

Runtime fails or reports degraded state when the engine or real sensor streams are missing. It must not silently replace missing camera, LiDAR, depth, or IMU data with generated values.
