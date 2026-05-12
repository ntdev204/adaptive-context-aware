# Runtime Data Plane

Production runs on the laptop in Docker. FastAPI is the control plane only; camera frames and sensor streams do not travel through HTTP.

## Control Plane

- `GET /health`
- `GET /ready`
- `GET /config`
- `GET /metrics`
- `POST /control/start`
- `POST /control/stop`

Default control API port: `8080`.

## Data Plane

- Laptop adaptive runtime: set `CTX_ADAPTIVE_HOST` to the laptop IP reachable from Raspberry Pi
- Raspberry Pi: `25.12.4.101`
- Sensor ingest: `tcp://<LAPTOP_IP>:5555`
- Result publish: `tcp://<LAPTOP_IP>:5556`
- Sensor transport: ZMQ `PULL` on laptop runtime, ZMQ `PUSH` on Pi
- Result transport: ZMQ `PUB` on laptop runtime
- Payload: protobuf wire format matching `proto/sensors.proto`

LiDAR and IMU data must come from real device drivers on the Pi. Synthetic sensor packets are reserved for tests and benchmarks, not production runtime.

## Camera

The AstraS RGB-D camera is local to the laptop runtime. The production pipeline must read RGB and depth frames from the laptop container runtime with the required device mounts. The control API does not accept HTTP frame uploads for the realtime path.

## Production Rule

Runtime fails or reports degraded state when the engine or real sensor streams are missing. It must not silently replace missing camera, LiDAR, depth, or IMU data with generated values.
