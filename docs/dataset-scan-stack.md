# Dataset Scan Stack

Run these from `adaptive-context-aware` on the Jetson:

```bash
make docker-build-dev
make docker-build-prod
make export-engine
make compose-up
```

`make export-engine` reuses `ctx-aware:dev`; there is no separate export-only image anymore.

`make compose-up` on Jetson starts only:

- `control-api`: adaptive runtime, ZMQ sensor ingest `5555`, result PUB `5556`, heartbeat `9093`.

Run `rai_website` separately on the laptop. Point it to the Jetson runtime with:

```bash
ADAPTIVE_API_URL=http://100.69.39.18:8080
ADAPTIVE_RESULT_HOST=100.69.39.18
ADAPTIVE_RESULT_PORT=5556
```

Default endpoints:

- Wheeltec ROS2 / SCADA host: `25.12.4.101`
- Adaptive host for Pi/ROS2: `25.12.4.100`
- Adaptive host for `rai_website`: `100.69.39.18`
- Website on laptop: `http://localhost:3000`
- Backend API on laptop: `http://localhost:8000`

Dataset scanning remains in `rai_website` at the Dataset page. Robot control, telemetry, and camera frames remain sourced from `wheeltec_ros2` through `wheeltec_scada_bridge`. The adaptive runtime subscribes to the same Wheeltec SCADA camera stream, combines it with LiDAR/IMU from `context_aware_bridge`, and publishes perception results back to the website result plane.

The compose stack sets `CTX_AUTOSTART=true`, so `control-api` starts its sensor ingest, camera subscriber, perception loop, result publisher, and heartbeat as soon as the service boots.

Override hosts when needed:

```bash
export CTX_JETSON_HOST="<jetson-ip-for-pi>"
export ZMQ_SCADA_HOST="<wheeltec-ros2-ip>"
export CTX_SCADA_CAMERA_HOST="<wheeltec-ros2-ip>"
make compose-up
```
