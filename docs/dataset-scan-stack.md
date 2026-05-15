# Dataset Scan Stack

Run these from `adaptive-context-aware` on the target machine:

```bash
make docker-build-dev
make docker-build-prod
make export-engine
make compose-up
```

`make compose-up` starts:

- `control-api`: adaptive runtime, ZMQ sensor ingest `5555`, result PUB `5556`, heartbeat `9093`.
- `rai-postgres`, `rai-server`, `rai-client`: website dataset and control UI.

Default endpoints:

- Wheeltec ROS2 / SCADA host: `25.12.4.101`
- Adaptive advertised host: `25.12.4.100`
- Website: `http://localhost:3000`
- Backend API: `http://localhost:8000`

Dataset scanning remains in `rai_website` at the Dataset page. Robot control, telemetry, and camera frames remain sourced from `wheeltec_ros2` through `wheeltec_scada_bridge`. The adaptive runtime subscribes to the same Wheeltec SCADA camera stream, combines it with LiDAR/IMU from `context_aware_bridge`, and publishes perception results back to the website result plane.

The compose stack sets `CTX_AUTOSTART=true`, so `control-api` starts its sensor ingest, camera subscriber, perception loop, result publisher, and heartbeat as soon as the service boots.

Override hosts when needed:

```powershell
$env:CTX_JETSON_HOST="<laptop-or-jetson-ip>"
$env:ZMQ_SCADA_HOST="<wheeltec-ros2-ip>"
$env:CTX_SCADA_CAMERA_HOST="<wheeltec-ros2-ip>"
make compose-up
```
