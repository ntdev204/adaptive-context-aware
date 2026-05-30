# Dataset Scan Stack

Run these from `adaptive-context-aware` on the Jetson:

```bash
make docker-build-dev
make docker-build-prod
make export-engine
make compose-up
```

`make export-engine` reuses `context-aware:jetson-dev`; there is no separate export-only image anymore. Exported `.engine` files are copied back to the codebase under `models/engines/`, not left only inside the Docker container.
If the TensorRT engine is missing, the container still boots in degraded mode so camera and sensor diagnostics are available; `/metrics` reports `ready=false` with `reason="waiting for inference engine"`. Set `CTX_REQUIRE_ENGINE_AT_BOOT=true` to restore fail-fast startup.
Camera backend defaults to `CTX_CAMERA_BACKEND=openni` for the Astra S. The runtime opens the camera through OpenNI only; it does not fall back to V4L2, so `/metrics.reason` reports the real Astra/OpenNI failure if the camera is not available.
`make compose-logs` includes startup diagnostics for engine path/size, model path/size, free disk, USB/OpenNI visibility, runtime readiness transitions, camera source errors, and perception-loop exceptions. Set `CTX_LOG_LEVEL=DEBUG` before `make compose-up` for more verbose Python logs.
`make docker-build-dev` and `make docker-build-prod` also build the `mlflow` stage in `docker/Dockerfile.jetson` (`context-aware:mlflow`), so MLflow dependencies are installed during the build step, not every time the service starts.

`make compose-up` on Jetson starts:

- `control-api`: adaptive runtime, ZMQ sensor ingest `5555`, result PUB `5556`, heartbeat `9093`.
- `mlflow`: lightweight MLOps tracking/logging UI on port `5000`.

Run `rai_website` separately on the laptop. Point it to the Jetson runtime with:

```bash
ZMQ_SCADA_HOST=100.120.77.81
ZMQ_SCADA_HOSTS=100.120.77.81
ZMQ_CAMERA_HOST=100.69.39.18
ADAPTIVE_API_URL=http://100.69.39.18:8080
ADAPTIVE_RESULT_HOST=100.69.39.18
ADAPTIVE_RESULT_PORT=5556
```

Default endpoints:

- Raspberry Pi <-> Jetson Ethernet: Pi `25.12.4.101`, Jetson `25.12.4.100`
- Raspberry Pi <-> `rai_website` command/telemetry: `100.120.77.81`
- Jetson <-> `rai_website` adaptive API/result/camera: `100.69.39.18`
- Website on laptop: `http://localhost:3000`
- Backend API on laptop: `http://localhost:8000`

Dataset scanning remains in `rai_website` at the Dataset page. Robot control, telemetry, and camera frames remain sourced from `wheeltec_ros2` through `wheeltec_scada_bridge`. The adaptive runtime subscribes to the same Wheeltec SCADA camera stream, combines it with LiDAR/IMU from `context_aware_bridge`, and publishes perception results back to the website result plane.

The compose stack sets `CTX_AUTOSTART=true`, so `control-api` starts its sensor ingest, camera subscriber, perception loop, result publisher, and heartbeat as soon as the service boots.

Override hosts when needed:

```bash
export CTX_JETSON_HOST="<jetson-ip-for-pi>"
export ZMQ_SCADA_HOST="<raspi-ip-for-website>"
export CTX_SCADA_CAMERA_HOST="<raspi-ethernet-ip-for-jetson>"
make compose-up
```
