# Phase 7: Data Collection & MLOps

> Collection tooling, annotation workflow, model registry, experiment tracking

## Goal

Data-Centric AI approach: build the tools for data collection → annotation → training → deployment cycle.

## Prerequisite

Phase 0.5 (Data Foundation) complete — schema, HDF5 format, and fixtures already exist.

## Scope Note

> **Data schema and record format are defined in Phase 0.5** (see `PLAN-phase-05-data-foundation.md`).
> This phase focuses on **collection tooling, annotation workflow, model registry, and experiment tracking**.
> T7.1 and T7.2 from the original plan have been moved to Phase 0.5.

## Spec Reference

- **Data:** `docs/specs/data-schema.md` — all formats and schemas
- **Benchmarking:** `docs/specs/benchmarking.md` — threshold upgrade path (bootstrap → production)

---

## Tasks

- [ ] **T7.1**: Annotation pipeline — labeling tool setup
  - CVAT hoặc Label Studio cho video annotation
  - Custom scripts cho LiDAR + activity labeling
  - Annotation format MUST match `data-schema.md` §3
  → Verify: Annotate 1 sample clip end-to-end, output validates against `annotation_schema.json`

- [ ] **T7.2**: Training pipeline — scripts cho train mỗi model
  - `train_detector.py`, `train_gnn.py`, `train_attention.py`
  - Config-driven, reproducible
  - Input/output shapes match `data-schema.md` §5
  → Verify: Train 1 epoch mỗi model, loss decreases

- [ ] **T7.3**: Model registry — versioning ONNX models
  - Git LFS hoặc DVC cho model files
  - Metadata: accuracy, latency, date, config
  → Verify: Push + pull model versions

- [ ] **T7.4**: Auto-rebuild trigger — GitHub webhook → Jetson rebuild
  - Push to main → build Docker image → push registry → Jetson pulls
  → Verify: Code change → Jetson running new version trong <10 phút

- [ ] **T7.5**: Experiment tracking — log training runs
  - MLflow hoặc custom JSON logs
  → Verify: Compare 2 training runs side-by-side

- [ ] **T7.6**: Threshold upgrade — transition from bootstrap to production thresholds
  - Once custom dataset >500 annotated frames: update detection thresholds
  - Once >10 robot sequences: update tracking thresholds
  - Update `tests/benchmark/baselines/` and `benchmarking.md` §4
  → Verify: New thresholds documented, CI baselines refreshed

## Done When

- [ ] Annotation pipeline produces schema-valid outputs
- [ ] Train + deploy cycle works end-to-end
- [ ] Model versioning in place
- [ ] Auto-rebuild pipeline functional
- [ ] Experiment tracking operational
- [ ] Threshold upgrade path documented and first upgrade executed (when data available)
