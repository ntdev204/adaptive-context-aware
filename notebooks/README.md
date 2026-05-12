# Notebooks

Thu muc nay luu notebook dung tren Colab Pro cho cac pipeline training hien co trong repo.

## Mapping notebook -> pipeline

- `01_yolo_finetune.ipynb`
  - `pipelines/train_detector.py`
  - Fine-tune `YOLO11` theo 2 stage:
    - `cctv_person`
    - `custom_1 + custom_2` (giu nguyen 6 class)

- `02_botsort_finetune.ipynb`
  - `pipelines/train_tracker_botsort.py`
  - Tune tracker theo 2 stage:
    - `mot20`
    - `hallway_rgbdt`

- `03_complexity_estimator_training.ipynb`
  - `pipelines/train_estimator.py`

- `04_gru_training.ipynb`
  - `pipelines/train_gru.py`

- `05_tcn_training.ipynb`
  - `pipelines/train_tcn.py`

- `06_attention_training.ipynb`
  - `pipelines/train_attention.py`

- `07_gnn_training.ipynb`
  - `pipelines/train_gnn.py`

- `08_reasoning_heads_training.ipynb`
  - `pipelines/train_fusion.py`
  - `pipelines/train_intent_predictor.py`
  - `pipelines/train_anomaly_detector.py`
  - `pipelines/train_reasoning.py`

## Cach dung tren Colab

1. Mount Google Drive neu muon luu checkpoint.
2. Clone repo vao Colab.
3. Cai dependency:
   - `pip install -e .[dev]`
   - hoac `pip install -e .[engine,dev]` neu can phan detector/runtime
4. Chinh tham so trong notebook.
5. Chay cell command goi thang `pipelines/`.

## Nguyen tac

- Notebook chi chua orchestration, setup, va notes.
- Logic tai su dung lau dai dua vao `pipelines/` hoac `scripts/`.
- Khong luu weight/checkpoint/artifact lon trong `notebooks/`.
