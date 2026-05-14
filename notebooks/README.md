# Notebooks

Thu muc nay luu notebook dung tren Colab Pro cho cac pipeline training hien co trong repo.

## Mapping notebook -> pipeline

- `01_yolo_finetune.ipynb`
  - `pipelines/train_detector.py`
  - Fine-tune `YOLO11` theo legacy detector split:
    - `cctv_person`
    - custom detector datasets nếu còn dùng tạm

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

1. Clone repo vao Colab.
2. Cai dependency:
   - `pip install -e .[dev]`
   - hoac `pip install -e .[engine,dev]` neu can phan detector/runtime
3. Download dataset cong khai truc tiep trong notebook bang `roboflow` hoac `kagglehub`.
4. Copy raw dataset tu nhieu nguon vao `repo/data/raw/...`.
5. Chinh tham so trong notebook.
6. Chay cell command goi thang `pipelines/`.

## Quy uoc dataset tren Colab

- Dataset cong khai duoc download truc tiep vao runtime Colab.
- Dataset custom duoc copy tu:
  - `MyDrive/deep/data/raw/custom_1`
  - `MyDrive/deep/data/raw/custom_2`
- Sau khi download/copy, notebook se move/copy vao:
  - `repo/data/raw/...`
- Ly do:
  - pipeline trong repo doc dataset theo duong dan `data/...`
  - khong can upload lai dataset cong khai len Drive

## Ghi chu cho H100

- `01_yolo_finetune.ipynb` da duoc set theo huong tan dung GPU manh hon:
  - batch size lon hon
  - image size lon hon
  - worker nhieu hon
  - artifact dir rieng cho H100 run
- `02_botsort_finetune.ipynb` da duoc tang so sequence va frame budget de khong lang phi tai nguyen.
- Neu van con du VRAM, ban co the thu tiep:
  - tang `BATCH_SIZE`
  - tang `IMGSZ`
  - tang `MAX_MOT20`, `MAX_HALLWAY`, `MAX_FRAMES`
- Neu gap OOM:
  - ha `BATCH_SIZE`
  - giu `WORKERS` va `IMGSZ` hop ly

## Nguyen tac

- Notebook chi chua orchestration, setup, va notes.
- Logic tai su dung lau dai dua vao `pipelines/` hoac `scripts/`.
- Khong luu weight/checkpoint/artifact lon trong `notebooks/`.
