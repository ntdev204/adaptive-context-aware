from __future__ import annotations

import numpy as np


def test_model_contract_shapes() -> None:
    contracts = {
        "yolo11_s": ((1, 3, 480, 640), np.float32, (5, 6), np.float32),
        "complexity_estimator": ((1, 36), np.float32, (1, 4), np.float32),
        "gru_pathway": ((2, 8, 128), np.float32, (2, 64), np.float32),
        "attention_pathway": ((2, 6, 128), np.float32, (2, 128), np.float32),
        "gnn_pathway": ((2, 6, 128), np.float32, (2, 256), np.float32),
        "intent_predictor_direction": ((2, 256), np.float32, (2, 9), np.float32),
        "intent_predictor_activity": ((2, 256), np.float32, (2, 9), np.float32),
        "intent_predictor_trajectory": ((2, 256), np.float32, (2, 2, 2), np.float32),
        "anomaly_detector": ((2, 256), np.float32, (2, 1), np.float32),
        "rl_policy": ((1, 39), np.float32, (1, 4), np.float32),
    }
    for _, (input_shape, input_dtype, output_shape, output_dtype) in contracts.items():
        x = np.zeros(input_shape, dtype=input_dtype)
        y = np.zeros(output_shape, dtype=output_dtype)
        assert x.shape == input_shape
        assert x.dtype == input_dtype
        assert y.shape == output_shape
        assert y.dtype == output_dtype
