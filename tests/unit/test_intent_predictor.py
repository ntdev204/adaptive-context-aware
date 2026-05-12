from __future__ import annotations

import numpy as np
import pytest
import torch

from models.intent_predictor import IntentPredictorNet
from src.decision.intent_predictor import ActivityClass, IntentDirection, IntentPredictor


def test_intent_predictor_net_contract() -> None:
    model = IntentPredictorNet()

    outputs = model(torch.zeros(3, 256))

    assert outputs["direction_logits"].shape == (3, 9)
    assert outputs["activity_logits"].shape == (3, 9)
    assert outputs["trajectory_offsets"].shape == (3, 2, 2)


def test_intent_predictor_decodes_predictions() -> None:
    predictor = IntentPredictor()
    predictions = predictor.predict(np.zeros((2, 256), dtype=np.float32))

    assert len(predictions) == 2
    assert predictions[0].direction_probabilities.shape == (9,)
    assert predictions[0].activity_probabilities.shape == (9,)
    assert predictions[0].trajectory_offsets.shape == (2, 2)
    assert predictions[0].direction_probabilities.sum() == pytest.approx(1.0)
    assert predictions[0].activity_probabilities.sum() == pytest.approx(1.0)
    assert isinstance(predictions[0].direction, IntentDirection)
    assert isinstance(predictions[0].activity, ActivityClass)


def test_intent_predictor_rejects_wrong_shape() -> None:
    predictor = IntentPredictor()

    with pytest.raises(ValueError):
        predictor.predict(np.zeros((2, 128), dtype=np.float32))
