"""Shared numeric utilities used across multiple subsystems."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax for 2-D logit arrays.

    Args:
        logits: Array of shape ``[B, C]`` with raw logits.

    Returns:
        Probabilities with the same shape, dtype ``float32``.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def clip01(value: float) -> float:
    """Clip *value* to the ``[0, 1]`` interval."""
    return float(np.clip(value, 0.0, 1.0))
