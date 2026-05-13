"""Shared array / tensor validation helpers."""

from __future__ import annotations

import numpy as np


def validate_ndarray(
    array: np.ndarray,
    *,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: np.dtype | type = np.float32,
    name: str = "array",
) -> None:
    """Raise :class:`ValueError` if *array* fails shape or dtype checks.

    Args:
        array: The numpy array to validate.
        expected_shape: If given, assert ``array.shape == expected_shape``.
        expected_dtype: Assert ``array.dtype == expected_dtype`` (default ``float32``).
        name: Human-readable label used in error messages.
    """
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"expected {name} shape {expected_shape}, got {array.shape}")
    if array.dtype != expected_dtype:
        raise ValueError(f"expected {name} dtype {expected_dtype}, got {array.dtype}")
