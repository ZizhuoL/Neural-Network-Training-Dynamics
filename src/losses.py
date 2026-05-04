"""Loss functions for regression, binary classification, and multiclass tasks."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(np.mean(loss))


def softmax_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    p = np.clip(y_pred, EPS, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))


def compute_loss(name: str, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if name == "mse":
        return mse(y_pred, y_true)
    if name == "binary_cross_entropy":
        return binary_cross_entropy(y_pred, y_true)
    if name == "softmax_cross_entropy":
        return softmax_cross_entropy(y_pred, y_true)
    raise ValueError(f"Unknown loss '{name}'")

