"""Metrics and norm diagnostics."""

from __future__ import annotations

import numpy as np


def binary_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    labels = (y_pred >= 0.5).astype(int)
    return float(np.mean(labels == y_true.astype(int)))


def multiclass_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return float(np.mean(pred_labels == true_labels))


def accuracy(y_pred: np.ndarray, y_true: np.ndarray, task: str) -> float:
    if task == "binary":
        return binary_accuracy(y_pred, y_true)
    if task == "multiclass":
        return multiclass_accuracy(y_pred, y_true)
    return float("nan")


def gradient_norm(grads: list[dict[str, np.ndarray]]) -> float:
    total = 0.0
    for grad in grads:
        total += float(np.sum(grad["dW"] ** 2) + np.sum(grad["db"] ** 2))
    return float(np.sqrt(total))


def parameter_norm(params: list[dict[str, np.ndarray]]) -> float:
    total = 0.0
    for layer in params:
        total += float(np.sum(layer["W"] ** 2) + np.sum(layer["b"] ** 2))
    return float(np.sqrt(total))

