"""Activation functions and derivatives for NumPy MLPs."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))


def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_grad(z: np.ndarray) -> np.ndarray:
    a = np.tanh(z)
    return 1.0 - a * a


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0.0, z, alpha * z)


def leaky_relu_grad(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0.0, 1.0, alpha)


def linear(z: np.ndarray) -> np.ndarray:
    return z


def linear_grad(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


_ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
    "leaky_relu": (leaky_relu, leaky_relu_grad),
    "linear": (linear, linear_grad),
}


def get_activation(name: str):
    """Return activation and derivative functions by name."""
    try:
        return _ACTIVATIONS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"Unknown activation '{name}'. Valid options: {valid}") from exc


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

