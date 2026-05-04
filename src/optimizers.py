"""Optimizers for NumPy parameter dictionaries."""

from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(
        self,
        name: str,
        learning_rate: float = 0.01,
        beta: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = None
        self.t = 0

    def _ensure_state(self, params):
        if self.state is not None:
            return
        self.state = []
        for layer in params:
            self.state.append(
                {
                    "mW": np.zeros_like(layer["W"]),
                    "mb": np.zeros_like(layer["b"]),
                    "vW": np.zeros_like(layer["W"]),
                    "vb": np.zeros_like(layer["b"]),
                }
            )

    def update(self, params, grads):
        self._ensure_state(params)
        self.t += 1
        for i, (layer, grad) in enumerate(zip(params, grads)):
            if self.name == "gd":
                layer["W"] -= self.learning_rate * grad["dW"]
                layer["b"] -= self.learning_rate * grad["db"]
            elif self.name == "momentum":
                state = self.state[i]
                state["mW"] = self.beta * state["mW"] + (1.0 - self.beta) * grad["dW"]
                state["mb"] = self.beta * state["mb"] + (1.0 - self.beta) * grad["db"]
                layer["W"] -= self.learning_rate * state["mW"]
                layer["b"] -= self.learning_rate * state["mb"]
            elif self.name == "adam":
                state = self.state[i]
                state["mW"] = self.beta1 * state["mW"] + (1.0 - self.beta1) * grad["dW"]
                state["mb"] = self.beta1 * state["mb"] + (1.0 - self.beta1) * grad["db"]
                state["vW"] = self.beta2 * state["vW"] + (1.0 - self.beta2) * grad["dW"] ** 2
                state["vb"] = self.beta2 * state["vb"] + (1.0 - self.beta2) * grad["db"] ** 2

                mW_hat = state["mW"] / (1.0 - self.beta1**self.t)
                mb_hat = state["mb"] / (1.0 - self.beta1**self.t)
                vW_hat = state["vW"] / (1.0 - self.beta2**self.t)
                vb_hat = state["vb"] / (1.0 - self.beta2**self.t)

                layer["W"] -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.eps)
                layer["b"] -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.eps)
            else:
                raise ValueError(f"Unknown optimizer '{self.name}'")
        return params

