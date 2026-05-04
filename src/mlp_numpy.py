"""Configurable multilayer perceptron implemented with NumPy."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .activations import get_activation, sigmoid, softmax
from .losses import compute_loss


class MLPNumPy:
    """A small, inspectable MLP with arbitrary depth and width."""

    def __init__(
        self,
        layer_sizes: list[int],
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid",
        initialization: str = "auto",
        seed: int = 0,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output dimensions")
        self.layer_sizes = list(layer_sizes)
        self.hidden_activation_name = hidden_activation
        self.output_activation = output_activation
        self.initialization = initialization
        self.rng = np.random.default_rng(seed)
        self.params = self._init_params()

    def _init_params(self) -> list[dict[str, np.ndarray]]:
        params = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = self._init_scale(fan_in, fan_out)
            W = self.rng.normal(0.0, scale, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            params.append({"W": W.astype(float), "b": b.astype(float)})
        return params

    def _init_scale(self, fan_in: int, fan_out: int) -> float:
        method = self.initialization
        if method == "auto":
            if self.hidden_activation_name in {"relu", "leaky_relu"}:
                method = "he"
            elif self.hidden_activation_name in {"sigmoid", "tanh"}:
                method = "xavier"
            else:
                method = "small_random"

        if method == "he":
            return float(np.sqrt(2.0 / fan_in))
        if method == "xavier":
            return float(np.sqrt(2.0 / (fan_in + fan_out)))
        if method == "small_random":
            return 0.01
        raise ValueError(f"Unknown initialization '{self.initialization}'")

    def copy(self) -> "MLPNumPy":
        clone = MLPNumPy(
            self.layer_sizes,
            hidden_activation=self.hidden_activation_name,
            output_activation=self.output_activation,
            initialization=self.initialization,
        )
        clone.params = self.copy_params()
        return clone

    def copy_params(self) -> list[dict[str, np.ndarray]]:
        return deepcopy(self.params)

    def set_params(self, params: list[dict[str, np.ndarray]]) -> None:
        self.params = deepcopy(params)

    def forward(self, X: np.ndarray, params: list[dict[str, np.ndarray]] | None = None):
        params = self.params if params is None else params
        act, _ = get_activation(self.hidden_activation_name)
        A = X
        cache = {"A": [X], "Z": []}

        for layer in params[:-1]:
            Z = A @ layer["W"] + layer["b"]
            A = act(Z)
            cache["Z"].append(Z)
            cache["A"].append(A)

        out_layer = params[-1]
        logits = A @ out_layer["W"] + out_layer["b"]
        cache["Z"].append(logits)
        if self.output_activation == "sigmoid":
            y_pred = sigmoid(logits)
        elif self.output_activation == "softmax":
            y_pred = softmax(logits)
        elif self.output_activation == "linear":
            y_pred = logits
        else:
            raise ValueError(f"Unknown output activation '{self.output_activation}'")
        cache["A"].append(y_pred)
        return y_pred, cache

    def loss(self, X: np.ndarray, y: np.ndarray, loss_name: str, params=None) -> float:
        y_pred, _ = self.forward(X, params=params)
        return compute_loss(loss_name, y_pred, y)

    def backward(self, cache: dict[str, list[np.ndarray]], y: np.ndarray, loss_name: str):
        m = y.shape[0]
        y_pred = cache["A"][-1]
        logits = cache["Z"][-1]

        if loss_name == "binary_cross_entropy" and self.output_activation == "sigmoid":
            dZ = y_pred - y
        elif loss_name == "softmax_cross_entropy" and self.output_activation == "softmax":
            dZ = y_pred - y
        elif loss_name == "mse":
            output_dim = y_pred.shape[1]
            dA = 2.0 * (y_pred - y) / output_dim
            if self.output_activation == "linear":
                dZ = dA
            elif self.output_activation == "sigmoid":
                dZ = dA * y_pred * (1.0 - y_pred)
            else:
                dZ = dA
        else:
            raise ValueError(
                f"Unsupported loss/output pair: loss={loss_name}, output={self.output_activation}"
            )

        grads: list[dict[str, np.ndarray]] = [None] * len(self.params)  # type: ignore[list-item]

        A_prev = cache["A"][-2]
        W = self.params[-1]["W"]
        grads[-1] = {
            "dW": A_prev.T @ dZ / m,
            "db": np.sum(dZ, axis=0, keepdims=True) / m,
        }
        dA_prev = dZ @ W.T

        _, hidden_grad = get_activation(self.hidden_activation_name)
        for i in reversed(range(len(self.params) - 1)):
            Z = cache["Z"][i]
            dZ = dA_prev * hidden_grad(Z)
            A_prev = cache["A"][i]
            W = self.params[i]["W"]
            grads[i] = {
                "dW": A_prev.T @ dZ / m,
                "db": np.sum(dZ, axis=0, keepdims=True) / m,
            }
            dA_prev = dZ @ W.T

        self._assert_gradient_shapes(grads)
        return grads

    def gradients(self, X: np.ndarray, y: np.ndarray, loss_name: str):
        _, cache = self.forward(X)
        return self.backward(cache, y, loss_name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred, _ = self.forward(X)
        if self.output_activation == "sigmoid":
            return (y_pred >= 0.5).astype(int)
        if self.output_activation == "softmax":
            return np.argmax(y_pred, axis=1)
        return y_pred

    def _assert_gradient_shapes(self, grads: list[dict[str, np.ndarray]]) -> None:
        for i, (layer, grad) in enumerate(zip(self.params, grads)):
            if grad["dW"].shape != layer["W"].shape:
                raise AssertionError(f"Layer {i} dW shape {grad['dW'].shape} != W {layer['W'].shape}")
            if grad["db"].shape != layer["b"].shape:
                raise AssertionError(f"Layer {i} db shape {grad['db'].shape} != b {layer['b'].shape}")

