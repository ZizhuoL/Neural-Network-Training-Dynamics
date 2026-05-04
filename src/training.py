"""Reusable training loop for controlled experiments."""

from __future__ import annotations

import numpy as np

from .losses import compute_loss
from .metrics import accuracy, gradient_norm, parameter_norm
from .optimizers import Optimizer


def make_batches(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def evaluate(model, X: np.ndarray, y: np.ndarray, loss_name: str, task: str):
    y_pred, _ = model.forward(X)
    return {
        "loss": compute_loss(loss_name, y_pred, y),
        "accuracy": accuracy(y_pred, y, task),
    }


def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    loss_name: str = "binary_cross_entropy",
    task: str = "binary",
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    epochs: int = 1000,
    batch_size: int = 64,
    seed: int = 0,
    log_every: int = 10,
):
    rng = np.random.default_rng(seed)
    optimizer = Optimizer(optimizer_name, learning_rate=learning_rate)
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "gradient_norm": [],
        "parameter_norm": [],
    }

    last_grads = None
    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in make_batches(X_train, y_train, batch_size, rng):
            y_pred, cache = model.forward(X_batch)
            _ = compute_loss(loss_name, y_pred, y_batch)
            grads = model.backward(cache, y_batch, loss_name)
            model.params = optimizer.update(model.params, grads)
            last_grads = grads

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            train_eval = evaluate(model, X_train, y_train, loss_name, task)
            test_eval = evaluate(model, X_test, y_test, loss_name, task)
            if last_grads is None:
                last_grads = model.gradients(X_train[:batch_size], y_train[:batch_size], loss_name)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_eval["loss"])
            history["test_loss"].append(test_eval["loss"])
            history["train_accuracy"].append(train_eval["accuracy"])
            history["test_accuracy"].append(test_eval["accuracy"])
            history["gradient_norm"].append(gradient_norm(last_grads))
            history["parameter_norm"].append(parameter_norm(model.params))
    return history

