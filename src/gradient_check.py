"""Gradient validation utilities."""

from __future__ import annotations

import random

import numpy as np


def finite_difference_check(
    model,
    X: np.ndarray,
    y: np.ndarray,
    loss_name: str,
    epsilon: float = 1e-5,
    num_checks: int = 30,
    seed: int = 0,
):
    """Sample parameter entries and compare backprop to centered differences."""
    rng = random.Random(seed)
    grads = model.gradients(X, y, loss_name)
    candidates = []
    for layer_idx, layer in enumerate(model.params):
        for param_name in ("W", "b"):
            arr = layer[param_name]
            grad_name = "d" + param_name
            for idx in np.ndindex(arr.shape):
                candidates.append((layer_idx, param_name, grad_name, idx))

    rng.shuffle(candidates)
    rows = []
    max_error = 0.0
    for layer_idx, param_name, grad_name, idx in candidates[:num_checks]:
        original = model.params[layer_idx][param_name][idx]

        model.params[layer_idx][param_name][idx] = original + epsilon
        loss_plus = model.loss(X, y, loss_name)

        model.params[layer_idx][param_name][idx] = original - epsilon
        loss_minus = model.loss(X, y, loss_name)

        model.params[layer_idx][param_name][idx] = original
        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        backprop = grads[layer_idx][grad_name][idx]
        rel_error = abs(backprop - numerical) / max(1e-8, abs(backprop) + abs(numerical))
        max_error = max(max_error, float(rel_error))
        rows.append(
            {
                "layer": layer_idx,
                "parameter": param_name,
                "index": str(idx),
                "backprop": float(backprop),
                "numerical": float(numerical),
                "relative_error": float(rel_error),
            }
        )
    return {"max_relative_error": max_error, "checks": rows}


def jax_autodiff_check(*_args, **_kwargs):
    """Compare NumPy backprop gradients against JAX autodiff gradients."""
    try:
        from .mlp_jax import gradients as jax_gradients
    except ImportError as exc:
        raise RuntimeError(
            "JAX is not installed in this environment. Install requirements.txt to enable "
            "the JAX autodiff comparison."
        ) from exc
    model, X, y, loss_name = _args[:4]
    numpy_grads = model.gradients(X, y, loss_name)
    jax_grads = jax_gradients(
        model.params,
        X,
        y,
        loss_name=loss_name,
        hidden_activation=model.hidden_activation_name,
        output_activation=model.output_activation,
    )
    rows = []
    max_error = 0.0
    for i, (np_grad, jax_grad) in enumerate(zip(numpy_grads, jax_grads)):
        for np_name, jax_name in [("dW", "W"), ("db", "b")]:
            backprop = np.asarray(np_grad[np_name])
            autodiff = np.asarray(jax_grad[jax_name])
            rel = np.abs(backprop - autodiff) / np.maximum(1e-8, np.abs(backprop) + np.abs(autodiff))
            err = float(np.max(rel))
            max_error = max(max_error, err)
            rows.append({"layer": i, "parameter": jax_name, "max_relative_error": err})
    return {"max_relative_error": max_error, "checks": rows}
