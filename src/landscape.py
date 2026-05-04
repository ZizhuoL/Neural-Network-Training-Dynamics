"""Weight-space interpolation utilities."""

from __future__ import annotations

import numpy as np


def interpolate_params(params_a, params_b, alpha: float):
    params = []
    for layer_a, layer_b in zip(params_a, params_b):
        params.append(
            {
                "W": (1.0 - alpha) * layer_a["W"] + alpha * layer_b["W"],
                "b": (1.0 - alpha) * layer_a["b"] + alpha * layer_b["b"],
            }
        )
    return params


def interpolation_curve(model, params_a, params_b, X, y, loss_name: str, points: int = 50):
    alphas = np.linspace(0.0, 1.0, points)
    losses = []
    original = model.copy_params()
    try:
        for alpha in alphas:
            params_alpha = interpolate_params(params_a, params_b, float(alpha))
            losses.append(model.loss(X, y, loss_name, params=params_alpha))
    finally:
        model.set_params(original)
    return {"alpha": alphas.tolist(), "loss": losses}

