"""JAX reference implementation for autodiff validation and HVPs."""

from __future__ import annotations


def require_jax():
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            "JAX is required for autodiff validation and native HVPs. "
            "Install the dependencies in requirements.txt to enable this module."
        ) from exc
    return jax, jnp


def available() -> bool:
    try:
        require_jax()
    except RuntimeError:
        return False
    return True


def _activation(jnp, name: str, z):
    if name == "sigmoid":
        return 1.0 / (1.0 + jnp.exp(-jnp.clip(z, -500.0, 500.0)))
    if name == "tanh":
        return jnp.tanh(z)
    if name == "relu":
        return jnp.maximum(0.0, z)
    if name == "leaky_relu":
        return jnp.where(z > 0.0, z, 0.01 * z)
    if name == "linear":
        return z
    raise ValueError(f"Unknown activation '{name}'")


def _softmax(jnp, logits):
    shifted = logits - jnp.max(logits, axis=1, keepdims=True)
    exp = jnp.exp(shifted)
    return exp / jnp.sum(exp, axis=1, keepdims=True)


def forward(params, X, hidden_activation: str = "relu", output_activation: str = "sigmoid"):
    _, jnp = require_jax()
    A = jnp.asarray(X)
    for layer in params[:-1]:
        A = _activation(jnp, hidden_activation, A @ jnp.asarray(layer["W"]) + jnp.asarray(layer["b"]))
    logits = A @ jnp.asarray(params[-1]["W"]) + jnp.asarray(params[-1]["b"])
    if output_activation == "sigmoid":
        return _activation(jnp, "sigmoid", logits)
    if output_activation == "softmax":
        return _softmax(jnp, logits)
    if output_activation == "linear":
        return logits
    raise ValueError(f"Unknown output activation '{output_activation}'")


def loss(
    params,
    X,
    y,
    loss_name: str = "binary_cross_entropy",
    hidden_activation: str = "relu",
    output_activation: str = "sigmoid",
):
    _, jnp = require_jax()
    y_true = jnp.asarray(y)
    y_pred = forward(params, X, hidden_activation=hidden_activation, output_activation=output_activation)
    eps = 1e-9
    if loss_name == "mse":
        return jnp.mean((y_pred - y_true) ** 2)
    if loss_name == "binary_cross_entropy":
        p = jnp.clip(y_pred, eps, 1.0 - eps)
        return -jnp.mean(y_true * jnp.log(p) + (1.0 - y_true) * jnp.log(1.0 - p))
    if loss_name == "softmax_cross_entropy":
        p = jnp.clip(y_pred, eps, 1.0)
        return -jnp.mean(jnp.sum(y_true * jnp.log(p), axis=1))
    raise ValueError(f"Unknown loss '{loss_name}'")


def gradients(
    params,
    X,
    y,
    loss_name: str = "binary_cross_entropy",
    hidden_activation: str = "relu",
    output_activation: str = "sigmoid",
):
    jax, _ = require_jax()

    def wrapped(p):
        return loss(
            p,
            X,
            y,
            loss_name=loss_name,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

    return jax.grad(wrapped)(params)
