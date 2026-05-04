"""Top Hessian eigenvalue estimation via Hessian-vector products."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def flatten_params(params):
    flat_parts = []
    spec = []
    for layer in params:
        layer_spec = {}
        for name in ("W", "b"):
            arr = layer[name]
            layer_spec[name] = (arr.shape, arr.size)
            flat_parts.append(arr.ravel())
        spec.append(layer_spec)
    return np.concatenate(flat_parts), spec


def unflatten_params(theta, spec):
    params = []
    pos = 0
    for layer_spec in spec:
        layer = {}
        for name in ("W", "b"):
            shape, size = layer_spec[name]
            layer[name] = theta[pos : pos + size].reshape(shape).copy()
            pos += size
        params.append(layer)
    return params


def flatten_grads(grads):
    return np.concatenate([part.ravel() for grad in grads for part in (grad["dW"], grad["db"])])


def numpy_hvp(model, theta, vector, spec, X, y, loss_name: str, epsilon: float = 1e-4):
    original = deepcopy(model.params)
    try:
        model.set_params(unflatten_params(theta + epsilon * vector, spec))
        grad_plus = flatten_grads(model.gradients(X, y, loss_name))
        model.set_params(unflatten_params(theta - epsilon * vector, spec))
        grad_minus = flatten_grads(model.gradients(X, y, loss_name))
    finally:
        model.set_params(original)
    return (grad_plus - grad_minus) / (2.0 * epsilon)


def estimate_top_hessian_eigenvalue_numpy(
    model,
    X,
    y,
    loss_name: str,
    num_iters: int = 30,
    seed: int = 0,
    epsilon: float = 1e-4,
):
    """Estimate largest local curvature using finite-difference HVPs."""
    theta, spec = flatten_params(model.params)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=theta.shape)
    v = v / (np.linalg.norm(v) + 1e-12)

    eigenvalue = 0.0
    for _ in range(num_iters):
        hv = numpy_hvp(model, theta, v, spec, X, y, loss_name, epsilon=epsilon)
        norm = np.linalg.norm(hv)
        if norm < 1e-12:
            return 0.0
        v = hv / norm
        eigenvalue = float(v @ numpy_hvp(model, theta, v, spec, X, y, loss_name, epsilon=epsilon))
    return eigenvalue


def estimate_top_hessian_eigenvalue_jax(*_args, **_kwargs):
    """Estimate largest Hessian eigenvalue with JAX jvp of the gradient."""
    try:
        import jax
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree
        from .mlp_jax import loss as jax_loss
    except ImportError as exc:
        raise RuntimeError(
            "JAX is not installed in this environment. Install requirements.txt to use "
            "jax.grad, jax.jvp, and ravel_pytree for HVP power iteration."
        ) from exc

    model, X, y, loss_name = _args[:4]
    num_iters = _kwargs.get("num_iters", 30)
    seed = _kwargs.get("seed", 0)
    theta, unravel = ravel_pytree(model.params)

    def loss_flat(theta_flat):
        params = unravel(theta_flat)
        return jax_loss(
            params,
            X,
            y,
            loss_name=loss_name,
            hidden_activation=model.hidden_activation_name,
            output_activation=model.output_activation,
        )

    grad_fn = jax.grad(loss_flat)

    def hvp(theta_flat, vector):
        _, hv = jax.jvp(grad_fn, (theta_flat,), (vector,))
        return hv

    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, theta.shape)
    v = v / (jnp.linalg.norm(v) + 1e-12)
    eigenvalue = 0.0
    for _ in range(num_iters):
        hv = hvp(theta, v)
        v = hv / (jnp.linalg.norm(hv) + 1e-12)
        eigenvalue = jnp.dot(v, hvp(theta, v))
    return float(eigenvalue)
