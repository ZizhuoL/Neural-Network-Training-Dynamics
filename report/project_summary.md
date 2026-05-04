# Project Summary: Neural Network Optimization & Training Dynamics

## Executive Summary

This project turns a basic neural-network assignment into a controlled implementation and training-dynamics study. The core artifact is a configurable NumPy MLP framework, supported by reusable experiment scripts, validation tables, loss/accuracy plots, PCA decision-boundary visualizations, interpolation curves, and local Hessian sharpness estimates.

The primary experiments use the real UCI Banknote Authentication dataset. The emphasis is not state-of-the-art accuracy. The emphasis is understanding how the model trains, how implementation correctness is validated, and how architecture and optimizer choices affect convergence.

## Dataset

The main dataset is UCI Banknote Authentication, cited as:

Lohweg, V. (2012). Banknote Authentication [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57

It contains 1,372 records with four continuous features extracted from banknote images and a binary target. The project keeps synthetic datasets available for optional toy diagnostics, but the generated experiment tables and figures use the real banknote dataset as the primary evidence.

## Implementation

The MLP accepts a general `layer_sizes` list, so `[2, 4, 1]`, `[2, 16, 16, 1]`, and `[2, 32, 32, 16, 1]` use the same forward and backward code. Parameters are stored as a list of layer dictionaries:

```python
{"W": W, "b": b}
```

The forward pass stores activations and pre-activations in a cache. The backward pass loops through the same parameter list in reverse and asserts that every gradient has the same shape as its parameter.

Supported training components include:

- sigmoid, tanh, ReLU, and Leaky ReLU
- MSE, binary cross-entropy, and softmax cross-entropy
- small random, Xavier, and He initialization
- gradient descent, momentum, and Adam
- loss, accuracy, gradient norm, and parameter norm diagnostics

## Gradient Validation

Backpropagation was validated with centered finite differences on small networks. In the generated local run:

- MSE max relative error: about `1.3e-9`
- binary cross-entropy max relative error: about `1.5e-10`

The code also includes optional JAX forward/loss/gradient support. JAX was not installed in the local Codex runtime, so the generated validation table marks JAX autodiff checks as skipped. Installing `requirements.txt` enables the JAX validation path.

## Experiments

The experiment suite uses deterministic train/test splits of the real UCI Banknote Authentication dataset. Each experiment changes one major factor while keeping the remaining setup fixed.

### Activation Comparison

The fixed architecture `[4, 16, 16, 1]` was trained with sigmoid, tanh, ReLU, and Leaky ReLU hidden activations. The generated results show that all four activations can solve this highly separable dataset, while the loss curves and gradient norms still differ.

### Learning-Rate Sweep

The learning-rate sweep used gradient descent with rates from `0.0001` through `1.0`. Small learning rates trained slowly, while larger rates converged faster on this problem. The gradient-norm curves provide diagnostic evidence for stability and convergence behavior.

### Architecture Comparison

The project compares:

- `[4, 4, 1]`
- `[4, 16, 1]`
- `[4, 16, 16, 1]`
- `[4, 32, 32, 16, 1]`

Because the dataset has four features, the decision-boundary SVGs are shown as PCA projections. They are useful visual diagnostics, but the quantitative train/test metrics are the primary comparison.

### Optimizer Comparison

Gradient descent, momentum, and Adam are compared under a fixed Leaky ReLU architecture. Adam provides a strong practical baseline, while GD and momentum remain useful because their behavior is easier to connect directly to optimization dynamics.

### Initialization Comparison

Small random, Xavier, and He initialization are compared for a ReLU network. The experiment tracks both loss and gradient norms so the discussion can connect initialization to gradient flow rather than only final accuracy.

## Loss-Landscape Analysis

Two networks with the same architecture are trained from different random seeds. The interpolation experiment evaluates:

```python
params_alpha = (1 - alpha) * params_A + alpha * params_B
```

The resulting curve is saved to `results/tables/interpolation_curve.csv` and plotted in `results/figures/interpolation_loss_curve.svg`. This should be described as empirical evidence about the straight-line path between two trained solutions, not as a general theorem about neural-network loss landscapes.

## Hessian Sharpness

The project estimates the largest local Hessian eigenvalue with Hessian-vector products and power iteration. In the local runtime, the HVP is computed through finite differences of gradients. The code also includes a JAX HVP implementation using `jax.grad`, `jax.jvp`, and `ravel_pytree` for environments with JAX installed.

The saved table reports:

- model setting
- train loss
- test accuracy
- final gradient norm
- top Hessian eigenvalue estimate
- HVP method

These eigenvalues are local sharpness estimates. They should be interpreted alongside train loss, test accuracy, and optimizer behavior.

## Limitations

- The project is intentionally small-scale and uses synthetic datasets by default.
- JAX checks require installing the optional dependency set.
- Hessian sharpness is a local diagnostic, not a complete theory of generalization.
- The project is not a PyTorch clone and does not aim for large-scale benchmarking.

## Final Takeaway

The finished project supports a defensible resume claim: it demonstrates from-scratch neural-network implementation, gradient validation, optimizer and architecture comparisons, training diagnostics, empirical interpolation analysis, and local sharpness estimation.
