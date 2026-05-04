# Neural Network Optimization & Training Dynamics

This project implements a configurable multilayer perceptron in NumPy and uses it to study how training behavior changes under different activations, learning rates, architectures, initializations, and optimizers. It also includes empirical loss-landscape analysis through weight-space interpolation and local sharpness estimation with Hessian-vector products.

The project is intentionally scoped as an implementation and analysis project, not a claim of new loss-landscape theory.

## What Is Implemented

- Configurable MLP with arbitrary `layer_sizes`, for example `[2, 16, 16, 1]`
- Hidden activations: sigmoid, tanh, ReLU, Leaky ReLU
- Output activations: sigmoid, softmax, linear
- Losses: MSE, binary cross-entropy, softmax cross-entropy
- Initializers: small random, Xavier, He, and activation-aware auto mode
- Optimizers: gradient descent, momentum, Adam
- Training diagnostics: loss, accuracy, gradient norm, parameter norm
- Gradient validation: finite differences and optional JAX autodiff
- Loss landscape: linear interpolation between independently trained networks
- Sharpness: top Hessian eigenvalue estimation with finite-difference HVPs locally and JAX HVP hooks when JAX is installed
- Reproducible experiment scripts that generate CSV tables and SVG figures

## Repository Structure

```text
.
  README.md
  requirements.txt
  main.py
  src/
    activations.py
    losses.py
    mlp_numpy.py
    mlp_jax.py
    optimizers.py
    metrics.py
    gradient_check.py
    landscape.py
    hessian.py
    visualization.py
    datasets.py
    training.py
  experiments/
    exp_gradient_validation.py
    exp_activation_comparison.py
    exp_learning_rate_sweep.py
    exp_architecture_comparison.py
    exp_optimizer_comparison.py
    exp_initialization_comparison.py
    exp_interpolation.py
    exp_hessian_sharpness.py
    run_all.py
  results/
    figures/
    tables/
  report/
    project_summary.md
  tests/
    test_core.py
```

## Quick Start

Use the bundled Python environment or any Python environment with NumPy installed.

```bash
python main.py --quick
```

Run the fuller experiment suite:

```bash
python main.py
```

The scripts write figures to `results/figures/` and tables to `results/tables/`.

## Optional Full Environment

The core project runs with NumPy only. For the full planned environment, install:

```bash
pip install -r requirements.txt
```

JAX enables autodiff gradient comparison and native JAX Hessian-vector products. In the local Codex runtime used to generate these artifacts, JAX was not installed, so the saved validation table reports JAX checks as skipped while finite-difference checks pass.

## Generated Evidence

Important generated files:

- `results/tables/gradient_validation.csv`
- `results/tables/activation_comparison.csv`
- `results/tables/learning_rate_sweep.csv`
- `results/tables/architecture_comparison.csv`
- `results/tables/optimizer_comparison.csv`
- `results/tables/initialization_comparison.csv`
- `results/tables/interpolation_curve.csv`
- `results/tables/hessian_sharpness.csv`
- `results/tables/all_experiment_summary.csv`
- `results/figures/activation_loss_curves.svg`
- `results/figures/optimizer_accuracy_curves.svg`
- `results/figures/learning_rate_gradient_norms.svg`
- `results/figures/interpolation_loss_curve.svg`
- `results/figures/hessian_top_eigenvalues.svg`
- `results/figures/decision_boundary_*.svg`

## Results Snapshot

On the generated two-moons experiments, ReLU and Leaky ReLU trained faster than sigmoid and tanh in the fixed architecture comparison. Gradient descent showed strong learning-rate sensitivity, while Adam gave a strong practical baseline. Larger architectures produced more flexible decision boundaries, but the summary table should be interpreted alongside train/test accuracy and gradient norms.

Finite-difference gradient checks passed for both MSE and binary cross-entropy with maximum relative errors near `1e-9` to `1e-10` in the local run.

The Hessian table reports top-eigenvalue estimates together with train loss and test accuracy. These values are local curvature estimates only; they should not be described as a complete explanation of generalization.

## Resume Entry

Neural Network Optimization & Training Dynamics | Python, NumPy, JAX, Matplotlib | Spring 2026

- Built a configurable MLP framework in NumPy with optional JAX validation support, including variable network depth and width, sigmoid, tanh, ReLU, and Leaky ReLU activations, MSE and cross-entropy losses, and gradient descent, momentum, and Adam optimizers.
- Verified backpropagation with finite-difference gradient checks and implemented JAX autodiff comparison hooks; benchmarked activation functions, learning rates, initialization schemes, and model capacity using convergence curves, accuracy, and gradient-norm diagnostics.
- Studied empirical loss-landscape behavior through weight-space interpolation between independently trained networks and estimated local sharpness using Hessian-vector-product power iteration.

