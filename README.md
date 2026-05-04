# Neural Network Optimization & Training Dynamics

This project implements a configurable multilayer perceptron in NumPy and uses it to study how training behavior changes under different activations, learning rates, architectures, initializations, and optimizers. The primary experiments use the real UCI Banknote Authentication dataset. The project also includes empirical loss-landscape analysis through weight-space interpolation and local sharpness estimation with Hessian-vector products.

The project is intentionally scoped as an implementation and analysis project, not a claim of new loss-landscape theory.

## What Is Implemented

- Configurable MLP with arbitrary `layer_sizes`, for example `[2, 16, 16, 1]`
- Hidden activations: sigmoid, tanh, ReLU, Leaky ReLU
- Output activations: sigmoid, softmax, linear
- Losses: MSE, binary cross-entropy, softmax cross-entropy
- Initializers: small random, Xavier, He, and activation-aware auto mode
- Optimizers: gradient descent, momentum, Adam
- Training diagnostics: loss, accuracy, gradient norm, parameter norm
- Gradient validation: finite differences and JAX automatic differentiation
- Loss landscape: linear interpolation between independently trained networks
- Sharpness: top Hessian eigenvalue estimation with JAX Hessian-vector products and power iteration
- Reproducible experiment scripts that generate CSV tables and SVG figures
- Real dataset loader for UCI Banknote Authentication, with synthetic datasets retained only as optional diagnostics

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
  data/
    banknote_authentication.csv
    README.md
  tests/
    test_core.py
```

## Dataset

The primary dataset is UCI Banknote Authentication:

- 1,372 instances
- 4 continuous features extracted from banknote images
- binary target
- no missing values
- DOI: `10.24432/C55P57`
- license: CC BY 4.0

Citation:

Lohweg, V. (2012). Banknote Authentication [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57

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

## Full Environment

Install the project dependencies, including JAX:

```bash
pip install -r requirements.txt
```

JAX is used for automatic-differentiation gradient validation and native Hessian-vector products. The generated `gradient_validation.csv` table includes passing JAX autodiff checks, and `hessian_sharpness.csv` reports `jax_jvp_hvp` as the HVP method.

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
- `results/figures/pca_decision_boundary_*.svg`

## Results Snapshot

On the real UCI Banknote Authentication experiments, the task is highly separable: several settings reach 100% test accuracy. This makes the dataset useful for validating implementation and comparing convergence behavior, but less useful as a difficult benchmark. Learning-rate and optimizer comparisons remain informative because they show different convergence speeds, gradient norms, and local sharpness estimates.

Finite-difference gradient checks passed for both MSE and binary cross-entropy with maximum relative errors near `1e-9` to `1e-10` in the local run.

The Hessian table reports top-eigenvalue estimates together with train loss and test accuracy. These values are local curvature estimates only; they should not be described as a complete explanation of generalization.

## Resume Entry

Neural Network Optimization & Training Dynamics | Python, NumPy, JAX, Matplotlib | Spring 2026

- Built a configurable NumPy and JAX MLP framework, including variable network depth and width, sigmoid, tanh, ReLU, and Leaky ReLU activations, MSE and cross-entropy losses, and gradient descent, momentum, and Adam optimizers.
- Verified backpropagation with finite-difference gradient checks and implemented JAX autodiff comparison hooks; benchmarked activation functions, learning rates, initialization schemes, and model capacity on the real UCI Banknote Authentication dataset using convergence curves, accuracy, and gradient-norm diagnostics.
- Studied empirical loss-landscape behavior through weight-space interpolation between independently trained networks and estimated local sharpness using Hessian-vector-product power iteration.
