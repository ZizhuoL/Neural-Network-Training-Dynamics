"""Run the full experiment suite and generate tables/figures."""

from __future__ import annotations

import argparse

from experiments.common import TABLES, ensure_results_dirs, save_rows_csv
from experiments.exp_activation_comparison import run as run_activation
from experiments.exp_architecture_comparison import run as run_architecture
from experiments.exp_hessian_sharpness import run as run_hessian
from experiments.exp_gradient_validation import run as run_gradient_validation
from experiments.exp_initialization_comparison import run as run_initialization
from experiments.exp_interpolation import run as run_interpolation
from experiments.exp_learning_rate_sweep import run as run_learning_rate
from experiments.exp_optimizer_comparison import run as run_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs/samples for a fast smoke run.")
    args = parser.parse_args()

    ensure_results_dirs()
    if args.quick:
        epochs = 80
        n_samples = 500
        hessian_epochs = 80
    else:
        epochs = 350
        n_samples = None
        hessian_epochs = 300

    print("Running gradient validation...")
    validation_rows = run_gradient_validation()
    print("Running activation comparison...")
    activation_rows = run_activation(epochs=epochs, n_samples=n_samples)
    print("Running learning-rate sweep...")
    lr_rows = run_learning_rate(epochs=max(epochs - 50, 60), n_samples=n_samples)
    print("Running architecture comparison...")
    architecture_rows = run_architecture(epochs=epochs, n_samples=n_samples)
    print("Running optimizer comparison...")
    optimizer_rows = run_optimizer(epochs=epochs, n_samples=n_samples)
    print("Running initialization comparison...")
    init_rows = run_initialization(epochs=epochs, n_samples=n_samples)
    print("Running interpolation analysis...")
    run_interpolation(epochs=epochs, n_samples=n_samples)
    print("Running Hessian sharpness analysis...")
    hessian_samples = 800 if n_samples is None else max(300, n_samples // 2)
    hessian_rows = run_hessian(epochs=hessian_epochs, n_samples=hessian_samples)

    summary_rows = []
    for group, rows in [
        ("activation", activation_rows),
        ("gradient_validation", validation_rows),
        ("learning_rate", lr_rows),
        ("architecture", architecture_rows),
        ("optimizer", optimizer_rows),
        ("initialization", init_rows),
        ("hessian", hessian_rows),
    ]:
        for row in rows:
            out = {"experiment": group}
            out.update(row)
            summary_rows.append(out)
    save_rows_csv(summary_rows, TABLES / "all_experiment_summary.csv")
    print("Done. Results are in results/figures and results/tables.")


if __name__ == "__main__":
    main()
