"""Validate NumPy backpropagation with finite differences and optional JAX autodiff."""

from __future__ import annotations

from experiments.common import TABLES, ensure_results_dirs, save_rows_csv
from src.datasets import make_moons
from src.gradient_check import finite_difference_check, jax_autodiff_check
from src.mlp_numpy import MLPNumPy


def run():
    ensure_results_dirs()
    X, y = make_moons(n_samples=48, noise=0.1, seed=3)
    settings = [
        ("mse", "tanh", "sigmoid"),
        ("binary_cross_entropy", "tanh", "sigmoid"),
    ]
    rows = []
    detail_rows = []
    for loss_name, hidden_activation, output_activation in settings:
        model = MLPNumPy([2, 3, 1], hidden_activation=hidden_activation, output_activation=output_activation, initialization="xavier", seed=3)
        finite = finite_difference_check(model, X[:16], y[:16], loss_name, num_checks=40, seed=3)
        rows.append(
            {
                "method": "finite_difference",
                "loss": loss_name,
                "hidden_activation": hidden_activation,
                "max_relative_error": finite["max_relative_error"],
                "status": "passed" if finite["max_relative_error"] < 1e-4 else "review",
            }
        )
        for check in finite["checks"]:
            detail = {"method": "finite_difference", "loss": loss_name}
            detail.update(check)
            detail_rows.append(detail)

        try:
            jax_result = jax_autodiff_check(model, X[:16], y[:16], loss_name)
            rows.append(
                {
                    "method": "jax_autodiff",
                    "loss": loss_name,
                    "hidden_activation": hidden_activation,
                    "max_relative_error": jax_result["max_relative_error"],
                    "status": "passed" if jax_result["max_relative_error"] < 1e-4 else "review",
                }
            )
        except RuntimeError as exc:
            rows.append(
                {
                    "method": "jax_autodiff",
                    "loss": loss_name,
                    "hidden_activation": hidden_activation,
                    "max_relative_error": "",
                    "status": f"skipped: {exc}",
                }
            )

    save_rows_csv(rows, TABLES / "gradient_validation.csv")
    save_rows_csv(detail_rows, TABLES / "gradient_validation_details.csv")
    return rows


if __name__ == "__main__":
    run()

