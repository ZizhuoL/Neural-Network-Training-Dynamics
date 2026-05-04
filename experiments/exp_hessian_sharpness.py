"""Estimate local sharpness via top Hessian eigenvalue."""

from __future__ import annotations

from experiments.common import FIGURES, TABLES, ensure_results_dirs, load_binary_split, save_rows_csv, summarize_run
from src.hessian import estimate_top_hessian_eigenvalue_numpy
from src.mlp_numpy import MLPNumPy
from src.training import train
from src.visualization import write_bar_chart


def run(epochs=300, n_samples=600):
    ensure_results_dirs()
    X_train, X_test, y_train, y_test = load_binary_split("moons", n_samples=n_samples, seed=16)
    settings = [
        ("ReLU, GD, He init", "relu", "gd", 0.05, "he"),
        ("ReLU, Momentum, He init", "relu", "momentum", 0.08, "he"),
        ("ReLU, Adam, He init", "relu", "adam", 0.01, "he"),
        ("Tanh, Adam, Xavier init", "tanh", "adam", 0.01, "xavier"),
        ("Leaky ReLU, Adam, He init", "leaky_relu", "adam", 0.01, "he"),
    ]
    rows = []
    labels = []
    eigenvalues = []
    for label, activation, optimizer, lr, init in settings:
        model = MLPNumPy([2, 12, 12, 1], hidden_activation=activation, output_activation="sigmoid", initialization=init, seed=16)
        history = train(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            optimizer_name=optimizer,
            learning_rate=lr,
            epochs=epochs,
            batch_size=64,
            log_every=20,
            seed=16,
        )
        eig = estimate_top_hessian_eigenvalue_numpy(
            model,
            X_train[:160],
            y_train[:160],
            "binary_cross_entropy",
            num_iters=16,
            seed=16,
        )
        row = summarize_run(label, model, history, X_train, y_train, X_test, y_test, "binary_cross_entropy")
        row["top_hessian_eigenvalue"] = eig
        row["hvp_method"] = "finite_difference_hvp_numpy"
        rows.append(row)
        labels.append(label)
        eigenvalues.append(max(eig, 0.0))

    save_rows_csv(rows, TABLES / "hessian_sharpness.csv")
    write_bar_chart(labels, eigenvalues, FIGURES / "hessian_top_eigenvalues.svg", "Top Hessian Eigenvalue Estimates", y_label="estimated lambda max")
    return rows


if __name__ == "__main__":
    run()

