"""Learning-rate sweep with a fixed architecture and activation."""

from __future__ import annotations

from experiments.common import FIGURES, TABLES, ensure_results_dirs, load_binary_split, save_rows_csv, summarize_run
from src.mlp_numpy import MLPNumPy
from src.training import train
from src.visualization import write_multi_line_plot


def run(epochs=300, n_samples=None):
    ensure_results_dirs()
    X_train, X_test, y_train, y_test = load_binary_split("banknote", n_samples=n_samples, seed=11)
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
    loss_series = []
    grad_series = []
    rows = []
    for lr in learning_rates:
        model = MLPNumPy([4, 16, 16, 1], hidden_activation="relu", output_activation="sigmoid", initialization="he", seed=9)
        history = train(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            optimizer_name="gd",
            learning_rate=lr,
            epochs=epochs,
            batch_size=64,
            log_every=10,
            seed=9,
        )
        label = f"lr={lr:g}"
        loss_series.append({"label": label, "x": history["epoch"], "y": history["test_loss"]})
        grad_series.append({"label": label, "x": history["epoch"], "y": history["gradient_norm"]})
        rows.append(summarize_run(label, model, history, X_train, y_train, X_test, y_test, "binary_cross_entropy"))

    write_multi_line_plot(loss_series, FIGURES / "learning_rate_loss_curves.svg", "Learning-Rate Sweep: Test Loss", y_label="test loss")
    write_multi_line_plot(grad_series, FIGURES / "learning_rate_gradient_norms.svg", "Learning-Rate Sweep: Gradient Norm", y_label="gradient norm")
    save_rows_csv(rows, TABLES / "learning_rate_sweep.csv")
    return rows


if __name__ == "__main__":
    run()
