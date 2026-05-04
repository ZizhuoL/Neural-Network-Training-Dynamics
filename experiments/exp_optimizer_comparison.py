"""Compare gradient descent, momentum, and Adam."""

from __future__ import annotations

from experiments.common import FIGURES, TABLES, ensure_results_dirs, load_binary_split, save_rows_csv, summarize_run
from src.mlp_numpy import MLPNumPy
from src.training import train
from src.visualization import write_multi_line_plot


def run(epochs=350, n_samples=None):
    ensure_results_dirs()
    X_train, X_test, y_train, y_test = load_binary_split("banknote", n_samples=n_samples, seed=13)
    settings = [
        ("gd", 0.05),
        ("momentum", 0.08),
        ("adam", 0.01),
    ]
    loss_series = []
    acc_series = []
    rows = []
    for optimizer, lr in settings:
        model = MLPNumPy([4, 16, 16, 1], hidden_activation="leaky_relu", output_activation="sigmoid", initialization="he", seed=13)
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
            log_every=10,
            seed=13,
        )
        label = f"{optimizer}, lr={lr:g}"
        loss_series.append({"label": label, "x": history["epoch"], "y": history["test_loss"]})
        acc_series.append({"label": label, "x": history["epoch"], "y": history["test_accuracy"]})
        rows.append(summarize_run(label, model, history, X_train, y_train, X_test, y_test, "binary_cross_entropy"))

    write_multi_line_plot(loss_series, FIGURES / "optimizer_loss_curves.svg", "Optimizer Comparison: Test Loss", y_label="test loss")
    write_multi_line_plot(acc_series, FIGURES / "optimizer_accuracy_curves.svg", "Optimizer Comparison: Test Accuracy", y_label="test accuracy")
    save_rows_csv(rows, TABLES / "optimizer_comparison.csv")
    return rows


if __name__ == "__main__":
    run()
