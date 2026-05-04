"""Compare initialization schemes."""

from __future__ import annotations

from experiments.common import FIGURES, TABLES, ensure_results_dirs, load_binary_split, save_rows_csv, summarize_run
from src.mlp_numpy import MLPNumPy
from src.training import train
from src.visualization import write_multi_line_plot


def run(epochs=350, n_samples=None):
    ensure_results_dirs()
    X_train, X_test, y_train, y_test = load_binary_split("banknote", n_samples=n_samples, seed=14)
    initializations = ["small_random", "xavier", "he"]
    loss_series = []
    grad_series = []
    rows = []
    for init in initializations:
        model = MLPNumPy([4, 16, 16, 1], hidden_activation="relu", output_activation="sigmoid", initialization=init, seed=14)
        history = train(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            optimizer_name="adam",
            learning_rate=0.01,
            epochs=epochs,
            batch_size=64,
            log_every=10,
            seed=14,
        )
        loss_series.append({"label": init, "x": history["epoch"], "y": history["test_loss"]})
        grad_series.append({"label": init, "x": history["epoch"], "y": history["gradient_norm"]})
        rows.append(summarize_run(init, model, history, X_train, y_train, X_test, y_test, "binary_cross_entropy"))

    write_multi_line_plot(loss_series, FIGURES / "initialization_loss_curves.svg", "Initialization Comparison: Test Loss", y_label="test loss")
    write_multi_line_plot(grad_series, FIGURES / "initialization_gradient_norms.svg", "Initialization Comparison: Gradient Norm", y_label="gradient norm")
    save_rows_csv(rows, TABLES / "initialization_comparison.csv")
    return rows


if __name__ == "__main__":
    run()
