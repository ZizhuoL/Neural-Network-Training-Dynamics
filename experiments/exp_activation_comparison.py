"""Compare hidden activation functions under a fixed architecture."""

from __future__ import annotations

from experiments.common import FIGURES, TABLES, ensure_results_dirs, load_binary_split, save_history_csv, save_rows_csv, summarize_run
from src.mlp_numpy import MLPNumPy
from src.training import train
from src.visualization import write_multi_line_plot


def run(epochs=350, n_samples=900):
    ensure_results_dirs()
    X_train, X_test, y_train, y_test = load_binary_split("moons", n_samples=n_samples, seed=10)
    activations = ["sigmoid", "tanh", "relu", "leaky_relu"]
    series = []
    rows = []
    for activation in activations:
        model = MLPNumPy([2, 16, 16, 1], hidden_activation=activation, output_activation="sigmoid", seed=7)
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
            seed=7,
        )
        save_history_csv(history, TABLES / f"activation_{activation}_history.csv")
        series.append({"label": activation, "x": history["epoch"], "y": history["test_loss"]})
        rows.append(summarize_run(activation, model, history, X_train, y_train, X_test, y_test, "binary_cross_entropy"))

    write_multi_line_plot(series, FIGURES / "activation_loss_curves.svg", "Activation Comparison: Test Loss", y_label="test loss")
    save_rows_csv(rows, TABLES / "activation_comparison.csv")
    return rows


if __name__ == "__main__":
    run()

