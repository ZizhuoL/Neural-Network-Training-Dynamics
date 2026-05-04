"""Compare model capacity and decision-boundary complexity."""

from __future__ import annotations

from experiments.common import FIGURES, TABLES, ensure_results_dirs, load_binary_split, save_rows_csv, summarize_run
from src.mlp_numpy import MLPNumPy
from src.training import train
from src.visualization import write_decision_boundary, write_multi_line_plot


def run(epochs=350, n_samples=900):
    ensure_results_dirs()
    X_train, X_test, y_train, y_test = load_binary_split("moons", n_samples=n_samples, seed=12)
    architectures = [
        [2, 4, 1],
        [2, 16, 1],
        [2, 16, 16, 1],
        [2, 32, 32, 16, 1],
    ]
    series = []
    rows = []
    for layer_sizes in architectures:
        label = "-".join(str(v) for v in layer_sizes)
        model = MLPNumPy(layer_sizes, hidden_activation="relu", output_activation="sigmoid", initialization="he", seed=12)
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
            seed=12,
        )
        series.append({"label": label, "x": history["epoch"], "y": history["test_accuracy"]})
        rows.append(summarize_run(label, model, history, X_train, y_train, X_test, y_test, "binary_cross_entropy"))
        write_decision_boundary(model, X_test, y_test, FIGURES / f"decision_boundary_{label}.svg", f"Decision Boundary: [{', '.join(map(str, layer_sizes))}]")

    write_multi_line_plot(series, FIGURES / "architecture_accuracy_curves.svg", "Architecture Comparison: Test Accuracy", y_label="test accuracy")
    save_rows_csv(rows, TABLES / "architecture_comparison.csv")
    return rows


if __name__ == "__main__":
    run()

