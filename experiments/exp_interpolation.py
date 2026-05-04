"""Weight-space interpolation between independently trained networks."""

from __future__ import annotations

import csv

from experiments.common import FIGURES, TABLES, ensure_results_dirs, load_binary_split
from src.landscape import interpolation_curve
from src.mlp_numpy import MLPNumPy
from src.training import train
from src.visualization import write_multi_line_plot


def run(epochs=400, n_samples=None):
    ensure_results_dirs()
    X_train, X_test, y_train, y_test = load_binary_split("banknote", n_samples=n_samples, seed=15)
    trained = []
    for seed in [21, 84]:
        model = MLPNumPy([4, 16, 16, 1], hidden_activation="relu", output_activation="sigmoid", initialization="he", seed=seed)
        train(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            optimizer_name="adam",
            learning_rate=0.01,
            epochs=epochs,
            batch_size=64,
            log_every=20,
            seed=seed,
        )
        trained.append(model.copy_params())

    probe = MLPNumPy([4, 16, 16, 1], hidden_activation="relu", output_activation="sigmoid", initialization="he", seed=999)
    curve = interpolation_curve(probe, trained[0], trained[1], X_test, y_test, "binary_cross_entropy", points=51)

    with (TABLES / "interpolation_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "loss"])
        writer.writeheader()
        for alpha, loss in zip(curve["alpha"], curve["loss"]):
            writer.writerow({"alpha": alpha, "loss": loss})

    write_multi_line_plot(
        [{"label": "linear path", "x": curve["alpha"], "y": curve["loss"]}],
        FIGURES / "interpolation_loss_curve.svg",
        "Weight-Space Interpolation: Test Loss",
        x_label="alpha",
        y_label="test loss",
    )
    return curve


if __name__ == "__main__":
    run()
