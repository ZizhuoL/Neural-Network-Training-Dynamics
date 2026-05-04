"""Shared helpers for experiment scripts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.datasets import get_binary_dataset, standardize_train_test, train_test_split
from src.metrics import gradient_norm


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"


def ensure_results_dirs():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def load_binary_split(dataset="banknote", n_samples=None, seed=0):
    X, y = get_binary_dataset(dataset, n_samples=n_samples or 1000, seed=seed)
    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))[:n_samples]
        X = X[idx]
        y = y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_fraction=0.25, seed=seed)
    X_train, X_test = standardize_train_test(X_train, X_test)
    return X_train, X_test, y_train, y_test


def save_history_csv(history, path):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history.keys()))
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({key: history[key][i] for key in history})


def save_rows_csv(rows, path):
    path = Path(path)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(label, model, history, X_train, y_train, X_test, y_test, loss_name):
    grads = model.gradients(X_train[: min(128, len(X_train))], y_train[: min(128, len(y_train))], loss_name)
    return {
        "setting": label,
        "train_loss": history["train_loss"][-1],
        "test_loss": history["test_loss"][-1],
        "train_accuracy": history["train_accuracy"][-1],
        "test_accuracy": history["test_accuracy"][-1],
        "final_gradient_norm": gradient_norm(grads),
        "epochs": history["epoch"][-1],
    }
