"""Small deterministic datasets used by the experiment scripts."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np


BANKNOTE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
BANKNOTE_COLUMNS = ["variance", "skewness", "curtosis", "entropy", "class"]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    test_size = int(round(len(X) * test_fraction))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def make_moons(n_samples: int = 1000, noise: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.uniform(0.0, np.pi, n_outer)
    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])

    theta_inner = rng.uniform(0.0, np.pi, n_inner)
    inner = np.column_stack([1.0 - np.cos(theta_inner), 0.5 - np.sin(theta_inner)])

    X = np.vstack([outer, inner])
    y = np.vstack([np.zeros((n_outer, 1)), np.ones((n_inner, 1))])
    X += rng.normal(0.0, noise, X.shape)
    return X.astype(float), y.astype(float)


def make_circles(n_samples: int = 1000, noise: float = 0.08, factor: float = 0.45, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.uniform(0.0, 2.0 * np.pi, n_outer)
    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])

    theta_inner = rng.uniform(0.0, 2.0 * np.pi, n_inner)
    inner = factor * np.column_stack([np.cos(theta_inner), np.sin(theta_inner)])

    X = np.vstack([outer, inner])
    y = np.vstack([np.zeros((n_outer, 1)), np.ones((n_inner, 1))])
    X += rng.normal(0.0, noise, X.shape)
    return X.astype(float), y.astype(float)


def make_toy_assignment(n_samples: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n_samples, 2))
    boundary = np.sin(3.0 * X[:, [0]]) + 0.5 * X[:, [1]]
    y = (boundary > 0.0).astype(float)
    X += rng.normal(0.0, 0.05, X.shape)
    return X.astype(float), y.astype(float)


def get_binary_dataset(name: str, n_samples: int = 1000, seed: int = 0):
    if name == "moons":
        return make_moons(n_samples=n_samples, noise=0.2, seed=seed)
    if name == "circles":
        return make_circles(n_samples=n_samples, noise=0.08, seed=seed)
    if name == "toy":
        return make_toy_assignment(n_samples=n_samples, seed=seed)
    if name == "banknote":
        return load_banknote_authentication()
    raise ValueError(f"Unknown dataset '{name}'")


def download_banknote_authentication(path: str | Path) -> Path:
    """Download the UCI Banknote Authentication dataset to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = urlopen(BANKNOTE_URL, timeout=30).read().decode("utf-8")
    path.write_text(",".join(BANKNOTE_COLUMNS) + "\n" + raw, encoding="utf-8")
    return path


def load_banknote_authentication(path: str | Path | None = None, download: bool = True):
    """Load the real UCI Banknote Authentication dataset.

    The dataset contains 1,372 records with four continuous features extracted
    from banknote images and a binary class target.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "banknote_authentication.csv"
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing dataset file: {path}")
        download_banknote_authentication(path)

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    X = data[:, :4].astype(float)
    y = data[:, [4]].astype(float)
    return X, y
