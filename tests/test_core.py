"""Basic smoke tests for the NumPy framework."""

from __future__ import annotations

from src.datasets import load_banknote_authentication, make_moons, standardize_train_test, train_test_split
from src.gradient_check import finite_difference_check
from src.mlp_numpy import MLPNumPy
from src.training import train


def test_gradient_check_binary_cross_entropy():
    X, y = make_moons(n_samples=32, noise=0.1, seed=1)
    model = MLPNumPy([2, 3, 1], hidden_activation="tanh", output_activation="sigmoid", initialization="xavier", seed=1)
    result = finite_difference_check(model, X[:12], y[:12], "binary_cross_entropy", num_checks=20, seed=1)
    assert result["max_relative_error"] < 1e-4


def test_training_reduces_loss():
    X, y = make_moons(n_samples=160, noise=0.2, seed=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_fraction=0.25, seed=2)
    X_train, X_test = standardize_train_test(X_train, X_test)
    model = MLPNumPy([2, 8, 8, 1], hidden_activation="relu", output_activation="sigmoid", initialization="he", seed=2)
    history = train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer_name="adam",
        learning_rate=0.01,
        epochs=80,
        batch_size=32,
        log_every=10,
        seed=2,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["test_accuracy"][-1] >= 0.75


def test_load_real_banknote_dataset():
    X, y = load_banknote_authentication(download=False)
    assert X.shape == (1372, 4)
    assert y.shape == (1372, 1)
    assert set(y.ravel().astype(int)) == {0, 1}
