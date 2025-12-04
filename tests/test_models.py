"""Simple tests for ML models."""

import numpy as np
import pytest

from app.models.implementations import (
    LinearRegressionModel,
    LogisticRegressionModel,
    RandomForestModel,
)
from app.models.registry import ModelRegistry


def test_linear_regression_train_predict():
    """Test Linear Regression training and prediction."""
    model = LinearRegressionModel(fit_intercept=True)
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])

    model.train(X, y)
    assert model.is_trained is True

    predictions = model.predict(np.array([[4]]))
    assert len(predictions) == 1
    assert predictions[0] == pytest.approx(8, abs=0.1)


def test_logistic_regression_train_predict():
    """Test Logistic Regression training and prediction."""
    model = LogisticRegressionModel(C=1.0, max_iter=100)
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model.train(X, y)
    assert model.is_trained is True

    predictions = model.predict(np.array([[1.5, 2.5]]))
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]


def test_random_forest_train_predict():
    """Test Random Forest training and prediction."""
    model = RandomForestModel(n_estimators=10, random_state=42)
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 1, 1, 2, 2])

    model.train(X, y)
    assert model.is_trained is True

    predictions = model.predict(np.array([[2, 3]]))
    assert len(predictions) == 1
    assert predictions[0] in [0, 1, 2]


def test_model_registry():
    """Test ModelRegistry functionality."""
    model_types = ModelRegistry.list_model_types()
    assert len(model_types) >= 2
    assert "linear_regression" in model_types

    model_class = ModelRegistry.get_model_class("linear_regression")
    assert model_class == LinearRegressionModel

    info = ModelRegistry.get_model_info("linear_regression")
    assert info["name"] == "linear_regression"
    assert "hyperparameters" in info

