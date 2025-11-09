"""Concrete implementations of ML models."""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.models.base import BaseMLModel


class LinearRegressionModel(BaseMLModel):
    """Linear Regression model for regression tasks."""

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        fit_intercept = hyperparameters.get("fit_intercept", True)
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    @staticmethod
    def get_hyperparameter_info() -> dict[str, str]:
        return {"fit_intercept": "bool: whether to calculate intercept (default: true)"}

    @staticmethod
    def get_description() -> str:
        return "Linear Regression model for regression tasks"


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression model for binary classification."""

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        C = hyperparameters.get("C", 1.0)
        max_iter = hyperparameters.get("max_iter", 100)
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    @staticmethod
    def get_hyperparameter_info() -> dict[str, str]:
        return {
            "C": "float: inverse of regularization strength (default: 1.0)",
            "max_iter": "int: maximum iterations (default: 100)",
        }

    @staticmethod
    def get_description() -> str:
        return "Logistic Regression model for binary classification"


class RandomForestModel(BaseMLModel):
    """Random Forest model for classification tasks."""

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", None)
        random_state = hyperparameters.get("random_state", 42)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    @staticmethod
    def get_hyperparameter_info() -> dict[str, str]:
        return {
            "n_estimators": "int: number of trees in the forest (default: 100)",
            "max_depth": "int: maximum depth of trees (default: None)",
            "random_state": "int: random seed (default: 42)",
        }

    @staticmethod
    def get_description() -> str:
        return "Random Forest model for classification tasks"

