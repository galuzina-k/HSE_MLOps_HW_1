"""Base class for ML models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseMLModel(ABC):
    """Abstract base class for ML models."""

    def __init__(self, **hyperparameters: Any) -> None:
        """
        Initialize the model with hyperparameters.

        Args:
            **hyperparameters: Model-specific hyperparameters
        """
        self.hyperparameters = hyperparameters
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on provided data.

        Args:
            X: Training features
            y: Training targets
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features for prediction

        Returns:
            Predictions array
        """
        pass

    @staticmethod
    @abstractmethod
    def get_hyperparameter_info() -> dict[str, str]:
        """
        Get information about available hyperparameters.

        Returns:
            Dictionary with hyperparameter names and descriptions
        """
        pass

    @staticmethod
    @abstractmethod
    def get_description() -> str:
        """
        Get model description.

        Returns:
            Model description string
        """
        pass

