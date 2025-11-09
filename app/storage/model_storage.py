"""Storage for trained models."""

import json
from pathlib import Path
from typing import Any

import joblib

from app.models.base import BaseMLModel
from app.utils.logger import logger


class ModelStorage:
    """Storage for managing trained models."""

    def __init__(self, storage_dir: str = "models") -> None:
        """
        Initialize model storage.

        Args:
            storage_dir: Directory path for storing models
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._models: dict[str, BaseMLModel] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self._metadata = json.load(f)

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def save_model(
        self, model_name: str, model: BaseMLModel, model_type: str
    ) -> None:
        """
        Save a trained model.

        Args:
            model_name: Unique identifier for the model
            model: Trained model instance
            model_type: Type of the model

        Raises:
            ValueError: If model name already exists
        """
        if model_name in self._models:
            raise ValueError(f"Model with name '{model_name}' already exists")

        model_path = self.storage_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        self._models[model_name] = model
        self._metadata[model_name] = {
            "type": model_type,
            "hyperparameters": model.hyperparameters,
        }
        self._save_metadata()

        logger.info(f"Model '{model_name}' saved successfully")

    def load_model(self, model_name: str) -> BaseMLModel:
        """
        Load a trained model by name.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model not found
        """
        if model_name in self._models:
            return self._models[model_name]

        model_path = self.storage_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise ValueError(f"Model '{model_name}' not found")

        model = joblib.load(model_path)
        self._models[model_name] = model

        logger.info(f"Model '{model_name}' loaded successfully")
        return model

    def delete_model(self, model_name: str) -> None:
        """
        Delete a trained model.

        Args:
            model_name: Name of the model to delete

        Raises:
            ValueError: If model not found
        """
        if model_name not in self._models and model_name not in self._metadata:
            raise ValueError(f"Model '{model_name}' not found")

        model_path = self.storage_dir / f"{model_name}.joblib"
        if model_path.exists():
            model_path.unlink()

        self._models.pop(model_name, None)
        self._metadata.pop(model_name, None)
        self._save_metadata()

        logger.info(f"Model '{model_name}' deleted successfully")

    def list_models(self) -> list[str]:
        """
        Get list of all stored model names.

        Returns:
            List of model names
        """
        return list(self._metadata.keys())

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """
        Get information about a stored model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information

        Raises:
            ValueError: If model not found
        """
        if model_name not in self._metadata:
            raise ValueError(f"Model '{model_name}' not found")

        return {
            "name": model_name,
            "type": self._metadata[model_name]["type"],
            "hyperparameters": self._metadata[model_name]["hyperparameters"],
        }

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists.

        Args:
            model_name: Name of the model

        Returns:
            True if model exists, False otherwise
        """
        return model_name in self._metadata

