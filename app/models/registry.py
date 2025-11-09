"""Model registry for managing available model types."""

from app.models.base import BaseMLModel
from app.models.implementations import (
    LinearRegressionModel,
    LogisticRegressionModel,
    RandomForestModel,
)


class ModelRegistry:
    """Registry for managing available model types."""

    _models: dict[str, type[BaseMLModel]] = {
        "linear_regression": LinearRegressionModel,
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
    }

    @classmethod
    def get_model_class(cls, model_type: str) -> type[BaseMLModel]:
        """
        Get model class by type name.

        Args:
            model_type: Model type identifier

        Returns:
            Model class

        Raises:
            ValueError: If model type is not found
        """
        if model_type not in cls._models:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {list(cls._models.keys())}"
            )
        return cls._models[model_type]

    @classmethod
    def list_model_types(cls) -> list[str]:
        """
        Get list of available model types.

        Returns:
            List of model type identifiers
        """
        return list(cls._models.keys())

    @classmethod
    def get_model_info(cls, model_type: str) -> dict[str, str | dict[str, str]]:
        """
        Get information about a model type.

        Args:
            model_type: Model type identifier

        Returns:
            Dictionary with model information
        """
        model_class = cls.get_model_class(model_type)
        return {
            "name": model_type,
            "description": model_class.get_description(),
            "hyperparameters": model_class.get_hyperparameter_info(),
        }

    @classmethod
    def get_all_models_info(cls) -> list[dict[str, str | dict[str, str]]]:
        """
        Get information about all available model types.

        Returns:
            List of dictionaries with model information
        """
        return [cls.get_model_info(model_type) for model_type in cls._models]

