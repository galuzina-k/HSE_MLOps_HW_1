import os
from typing import Any

import mlflow
import numpy as np

from app.models.base import BaseMLModel
from app.utils.logger import logger


class MLflowTracker:
    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.enabled = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"

        if self.enabled:
            logger.info(f"MLflow tracking enabled at {self.tracking_uri}")
        else:
            logger.info("MLflow tracking disabled")

    def start_run(self, experiment_name: str = "default"):
        if not self.enabled:
            return None

        try:
            mlflow.set_experiment(experiment_name)
            return mlflow.start_run()
        except Exception as e:
            logger.warning(f"Could not start MLflow run: {e}")
            return None

    def log_params(self, params: dict[str, Any]):
        if not self.enabled:
            return

        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Could not log params to MLflow: {e}")

    def log_metrics(self, metrics: dict[str, float]):
        if not self.enabled:
            return

        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"Could not log metrics to MLflow: {e}")

    def log_model(self, model: BaseMLModel, model_name: str):
        if not self.enabled:
            return

        try:
            mlflow.sklearn.log_model(model.model, model_name)
        except Exception as e:
            logger.warning(f"Could not log model to MLflow: {e}")

    def end_run(self):
        if not self.enabled:
            return

        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"Could not end MLflow run: {e}")

    def track_training(
        self,
        model_name: str,
        model_type: str,
        model: BaseMLModel,
        hyperparameters: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        if not self.enabled:
            return

        run = self.start_run(experiment_name="mlops-training")
        if run is None:
            return

        try:
            self.log_params({
                "model_name": model_name,
                "model_type": model_type,
                **hyperparameters,
            })

            self.log_metrics({
                "n_samples": len(X_train),
                "n_features": X_train.shape[1] if len(X_train.shape) > 1 else 1,
            })

            self.log_model(model, model_name)

        finally:
            self.end_run()

