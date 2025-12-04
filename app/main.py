"""FastAPI application for ML model training and serving."""

import numpy as np
from fastapi import FastAPI, HTTPException

from app.models.registry import ModelRegistry
from app.schemas.api_schemas import (
    ModelInfo,
    ModelTypeInfo,
    PredictRequest,
    PredictResponse,
    StatusResponse,
    TrainRequest,
    TrainResponse,
)
from app.storage.dataset_storage import DatasetStorage
from app.storage.model_storage import ModelStorage
from app.tracking.mlflow_tracker import MLflowTracker
from app.utils.logger import logger

app = FastAPI(
    title="MLOps API",
    description="REST API for training and serving ML models",
    version="0.1.0",
)

storage = ModelStorage()
dataset_storage = DatasetStorage()
mlflow_tracker = MLflowTracker()


@app.get("/", response_model=StatusResponse)
async def root() -> StatusResponse:
    logger.info("Root endpoint accessed")
    return StatusResponse(status="ok", message="MLOps API is running")


@app.get("/health", response_model=StatusResponse)
async def health_check() -> StatusResponse:
    logger.info("Health check endpoint accessed")
    return StatusResponse(status="healthy", message="Service is operational")


@app.get("/models/types", response_model=list[ModelTypeInfo])
async def list_model_types() -> list[ModelTypeInfo]:
    """
    Get list of available model types.

    Returns:
        List of available model types with their descriptions and hyperparameters
    """
    logger.info("Listing available model types")
    try:
        models_info = ModelRegistry.get_all_models_info()
        return [ModelTypeInfo(**info) for info in models_info]
    except Exception as e:
        logger.error(f"Error listing model types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=list[ModelInfo])
async def list_trained_models() -> list[ModelInfo]:
    """
    Get list of all trained models.

    Returns:
        List of trained models with their information
    """
    logger.info("Listing trained models")
    try:
        model_names = storage.list_models()
        models = [storage.get_model_info(name) for name in model_names]
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        logger.error(f"Error listing trained models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str) -> ModelInfo:
    """
    Get information about a specific trained model.

    Args:
        model_name: Name of the model

    Returns:
        Model information

    Raises:
        HTTPException: If model not found
    """
    logger.info(f"Getting info for model '{model_name}'")
    try:
        model_info = storage.get_model_info(model_name)
        return ModelInfo(**model_info)
    except ValueError as e:
        logger.warning(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/train", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    Train a new ML model. If model exists, deletes it and trains a new one.

    Args:
        request: Training request with model type, name, hyperparameters, and data

    Returns:
        Training response with model information

    Raises:
        HTTPException: If training fails
    """
    logger.info(f"Training model '{request.model_name}' of type '{request.model_type}'")

    try:
        if storage.model_exists(request.model_name):
            logger.info(f"Model '{request.model_name}' exists, deleting and retraining")
            storage.delete_model(request.model_name)

        model_class = ModelRegistry.get_model_class(request.model_type)
        model = model_class(**request.hyperparameters)

        X_train = np.array(request.X_train)
        y_train = np.array(request.y_train)

        dataset_storage.save_dataset(
            f"{request.model_name}_train_data", X_train, y_train, push_to_dvc=True
        )

        model.train(X_train, y_train)

        mlflow_tracker.track_training(
            request.model_name,
            request.model_type,
            model,
            request.hyperparameters,
            X_train,
            y_train,
        )

        storage.save_model(request.model_name, model, request.model_type)

        logger.info(f"Model '{request.model_name}' trained successfully")
        return TrainResponse(
            message="Model trained successfully",
            model_name=request.model_name,
            model_type=request.model_type,
        )

    except ValueError as e:
        logger.warning(f"Training failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Make predictions with a trained model.

    Args:
        request: Prediction request with model name and features

    Returns:
        Predictions from the model

    Raises:
        HTTPException: If prediction fails or model not found
    """
    logger.info(f"Making predictions with model '{request.model_name}'")

    try:
        model = storage.load_model(request.model_name)
        X = np.array(request.X)

        predictions = model.predict(X)

        logger.info(f"Predictions made successfully with model '{request.model_name}'")
        return PredictResponse(
            model_name=request.model_name, predictions=predictions.tolist()
        )

    except ValueError as e:
        logger.warning(f"Prediction failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}", response_model=StatusResponse)
async def delete_model(model_name: str) -> StatusResponse:
    """
    Delete a trained model.

    Args:
        model_name: Name of the model to delete

    Returns:
        Deletion status

    Raises:
        HTTPException: If deletion fails or model not found
    """
    logger.info(f"Deleting model '{model_name}'")

    try:
        storage.delete_model(model_name)
        return StatusResponse(
            status="success", message=f"Model '{model_name}' deleted successfully"
        )

    except ValueError as e:
        logger.warning(f"Deletion failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

