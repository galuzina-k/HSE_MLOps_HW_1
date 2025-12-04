"""Pydantic schemas for API requests and responses."""

from typing import Any

from pydantic import BaseModel


class TrainRequest(BaseModel):
    """Training request."""

    model_type: str
    model_name: str
    hyperparameters: dict[str, Any] = {}
    X_train: list[list[float]]
    y_train: list[float]


class PredictRequest(BaseModel):
    """Prediction request."""

    model_name: str
    X: list[list[float]]


class TrainResponse(BaseModel):
    """Training response."""

    message: str
    model_name: str
    model_type: str


class PredictResponse(BaseModel):
    """Prediction response."""

    model_name: str
    predictions: list[float]


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    type: str
    hyperparameters: dict[str, Any]


class ModelTypeInfo(BaseModel):
    """Model type information."""

    name: str
    description: str
    hyperparameters: dict[str, str]


class StatusResponse(BaseModel):
    """Status response."""

    status: str
    message: str

class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None


