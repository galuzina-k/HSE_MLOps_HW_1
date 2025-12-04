"""Simple tests for the MLOps API."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_list_model_types():
    """Test listing available model types."""
    response = client.get("/models/types")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    assert any(m["name"] == "linear_regression" for m in data)
    assert any(m["name"] == "logistic_regression" for m in data)


def test_train_and_predict():
    """Test training a model and making predictions."""
    train_data = {
        "model_type": "linear_regression",
        "model_name": "test_model",
        "hyperparameters": {"fit_intercept": True},
        "X_train": [[1], [2], [3]],
        "y_train": [2, 4, 6],
    }

    response = client.post("/models/train", json=train_data)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test_model"

    predict_data = {"model_name": "test_model", "X": [[4], [5]]}

    response = client.post("/models/predict", json=predict_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

    response = client.delete("/models/test_model")
    assert response.status_code == 200


def test_predict_nonexistent_model():
    """Test prediction with non-existent model returns 404."""
    predict_data = {"model_name": "nonexistent_model", "X": [[1, 2]]}

    response = client.post("/models/predict", json=predict_data)
    assert response.status_code == 404


def test_train_duplicate_model():
    """Test training a model with duplicate name replaces the old one."""
    train_data = {
        "model_type": "linear_regression",
        "model_name": "duplicate_test",
        "hyperparameters": {"fit_intercept": True},
        "X_train": [[1], [2]],
        "y_train": [1, 2],
    }

    response1 = client.post("/models/train", json=train_data)
    assert response1.status_code == 200

    response2 = client.post("/models/train", json=train_data)
    assert response2.status_code == 200

    client.delete("/models/duplicate_test")

