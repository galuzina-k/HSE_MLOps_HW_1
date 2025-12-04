import requests

API_URL = "http://localhost:8000"


def health_check():
    print("=== Health Check ===")
    response = requests.get(f"{API_URL}/health")
    print(response.json())
    print()


def list_model_types():
    print("=== Available Model Types ===")
    response = requests.get(f"{API_URL}/models/types")
    for model in response.json():
        print(f"\nType: {model['name']}")
        print(f"Description: {model['description']}")
        print(f"Hyperparameters: {model['hyperparameters']}")
    print()


def train_linear_regression():
    print("=== Training Linear Regression ===")
    data = {
        "model_name": "my_linear_model",
        "model_type": "linear_regression",
        "hyperparameters": {"fit_intercept": True},
        "X_train": [[1], [2], [3], [4], [5]],
        "y_train": [2, 4, 6, 8, 10],
    }
    response = requests.post(f"{API_URL}/models/train", json=data)
    print(response.json())
    print()


def train_logistic_regression():
    print("=== Training Logistic Regression ===")
    data = {
        "model_name": "my_logistic_model",
        "model_type": "logistic_regression",
        "hyperparameters": {"C": 1.0, "max_iter": 100},
        "X_train": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
        "y_train": [0, 0, 0, 1, 1, 1],
    }
    response = requests.post(f"{API_URL}/models/train", json=data)
    print(response.json())
    print()


def train_random_forest():
    print("=== Training Random Forest ===")
    data = {
        "model_name": "my_rf_model",
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        "X_train": [
            [1, 2], [2, 3], [3, 4], [4, 5],
            [5, 6], [6, 7], [7, 8], [8, 9]
        ],
        "y_train": [0, 0, 0, 0, 1, 1, 1, 1],
    }
    response = requests.post(f"{API_URL}/models/train", json=data)
    print(response.json())
    print()


def list_models():
    print("=== List Trained Models ===")
    response = requests.get(f"{API_URL}/models")
    for model in response.json():
        print(f"\nName: {model['name']}")
        print(f"Type: {model['type']}")
        print(f"Hyperparameters: {model['hyperparameters']}")
    print()


def get_model_info(model_name):
    print(f"=== Model Info: {model_name} ===")
    response = requests.get(f"{API_URL}/models/{model_name}")
    print(response.json())
    print()


def predict(model_name, X):
    print(f"=== Predictions from {model_name} ===")
    data = {"model_name": model_name, "X": X}
    response = requests.post(f"{API_URL}/models/predict", json=data)
    result = response.json()
    print(f"Predictions: {result['predictions']}")
    print()


def delete_model(model_name):
    print(f"=== Deleting {model_name} ===")
    response = requests.delete(f"{API_URL}/models/{model_name}")
    print(response.json())
    print()


if __name__ == "__main__":
    health_check()

    list_model_types()

    train_linear_regression()
    train_logistic_regression()
    train_random_forest()

    list_models()

    get_model_info("my_linear_model")

    predict("my_linear_model", [[6], [7], [8]])
    predict("my_logistic_model", [[7, 8], [1, 1]])
    predict("my_rf_model", [[9, 10], [1, 1]])

    print("#" * 10, "SUCCESS", "#" * 10)

