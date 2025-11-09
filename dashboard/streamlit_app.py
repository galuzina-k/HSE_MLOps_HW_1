"""Streamlit dashboard for MLOps API."""

import json

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MLOps Dashboard", page_icon="🤖")
st.title("🤖 MLOps Dashboard")

try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    if response.status_code != 200:
        st.error("❌ API is not available")
        st.code("uvicorn app.main:app --reload")
        st.stop()
except Exception:
    st.error("❌ API is not available")
    st.code("uvicorn app.main:app --reload")
    st.stop()

st.success("✅ API is connected")

tab1, tab2, tab3 = st.tabs(["📋 Models", "🎓 Train", "🔮 Predict"])

with tab1:
    st.header("Available Models")

    response = requests.get(f"{API_URL}/models/types")
    if response.status_code == 200:
        st.subheader("Model Types")
        for model in response.json():
            with st.expander(f"{model['name']}"):
                st.write(model["description"])
                st.json(model["hyperparameters"])

    response = requests.get(f"{API_URL}/models")
    if response.status_code == 200:
        trained_models = response.json()
        st.subheader(f"Trained Models ({len(trained_models)})")
        for model in trained_models:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{model['name']}** - {model['type']}")
            with col2:
                if st.button("Delete", key=f"del_{model['name']}"):
                    requests.delete(f"{API_URL}/models/{model['name']}")
                    st.rerun()

with tab2:
    st.header("Train Model")

    response = requests.get(f"{API_URL}/models/types")
    model_types = [m["name"] for m in response.json()]

    model_type = st.selectbox("Model Type", model_types)
    model_name = st.text_input("Model Name", value="my_model")

    hyperparams_json = st.text_area(
        "Hyperparameters (JSON)",
        value='{"C": 1.0, "max_iter": 100}',
    )

    X_train_input = st.text_area(
        "Features (comma-separated, one row per line)",
        value="1.0, 2.0\n3.0, 4.0\n5.0, 6.0",
    )

    y_train_input = st.text_area(
        "Targets (one value per line)",
        value="0\n1\n0",
    )

    if st.button("Train", type="primary"):
        try:
            hyperparams = json.loads(hyperparams_json)
            X_train = [
                [float(x.strip()) for x in row.split(",")]
                for row in X_train_input.strip().split("\n")
            ]
            y_train = [float(y.strip()) for y in y_train_input.strip().split("\n")]

            payload = {
                "model_type": model_type,
                "model_name": model_name,
                "hyperparameters": hyperparams,
                "X_train": X_train,
                "y_train": y_train,
            }

            response = requests.post(f"{API_URL}/models/train", json=payload)
            if response.status_code == 200:
                st.success("✅ Model trained successfully")
                st.rerun()
            else:
                st.error(f"❌ {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    st.header("Make Predictions")

    response = requests.get(f"{API_URL}/models")
    trained_models = response.json()

    if not trained_models:
        st.warning("No trained models available")
        st.stop()

    model_options = [m["name"] for m in trained_models]
    selected_model = st.selectbox("Select Model", model_options)

    X_predict_input = st.text_area(
        "Features (comma-separated, one row per line)",
        value="1.0, 2.0\n3.0, 4.0",
    )

    if st.button("Predict", type="primary"):
        try:
            X_predict = [
                [float(x.strip()) for x in row.split(",")]
                for row in X_predict_input.strip().split("\n")
            ]

            payload = {"model_name": selected_model, "X": X_predict}
            response = requests.post(f"{API_URL}/models/predict", json=payload)

            if response.status_code == 200:
                predictions = response.json()["predictions"]
                st.success("✅ Predictions completed")
                for i, pred in enumerate(predictions, 1):
                    st.write(f"Sample {i}: {pred}")
            else:
                st.error(f"❌ {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error: {e}")
