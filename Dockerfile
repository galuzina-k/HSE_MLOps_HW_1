FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

COPY app ./app
COPY models ./models

RUN git config --global user.email "mlops@example.com" && \
    git config --global user.name "MLOps System" && \
    git init && \
    git add -A && \
    git commit -m "Initial commit" || true

RUN mkdir -p data/datasets

ENV PYTHONPATH=/app
ENV USE_S3=true
ENV S3_ENDPOINT_URL=http://minio:9000
ENV S3_ACCESS_KEY=minioadmin
ENV S3_SECRET_KEY=minioadmin
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_ENABLED=true

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

