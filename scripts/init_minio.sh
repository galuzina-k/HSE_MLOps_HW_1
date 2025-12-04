#!/bin/bash

sleep 5

mc alias set myminio http://minio:9000 minioadmin minioadmin

mc mb myminio/mlops-models --ignore-existing
mc mb myminio/mlops-dvc --ignore-existing
mc mb myminio/mlflow-artifacts --ignore-existing

echo "Minio buckets created successfully"

