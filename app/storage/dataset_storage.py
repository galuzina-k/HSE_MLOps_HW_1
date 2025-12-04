import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.logger import logger


class DatasetStorage:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.datasets_dir = self.data_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)

    def save_dataset(
        self, dataset_name: str, X: np.ndarray, y: np.ndarray, push_to_dvc: bool = True
    ):
        dataset_path = self.datasets_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)

        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns=["target"])

        X_file = dataset_path / "X.csv"
        y_file = dataset_path / "y.csv"
        metadata_file = dataset_path / "metadata.json"

        X_df.to_csv(X_file, index=False)
        y_df.to_csv(y_file, index=False)

        metadata = {"shape": X.shape, "n_samples": len(X), "n_features": X.shape[1] if len(X.shape) > 1 else 1}
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset '{dataset_name}' saved locally")

        if push_to_dvc:
            self._add_to_dvc(dataset_path)

    def load_dataset(self, dataset_name: str, pull_from_dvc: bool = True) -> tuple[np.ndarray, np.ndarray]:
        dataset_path = self.datasets_dir / dataset_name

        if not dataset_path.exists() and pull_from_dvc:
            self._pull_from_dvc(dataset_path)

        X_file = dataset_path / "X.csv"
        y_file = dataset_path / "y.csv"

        if not X_file.exists() or not y_file.exists():
            raise ValueError(f"Dataset '{dataset_name}' not found")

        X_df = pd.read_csv(X_file)
        y_df = pd.read_csv(y_file)

        logger.info(f"Dataset '{dataset_name}' loaded")
        return X_df.values, y_df.values.ravel()

    def _add_to_dvc(self, dataset_path: Path):
        try:
            add_result = subprocess.run(
                ["dvc", "add", str(dataset_path)],
                cwd=self.data_dir.parent,
                capture_output=True,
                text=True,
            )
            if add_result.returncode == 0:
                logger.info(f"Added {dataset_path.name} to DVC")

                push_result = subprocess.run(
                    ["dvc", "push", f"{dataset_path}.dvc"],
                    cwd=self.data_dir.parent,
                    capture_output=True,
                    text=True,
                )
                if push_result.returncode == 0:
                    logger.info(f"Pushed {dataset_path.name} to DVC remote")
                else:
                    logger.warning(f"DVC push failed: {push_result.stderr}")
            else:
                logger.warning(f"DVC add failed: {add_result.stderr}")
        except Exception as e:
            logger.warning(f"Could not add to DVC: {e}")

    def _pull_from_dvc(self, dataset_path: Path):
        try:
            result = subprocess.run(
                ["dvc", "pull", f"{dataset_path}.dvc"],
                cwd=self.data_dir.parent,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"Pulled {dataset_path.name} from DVC")
            else:
                logger.warning(f"DVC pull failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Could not pull from DVC: {e}")

