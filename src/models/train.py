"""Train XGBoost regression models on the waveguide dataset."""
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config.agent_config import DATASET_PATH, MODEL_ARTIFACTS_PATH
from src.models.features import prepare_features_and_target

MODEL_FILENAME = "xgboost_loss_model.joblib"


def train_loss_model(dataset_path: str = DATASET_PATH, output_dir: str = MODEL_ARTIFACTS_PATH):
    """Train XGBoost model for propagation loss prediction.

    The model is written to the directory predict_loss reads from
    (MODEL_ARTIFACTS_PATH), so training and inference cannot disagree on
    where the artifact lives.
    """
    df = pd.read_csv(dataset_path)
    X, y, missing = prepare_features_and_target(df)
    if missing:
        print(f"Missing required columns: {missing}. Available: {list(df.columns)}")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    print(f"XGBoost R² score: {model.score(X_test, y_test):.4f}")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, MODEL_FILENAME)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    train_loss_model()
