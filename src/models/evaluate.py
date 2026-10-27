"""Model evaluation and metrics."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.features import prepare_features_and_target


def evaluate_model(model_path: str, test_data_path: str) -> dict:
    """Evaluate a trained model against test data.

    Returns a dictionary containing `r2`, `mae`, `rmse`, and `num_samples`.
    """
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    X, y, missing = prepare_features_and_target(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    predictions = model.predict(X)
    return {
        "r2": float(r2_score(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "num_samples": len(y),
    }
