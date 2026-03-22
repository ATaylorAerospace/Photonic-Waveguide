"""Model serving for inference."""
import joblib


def load_model(model_path: str):
    """Load a trained model for inference."""
    return joblib.load(model_path)
