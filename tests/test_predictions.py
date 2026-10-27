"""ML model training, evaluation, and prediction tests."""
import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import evaluate_model
from src.models.features import encode_categorical, prepare_features_and_target
from src.models.train import train_loss_model


@pytest.fixture
def synthetic_dataset_csv(tmp_path):
    rng = np.random.default_rng(42)
    n = 60
    widths = rng.uniform(0.5, 2.5, size=n)
    heights = rng.uniform(200.0, 600.0, size=n)
    wavelengths = rng.choice([1310.0, 1550.0], size=n)
    polarizations = rng.choice(["TE", "TM"], size=n)
    depositions = rng.choice(["LPCVD", "PECVD", "HDPCVD"], size=n)
    etches = rng.choice(["RIE", "ICP-RIE", "Wet"], size=n)
    claddings = rng.choice(["SiO2", "Air", "SiN"], size=n)
    anneal_temps = rng.uniform(800.0, 1100.0, size=n)
    anneal_hours = rng.uniform(1.0, 5.0, size=n)
    loss = 1.5 - 0.3 * widths - 0.001 * heights + np.where(depositions == "LPCVD", -0.2, 0.2)

    df = pd.DataFrame({
        "width_um": widths,
        "height_nm": heights,
        "wavelength_nm": wavelengths,
        "polarization": polarizations,
        "deposition_method": depositions,
        "etch_method": etches,
        "cladding_material": claddings,
        "anneal_temp_c": anneal_temps,  # lowercase 'c' tests case-insensitive canonicalization
        "anneal_hours": anneal_hours,
        "propagation_loss_dB_cm": loss,
    })
    csv_path = tmp_path / "synthetic_waveguides.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_encode_categorical_strips_whitespace_and_handles_unknown():
    assert encode_categorical("polarization", " TM ") == 1
    assert encode_categorical("deposition_method", "PECVD") == 1
    assert encode_categorical("etch_method", "Unknown") == 0


def test_prepare_features_missing_columns():
    df = pd.DataFrame({"width_um": [1.0, 2.0]})
    X, y, missing = prepare_features_and_target(df)
    assert X is None
    assert y is None
    assert "height_nm" in missing


def test_train_evaluate_and_predict_loss(synthetic_dataset_csv, tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_path = train_loss_model(dataset_path=synthetic_dataset_csv, output_dir=str(model_dir))
    assert model_path is not None

    metrics = evaluate_model(model_path, synthetic_dataset_csv)
    assert set(metrics.keys()) == {"r2", "mae", "rmse", "num_samples"}
    assert metrics["num_samples"] == 60
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0

    from src.tools import prediction_tools
    monkeypatch.setattr(prediction_tools, "MODEL_PATH", str(model_dir))
    result = prediction_tools.predict_loss(width_um=1.5, height_nm=400.0)
    assert "predicted_propagation_loss_dB_cm" in result
    assert np.isfinite(result["predicted_propagation_loss_dB_cm"])
