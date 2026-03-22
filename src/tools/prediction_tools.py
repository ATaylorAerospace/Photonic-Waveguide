"""XGBoost-based prediction tools for fast first-pass estimates."""
import os
import joblib
import numpy as np
from strands import tool

MODEL_PATH = os.getenv("MODEL_ARTIFACTS_PATH", "models/")


@tool
def predict_loss(
    width_um: float, height_nm: float, wavelength_nm: float = 1550.0,
    polarization: str = "TE", deposition_method: str = "LPCVD",
    etch_method: str = "RIE", cladding: str = "SiO2",
    anneal_temp_c: float = 900.0, anneal_hours: float = 3.0,
) -> dict:
    """Predict propagation loss using XGBoost (fast first-pass estimate).

    For exact physics, use the MCP server's solve_waveguide_mode tool instead.
    """
    model_file = os.path.join(MODEL_PATH, "xgboost_loss_model.joblib")
    if not os.path.exists(model_file):
        return {"error": "Model not trained yet. Run: python src/models/train.py"}
    model = joblib.load(model_file)
    pol_enc = 0 if polarization == "TE" else 1
    dep_enc = {"LPCVD": 0, "PECVD": 1, "HDPCVD": 2}.get(deposition_method, 0)
    etch_enc = {"RIE": 0, "ICP-RIE": 1, "Wet": 2}.get(etch_method, 0)
    clad_enc = {"SiO2": 0, "Air": 1, "SiN": 2}.get(cladding, 0)
    features = np.array([[
        width_um, height_nm, wavelength_nm, pol_enc,
        dep_enc, etch_enc, clad_enc, anneal_temp_c, anneal_hours,
    ]])
    prediction = model.predict(features)
    return {
        "predicted_propagation_loss_dB_cm": float(prediction[0]),
        "model": "XGBoost (fast first-pass)",
        "note": "For exact physics, use solve_waveguide_mode via MCP server",
    }
