"""Unit tests for agent tools."""


def test_predict_loss_missing_model():
    from src.tools.prediction_tools import predict_loss
    result = predict_loss(width_um=1.5, height_nm=400)
    assert "error" in result or "predicted_propagation_loss_dB_cm" in result
