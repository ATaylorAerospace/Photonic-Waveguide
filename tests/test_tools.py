"""Unit tests for agent tools."""


def test_predict_loss_missing_model(tmp_path, monkeypatch):
    from src.tools import prediction_tools
    monkeypatch.setattr(prediction_tools, "MODEL_PATH", str(tmp_path))
    result = prediction_tools.predict_loss(width_um=1.5, height_nm=400)
    assert "error" in result
