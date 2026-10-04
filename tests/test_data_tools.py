"""Unit tests for the agent-side data tools (in-memory dataset, no AWS)."""
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

pytest.importorskip("strands")

from src.tools import data_tools  # noqa: E402


@pytest.fixture
def dataset(monkeypatch):
    df = pd.DataFrame({
        "width_um": [1.0, 1.5, 1.0, 1.5],
        "height_nm": [300, 300, 400, 400],
        "deposition_method": ["LPCVD", "PECVD", "LPCVD", "PECVD"],
        "propagation_loss_dB_cm": [0.5, 0.3, 0.2, 0.1],
    })
    # Serve the S3 Tables layer from memory and the fallback from the same frame.
    monkeypatch.setattr(data_tools._s3_tables, "_dataset", ds.dataset(pa.Table.from_pandas(df)))
    monkeypatch.setattr(data_tools, "_df_cache", df)
    return df


class TestQueryDatasetGroupBy:
    def test_group_by_string_column(self, dataset):
        rows = data_tools.query_dataset(group_by="deposition_method")
        by_method = {r["deposition_method"]: r for r in rows}
        assert by_method["LPCVD"]["propagation_loss_dB_cm"] == pytest.approx(0.35)
        assert by_method["PECVD"]["height_nm"] == pytest.approx(350.0)

    def test_group_by_numeric_column(self, dataset):
        rows = data_tools.query_dataset(group_by="height_nm")
        by_height = {r["height_nm"]: r for r in rows}
        assert by_height[300]["propagation_loss_dB_cm"] == pytest.approx(0.4)
        assert by_height[400]["propagation_loss_dB_cm"] == pytest.approx(0.15)

    def test_group_by_numeric_column_local_fallback(self, dataset, monkeypatch):
        def fail(*args, **kwargs):
            raise RuntimeError("S3 Tables unavailable")
        monkeypatch.setattr(data_tools._s3_tables, "query", fail)
        rows = data_tools.query_dataset(group_by="height_nm")
        assert {r["height_nm"] for r in rows} == {300, 400}

    def test_filter_then_limit(self, dataset):
        rows = data_tools.query_dataset(filters={"deposition_method": "LPCVD"}, limit=1)
        assert len(rows) == 1
        assert rows[0]["deposition_method"] == "LPCVD"
