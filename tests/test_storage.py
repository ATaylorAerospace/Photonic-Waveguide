"""Unit tests for the hybrid storage layer."""
import pytest
from mcp_server.storage.cache import SimulationCache


class TestSimulationCacheHashing:
    def test_hash_is_deterministic(self):
        params_a = {"width_um": 1.5, "height_nm": 400.0, "polarization": "TE"}
        params_b = {"height_nm": 400.0, "polarization": "TE", "width_um": 1.5}
        assert SimulationCache._hash_params(params_a) == SimulationCache._hash_params(params_b)

    def test_hash_rounds_floats(self):
        params_a = {"width_um": 1.5000001, "height_nm": 400.0}
        params_b = {"width_um": 1.5000002, "height_nm": 400.0}
        assert SimulationCache._hash_params(params_a) == SimulationCache._hash_params(params_b)

    def test_different_params_different_hash(self):
        params_a = {"width_um": 1.5, "height_nm": 400.0}
        params_b = {"width_um": 2.0, "height_nm": 400.0}
        assert SimulationCache._hash_params(params_a) != SimulationCache._hash_params(params_b)


class TestSimulationCacheSortKey:
    def test_mode_solver_key(self):
        key = SimulationCache._build_sort_key("solve_waveguide_mode", {
            "wavelength_nm": 1550.0, "polarization": "TE",
        })
        assert key == "MODE#1550.0nm#TE"

    def test_optimize_key(self):
        key = SimulationCache._build_sort_key("optimize_waveguide", {
            "target_metric": "propagation_loss", "target_value": 0.2,
        })
        assert key == "OPTIM#propagation_loss#0.2"

    def test_mask_gen_key(self):
        key = SimulationCache._build_sort_key("generate_mask", {
            "io_type": "edge_coupler", "routing": "bezier",
        })
        assert key == "GDS#edge_coupler#bezier"


class TestSimulationCacheGracefulDegradation:
    def test_disabled_cache_returns_none_on_get(self):
        cache = SimulationCache.__new__(SimulationCache)
        cache._enabled = False
        assert cache.get("solve_waveguide_mode", {"width_um": 1.5}) is None

    def test_disabled_cache_put_is_noop(self):
        cache = SimulationCache.__new__(SimulationCache)
        cache._enabled = False
        cache.put("solve_waveguide_mode", {"width_um": 1.5}, {"n_eff": 1.8})


class TestS3DatasetReader:
    def test_local_csv_fallback(self, tmp_path):
        import pandas as pd
        from mcp_server.storage.s3_dataset import S3DatasetReader
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({
            "width_um": [1.0, 1.5, 2.0],
            "height_nm": [300, 400, 500],
            "propagation_loss_dB_cm": [0.5, 0.3, 0.2],
        }).to_csv(csv_path, index=False)
        reader = S3DatasetReader(bucket="nonexistent-bucket", local_fallback=str(csv_path))
        df = reader.to_pandas(columns=["width_um", "height_nm"])
        assert len(df) == 3
        assert list(df.columns) == ["width_um", "height_nm"]


class TestS3TablesQuery:
    def test_local_csv_query(self, tmp_path):
        import pandas as pd
        from mcp_server.storage.s3_tables import S3TablesQuery
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({
            "width_um": [1.0, 1.5, 2.0],
            "height_nm": [300, 400, 500],
            "propagation_loss_dB_cm": [3.0, 0.3, 0.2],
            "polarization": ["TE", "TE", "TM"],
        }).to_csv(csv_path, index=False)
        query = S3TablesQuery(bucket="nonexistent-bucket", local_fallback=str(csv_path))
        df = query.query_low_loss(max_loss_db_cm=1.0, polarization="TE")
        assert len(df) == 1
        assert df.iloc[0]["width_um"] == 1.5
