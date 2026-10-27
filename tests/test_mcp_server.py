"""Unit tests for MCP server physics tools."""
import pytest
from pydantic import ValidationError

from mcp_server.schemas.waveguide import (
    InverseDesignInput,
    MaskGenInput,
    ModeSolverInput,
)


class TestModeSolverInput:
    def test_valid_input(self):
        params = ModeSolverInput(width_um=1.5, height_nm=400)
        assert params.width_um == 1.5
        assert params.core_material == "SiN"
        assert params.polarization == "TE"

    def test_invalid_width(self):
        with pytest.raises(ValidationError):
            ModeSolverInput(width_um=-1.0, height_nm=400)

    def test_invalid_polarization(self):
        with pytest.raises(ValidationError):
            ModeSolverInput(width_um=1.5, height_nm=400, polarization="XX")

    def test_rejects_unknown_material(self):
        with pytest.raises(ValidationError, match="Unsupported core_material"):
            ModeSolverInput(width_um=1.5, height_nm=400, core_material="GaAs")

    def test_rejects_cladding_index_greater_or_equal_to_core(self):
        with pytest.raises(ValidationError, match="higher refractive index"):
            ModeSolverInput(width_um=1.5, height_nm=400, core_material="SiO2", cladding_material="SiN")

    def test_rejects_non_finite_dimensions(self):
        with pytest.raises(ValidationError):
            ModeSolverInput(width_um=float("nan"), height_nm=400)
        with pytest.raises(ValidationError):
            ModeSolverInput(width_um=float("inf"), height_nm=400)


class TestInverseDesignInput:
    def test_valid_input(self):
        params = InverseDesignInput(target_metric="propagation_loss", target_value=0.2)
        assert params.max_iterations == 200

    def test_invalid_metric(self):
        with pytest.raises(ValidationError):
            InverseDesignInput(target_metric="invalid", target_value=0.2)

    def test_fractional_metric_target_must_be_in_unit_interval(self):
        with pytest.raises(ValidationError):
            InverseDesignInput(target_metric="confinement", target_value=1.5)
        with pytest.raises(ValidationError):
            InverseDesignInput(target_metric="coupling_efficiency", target_value=0.0)

    def test_loss_target_must_be_positive(self):
        with pytest.raises(ValidationError):
            InverseDesignInput(target_metric="propagation_loss", target_value=-0.1)

    def test_search_bounds_must_be_ordered(self):
        with pytest.raises(ValidationError):
            InverseDesignInput(target_metric="confinement", target_value=0.5,
                               width_range_um=(2.0, 1.0))

    def test_rejects_nan_or_inf_target_and_bounds(self):
        with pytest.raises(ValidationError):
            InverseDesignInput(target_metric="propagation_loss", target_value=float("nan"))
        with pytest.raises(ValidationError):
            InverseDesignInput(target_metric="propagation_loss", target_value=float("inf"))
        with pytest.raises(ValidationError):
            InverseDesignInput(
                target_metric="propagation_loss",
                target_value=0.2,
                width_range_um=(0.3, float("inf")),
            )


class TestMaskGenInput:
    def test_valid_input(self):
        params = MaskGenInput(width_um=1.5, height_nm=400, length_mm=10.0)
        assert params.io_type == "edge_coupler"
        assert params.output_filename == "waveguide_design.gds"

    def test_rejects_path_traversal_filename(self):
        with pytest.raises(ValidationError):
            MaskGenInput(width_um=1.5, height_nm=400, length_mm=10.0,
                         output_filename="../../evil.gds")

    def test_rejects_absolute_path_filename(self):
        with pytest.raises(ValidationError):
            MaskGenInput(width_um=1.5, height_nm=400, length_mm=10.0,
                         output_filename="/tmp/evil.gds")
