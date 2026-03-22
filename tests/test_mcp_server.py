"""Unit tests for MCP server physics tools."""
import pytest
from mcp_server.schemas.waveguide import ModeSolverInput, InverseDesignInput, MaskGenInput


class TestModeSolverInput:
    def test_valid_input(self):
        params = ModeSolverInput(width_um=1.5, height_nm=400)
        assert params.width_um == 1.5
        assert params.core_material == "SiN"
        assert params.polarization == "TE"

    def test_invalid_width(self):
        with pytest.raises(Exception):
            ModeSolverInput(width_um=-1.0, height_nm=400)

    def test_invalid_polarization(self):
        with pytest.raises(Exception):
            ModeSolverInput(width_um=1.5, height_nm=400, polarization="XX")


class TestInverseDesignInput:
    def test_valid_input(self):
        params = InverseDesignInput(target_metric="propagation_loss", target_value=0.2)
        assert params.max_iterations == 200

    def test_invalid_metric(self):
        with pytest.raises(Exception):
            InverseDesignInput(target_metric="invalid", target_value=0.2)


class TestMaskGenInput:
    def test_valid_input(self):
        params = MaskGenInput(width_um=1.5, height_nm=400, length_mm=10.0)
        assert params.routing == "bezier"
        assert params.output_filename == "waveguide_design.gds"
