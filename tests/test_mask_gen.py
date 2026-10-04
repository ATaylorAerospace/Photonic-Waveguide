"""Tests for the gdsfactory-backed mask generator (skipped when gdsfactory is absent)."""
import os

import pytest

pytest.importorskip("gdsfactory")

from mcp_server.schemas.waveguide import MaskGenInput  # noqa: E402
from mcp_server.tools.mask_gen import MaskGenerator  # noqa: E402


@pytest.fixture
def generator(tmp_path):
    return MaskGenerator(output_dir=str(tmp_path))


def test_edge_coupler_layout(generator):
    out = generator.generate(MaskGenInput(width_um=1.5, height_nm=400, length_mm=1.0))
    assert os.path.exists(out.gds_file_path)
    # 1 mm straight plus two 200 um inverse tapers.
    assert out.total_length_um == pytest.approx(1400.0)
    (x_min, y_min), (x_max, y_max) = out.bounding_box
    assert x_max - x_min == pytest.approx(1400.0)
    assert y_max - y_min == pytest.approx(1.5)


def test_repeated_calls_in_one_process(generator):
    first = generator.generate(MaskGenInput(width_um=1.5, height_nm=400, length_mm=1.0))
    again = generator.generate(MaskGenInput(width_um=1.5, height_nm=400, length_mm=1.0))
    other = generator.generate(MaskGenInput(width_um=2.0, height_nm=400, length_mm=1.0))
    assert first.gds_file_path == again.gds_file_path
    assert other.gds_file_path != first.gds_file_path
    assert os.path.exists(other.gds_file_path)


def test_grating_coupler_layout(generator):
    out = generator.generate(
        MaskGenInput(width_um=1.5, height_nm=400, length_mm=1.0, io_type="grating_coupler")
    )
    assert os.path.exists(out.gds_file_path)
    # Straight section plus a grating coupler at each end.
    assert out.total_length_um > 1000.0
