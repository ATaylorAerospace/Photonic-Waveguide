"""Tests for the JAX inverse-design engine (skipped when JAX is absent)."""
import pytest

pytest.importorskip("jax")

from mcp_server.schemas.waveguide import InverseDesignInput  # noqa: E402
from mcp_server.tools.inverse_design import InverseDesigner  # noqa: E402


def test_reaches_a_reachable_target():
    out = InverseDesigner().optimize(InverseDesignInput(
        target_metric="confinement", target_value=0.9,
        width_range_um=(0.3, 1.5), max_iterations=2000, learning_rate=0.5,
    ))
    assert out.converged
    assert out.iterations < 2000
    assert abs(out.achieved_value - 0.9) < 1e-3
    assert 0.3 <= out.optimized_width_um <= 1.5


def test_reports_an_unreached_target():
    # The surrogate's propagation loss floors at 0.1 dB/cm, so 0.05 is unreachable.
    out = InverseDesigner().optimize(InverseDesignInput(
        target_metric="propagation_loss", target_value=0.05, max_iterations=50,
    ))
    assert not out.converged
    assert out.iterations == 50
    assert len(out.convergence_history) == 50
    assert 0.3 <= out.optimized_width_um <= 5.0
    assert 100.0 <= out.optimized_height_nm <= 800.0


def test_compilation_is_reused_across_requests():
    designer = InverseDesigner()
    designer.optimize(InverseDesignInput(
        target_metric="confinement", target_value=0.8, max_iterations=5,
    ))
    designer.optimize(InverseDesignInput(
        target_metric="confinement", target_value=0.6, wavelength_nm=1310.0,
        width_range_um=(0.5, 2.0), max_iterations=5,
    ))
    assert list(designer._compiled) == ["confinement"]
    compiled = designer._compiled["confinement"]
    if hasattr(compiled, "_cache_size"):
        # Different targets, wavelengths and bounds must not trigger a retrace.
        assert compiled._cache_size() == 1
