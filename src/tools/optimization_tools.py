"""Optimization tools that call the MCP server's optimize_waveguide endpoint."""
from strands import tool

from src.tools.mcp_client import call_mcp_tool


@tool
def optimize_design(
    target_metric: str, target_value: float,
    wavelength_nm: float = 1550.0, polarization: str = "TE",
    constraints: dict | None = None,
) -> dict:
    """Optimize waveguide geometry using gradient-based inverse design via SAX + JAX.

    Uses automatic differentiation to find the optimal width and height
    that achieve the target metric value.
    """
    return call_mcp_tool("optimize_waveguide", {
        "target_metric": target_metric, "target_value": target_value,
        "wavelength_nm": wavelength_nm, "polarization": polarization,
        "constraints": constraints or {},
    })
