"""Mask generation tools that call the MCP server's generate_mask endpoint."""
from strands import tool

from src.tools.mcp_client import call_mcp_tool


@tool
def generate_foundry_mask(
    width_um: float, height_nm: float, length_mm: float,
    io_type: str = "edge_coupler",
) -> dict:
    """Generate a GDSII mask file for the finalized waveguide design using gdsfactory.

    Produces a foundry-ready straight waveguide layout with inverse tapers
    and optional grating couplers.
    """
    return call_mcp_tool("generate_mask", {
        "width_um": width_um, "height_nm": height_nm, "length_mm": length_mm,
        "io_type": io_type,
    })
