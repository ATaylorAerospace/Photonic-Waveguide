"""Physics tools that call the MCP server's solve_waveguide_mode endpoint."""
from strands import tool

from src.tools.mcp_client import call_mcp_tool


@tool
def solve_mode(
    width_um: float, height_nm: float,
    core_material: str = "SiN", cladding_material: str = "SiO2",
    wavelength_nm: float = 1550.0, polarization: str = "TE",
) -> dict:
    """Solve waveguide eigenmodes using modesolverpy via the MCP physics server.

    Computes effective refractive index, confinement factor, mode field diameter,
    and group index using fully vectorial eigenmode expansion.
    """
    return call_mcp_tool("solve_waveguide_mode", {
        "width_um": width_um, "height_nm": height_nm,
        "core_material": core_material, "cladding_material": cladding_material,
        "wavelength_nm": wavelength_nm, "polarization": polarization,
    })
