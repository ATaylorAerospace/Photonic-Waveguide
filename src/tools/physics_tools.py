"""Physics tools that call the MCP server's solve_waveguide_mode endpoint."""
import os
import asyncio
from strands import tool
from fastmcp import Client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


async def _call_mode_solver(params: dict) -> dict:
    async with Client(MCP_SERVER_URL) as client:
        return await client.call_tool("solve_waveguide_mode", params)


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
    return asyncio.run(_call_mode_solver({
        "width_um": width_um, "height_nm": height_nm,
        "core_material": core_material, "cladding_material": cladding_material,
        "wavelength_nm": wavelength_nm, "polarization": polarization,
    }))
