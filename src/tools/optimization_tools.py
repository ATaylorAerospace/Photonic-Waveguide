"""Optimization tools that call the MCP server's optimize_waveguide endpoint."""
import os
import asyncio
from strands import tool
from fastmcp import Client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


async def _call_optimize(params: dict) -> dict:
    async with Client(MCP_SERVER_URL) as client:
        return await client.call_tool("optimize_waveguide", params)


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
    return asyncio.run(_call_optimize({
        "target_metric": target_metric, "target_value": target_value,
        "wavelength_nm": wavelength_nm, "polarization": polarization,
        "constraints": constraints or {},
    }))
