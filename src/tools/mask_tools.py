"""Mask generation tools that call the MCP server's generate_mask endpoint."""
import os
import asyncio
from strands import tool
from fastmcp import Client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


async def _call_generate_mask(params: dict) -> dict:
    async with Client(MCP_SERVER_URL) as client:
        return await client.call_tool("generate_mask", params)


@tool
def generate_foundry_mask(
    width_um: float, height_nm: float, length_mm: float,
    bend_radius_um: float = 50.0, io_type: str = "edge_coupler",
    routing: str = "bezier",
) -> dict:
    """Generate a GDSII mask file for the finalized waveguide design using gdsfactory.

    Produces a foundry-ready layout with tapers, routing, and I/O couplers.
    """
    return asyncio.run(_call_generate_mask({
        "width_um": width_um, "height_nm": height_nm, "length_mm": length_mm,
        "bend_radius_um": bend_radius_um, "io_type": io_type, "routing": routing,
    }))
