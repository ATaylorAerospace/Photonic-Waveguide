"""Integration tests for MCP server via FastMCP client."""
import pytest


@pytest.mark.asyncio
async def test_solve_mode_via_mcp():
    """Requires MCP server running on localhost:8000."""
    from fastmcp import Client
    try:
        async with Client("http://localhost:8000/mcp") as client:
            result = await client.call_tool("solve_waveguide_mode", {
                "width_um": 1.5, "height_nm": 400,
                "core_material": "SiN", "cladding_material": "SiO2",
                "wavelength_nm": 1550, "polarization": "TE",
            })
            assert "n_eff" in result
    except ConnectionError:
        pytest.skip("MCP server not running")


@pytest.mark.asyncio
async def test_optimize_via_mcp():
    """Requires MCP server running."""
    from fastmcp import Client
    try:
        async with Client("http://localhost:8000/mcp") as client:
            result = await client.call_tool("optimize_waveguide", {
                "target_metric": "propagation_loss", "target_value": 0.2,
            })
            assert "optimized_width_um" in result
    except ConnectionError:
        pytest.skip("MCP server not running")
