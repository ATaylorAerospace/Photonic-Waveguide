"""FastMCP server entry point.

Registers three physics tools and serves them via the Model Context Protocol.
Each tool integrates with the DynamoDB simulation cache — checking for cached
results before running expensive physics, and writing results back on cache miss.

Run with: python -m mcp_server.server
"""
from fastmcp import FastMCP
from mcp_server.schemas.waveguide import (
    ModeSolverInput,
    InverseDesignInput,
    MaskGenInput,
)
from mcp_server.tools.mode_solver import WaveguideSolver
from mcp_server.tools.inverse_design import InverseDesigner
from mcp_server.tools.mask_gen import MaskGenerator
from mcp_server.storage.cache import SimulationCache

mcp = FastMCP(
    name="photonic-waveguide-mcp-server",
    description=(
        "Deterministic physics tools for SiN photonic waveguide design. "
        "Provides eigenmode solving (modesolverpy), gradient-based inverse "
        "design (SAX + JAX), and GDSII mask generation (gdsfactory). "
        "Backed by a hybrid storage layer: DynamoDB simulation cache, "
        "S3 Express One Zone Parquet bulk data, and S3 Tables SQL queries."
    ),
)

_solver = WaveguideSolver()
_designer = InverseDesigner()
_mask_gen = MaskGenerator()
_cache = SimulationCache()


@mcp.tool()
def solve_waveguide_mode(
    width_um: float,
    height_nm: float,
    core_material: str = "SiN",
    cladding_material: str = "SiO2",
    wavelength_nm: float = 1550.0,
    polarization: str = "TE",
    num_modes: int = 1,
) -> dict:
    """Compute waveguide eigenmode properties using fully vectorial EME.

    Returns effective refractive index (n_eff), optical confinement factor,
    mode field diameter (MFD), and group index for the specified waveguide
    cross-section and operating conditions.

    Results are cached in DynamoDB — repeated calls for the same geometry
    return in sub-10ms without re-running the eigenmode solver.

    Args:
        width_um: Waveguide core width in microns (e.g. 1.5)
        height_nm: Waveguide core height in nanometers (e.g. 400)
        core_material: Core material — "SiN", "Si3N4", or "Si"
        cladding_material: Cladding material — "SiO2", "Air", or "SiN"
        wavelength_nm: Operating wavelength in nm (e.g. 1550)
        polarization: "TE" or "TM"
        num_modes: Number of modes to solve (1-10)
    """
    raw_params = {
        "width_um": width_um, "height_nm": height_nm,
        "core_material": core_material, "cladding_material": cladding_material,
        "wavelength_nm": wavelength_nm, "polarization": polarization,
        "num_modes": num_modes,
    }
    cached = _cache.get("solve_waveguide_mode", raw_params)
    if cached is not None:
        cached["cache_hit"] = True
        return cached
    params = ModeSolverInput(**raw_params)
    result = _solver.solve(params).model_dump()
    result["cache_hit"] = False
    _cache.put("solve_waveguide_mode", raw_params, result)
    return result


@mcp.tool()
def optimize_waveguide(
    target_metric: str,
    target_value: float,
    wavelength_nm: float = 1550.0,
    polarization: str = "TE",
    constraints: dict | None = None,
    width_range_um: tuple[float, float] = (0.3, 5.0),
    height_range_nm: tuple[float, float] = (100.0, 800.0),
    max_iterations: int = 200,
    learning_rate: float = 0.01,
) -> dict:
    """Perform gradient-based inverse design to optimize waveguide geometry.

    Uses SAX (JAX-based photonic circuit solver) with automatic differentiation
    to find the optimal width and height that achieve the target metric value.

    Results are cached in DynamoDB — identical optimization requests return
    the previously computed optimum instantly.

    Args:
        target_metric: "insertion_loss", "coupling_efficiency", "propagation_loss", or "confinement"
        target_value: Desired value (e.g. 0.2 for propagation_loss in dB/cm)
        wavelength_nm: Operating wavelength in nm
        polarization: "TE" or "TM"
        constraints: Fixed parameters e.g. {"deposition": "LPCVD"}
        width_range_um: (min, max) width bounds in microns
        height_range_nm: (min, max) height bounds in nanometers
        max_iterations: Maximum optimization iterations
        learning_rate: Gradient descent step size
    """
    raw_params = {
        "target_metric": target_metric, "target_value": target_value,
        "wavelength_nm": wavelength_nm, "polarization": polarization,
        "constraints": constraints or {},
        "width_range_um": list(width_range_um),
        "height_range_nm": list(height_range_nm),
        "max_iterations": max_iterations, "learning_rate": learning_rate,
    }
    cached = _cache.get("optimize_waveguide", raw_params)
    if cached is not None:
        cached["cache_hit"] = True
        return cached
    params = InverseDesignInput(
        target_metric=target_metric, target_value=target_value,
        wavelength_nm=wavelength_nm, polarization=polarization,
        constraints=constraints or {},
        width_range_um=width_range_um, height_range_nm=height_range_nm,
        max_iterations=max_iterations, learning_rate=learning_rate,
    )
    result = _designer.optimize(params).model_dump()
    result["cache_hit"] = False
    _cache.put("optimize_waveguide", raw_params, result)
    return result


@mcp.tool()
def generate_mask(
    width_um: float,
    height_nm: float,
    length_mm: float,
    bend_radius_um: float = 50.0,
    io_type: str = "edge_coupler",
    taper_length_um: float = 200.0,
    routing: str = "bezier",
    layer: tuple[int, int] = (1, 0),
    output_filename: str = "waveguide_design.gds",
) -> dict:
    """Generate a foundry-ready GDSII mask file from waveguide parameters.

    Uses gdsfactory to create a photonic layout with proper routing,
    tapers, and I/O couplers. Returns the file path to the generated .gds file.

    GDSII file paths are cached in DynamoDB — if the exact same layout was
    previously generated, the cached file path is returned immediately.

    Args:
        width_um: Waveguide width in microns
        height_nm: Waveguide height in nm (stored as layout metadata)
        length_mm: Total waveguide length in millimeters
        bend_radius_um: Bend radius in microns
        io_type: "edge_coupler" or "grating_coupler"
        taper_length_um: Taper length in microns
        routing: "manhattan" or "bezier"
        layer: GDS layer and datatype as (layer, datatype)
        output_filename: Output filename for the .gds file
    """
    raw_params = {
        "width_um": width_um, "height_nm": height_nm,
        "length_mm": length_mm, "bend_radius_um": bend_radius_um,
        "io_type": io_type, "taper_length_um": taper_length_um,
        "routing": routing, "layer": list(layer),
        "output_filename": output_filename,
    }
    cached = _cache.get("generate_mask", raw_params)
    if cached is not None:
        import os
        if os.path.exists(cached.get("gds_file_path", "")):
            cached["cache_hit"] = True
            return cached
    params = MaskGenInput(
        width_um=width_um, height_nm=height_nm, length_mm=length_mm,
        bend_radius_um=bend_radius_um, io_type=io_type,
        taper_length_um=taper_length_um, routing=routing,
        layer=layer, output_filename=output_filename,
    )
    result = _mask_gen.generate(params).model_dump()
    result["cache_hit"] = False
    _cache.put("generate_mask", raw_params, result)
    return result


if __name__ == "__main__":
    mcp.run()
