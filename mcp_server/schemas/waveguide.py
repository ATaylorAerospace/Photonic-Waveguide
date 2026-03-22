"""Pydantic models for all MCP tool I/O validation."""
from pydantic import BaseModel, Field
from typing import Optional


# --- Mode Solver Schemas ---
class ModeSolverInput(BaseModel):
    """Input parameters for the solve_waveguide_mode tool."""
    width_um: float = Field(..., gt=0, description="Waveguide core width in microns")
    height_nm: float = Field(..., gt=0, description="Waveguide core height in nanometers")
    core_material: str = Field(default="SiN", description="Core material: SiN, Si3N4, Si")
    cladding_material: str = Field(default="SiO2", description="Cladding material: SiO2, Air, SiN")
    wavelength_nm: float = Field(default=1550.0, gt=0, description="Operating wavelength in nm")
    polarization: str = Field(default="TE", pattern="^(TE|TM)$", description="TE or TM polarization")
    num_modes: int = Field(default=1, ge=1, le=10, description="Number of modes to solve")


class ModeSolverOutput(BaseModel):
    """Output from the solve_waveguide_mode tool."""
    n_eff: float = Field(description="Effective refractive index of the fundamental mode")
    confinement_factor: float = Field(description="Optical confinement factor (0 to 1)")
    mfd_um: float = Field(description="Mode field diameter in microns at 1/e² intensity")
    group_index: float = Field(description="Group index n_g")
    mode_profile_path: Optional[str] = Field(default=None, description="Path to saved mode profile image")


# --- Inverse Design Schemas ---
class InverseDesignInput(BaseModel):
    """Input parameters for the optimize_waveguide tool."""
    target_metric: str = Field(
        ...,
        pattern="^(insertion_loss|coupling_efficiency|propagation_loss|confinement)$",
        description="Target metric to optimize"
    )
    target_value: float = Field(..., description="Desired value for the target metric")
    wavelength_nm: float = Field(default=1550.0, gt=0, description="Operating wavelength in nm")
    polarization: str = Field(default="TE", pattern="^(TE|TM)$", description="TE or TM polarization")
    constraints: dict = Field(default_factory=dict, description="Fixed parameters e.g. {'deposition': 'LPCVD'}")
    width_range_um: tuple[float, float] = Field(default=(0.3, 5.0), description="Width search bounds in µm")
    height_range_nm: tuple[float, float] = Field(default=(100.0, 800.0), description="Height search bounds in nm")
    max_iterations: int = Field(default=200, ge=1, le=5000, description="Max optimization iterations")
    learning_rate: float = Field(default=0.01, gt=0, description="Gradient descent learning rate")


class InverseDesignOutput(BaseModel):
    """Output from the optimize_waveguide tool."""
    optimized_width_um: float = Field(description="Optimized waveguide width in microns")
    optimized_height_nm: float = Field(description="Optimized waveguide height in nanometers")
    achieved_value: float = Field(description="Achieved value of the target metric")
    convergence_history: list[float] = Field(description="Loss value at each iteration")
    iterations: int = Field(description="Number of iterations to converge")


# --- Mask Generation Schemas ---
class MaskGenInput(BaseModel):
    """Input parameters for the generate_mask tool."""
    width_um: float = Field(..., gt=0, description="Waveguide width in microns")
    height_nm: float = Field(..., gt=0, description="Waveguide height in nm (stored as metadata)")
    length_mm: float = Field(..., gt=0, description="Total waveguide length in mm")
    bend_radius_um: float = Field(default=50.0, gt=0, description="Bend radius in microns")
    io_type: str = Field(default="edge_coupler", pattern="^(edge_coupler|grating_coupler)$")
    taper_length_um: float = Field(default=200.0, gt=0, description="Taper length in microns")
    routing: str = Field(default="bezier", pattern="^(manhattan|bezier)$")
    layer: tuple[int, int] = Field(default=(1, 0), description="GDS layer and datatype")
    output_filename: str = Field(default="waveguide_design.gds", description="Output GDS filename")


class MaskGenOutput(BaseModel):
    """Output from the generate_mask tool."""
    gds_file_path: str = Field(description="Path to the generated GDSII file")
    cell_name: str = Field(description="Top-level cell name")
    total_length_um: float = Field(description="Total physical length in microns")
    num_bends: int = Field(description="Number of bends in the layout")
    bounding_box: tuple[tuple[float, float], tuple[float, float]] = Field(
        description="Bounding box as ((x_min, y_min), (x_max, y_max)) in µm"
    )
