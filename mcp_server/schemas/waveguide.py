"""Pydantic models for all MCP tool I/O validation."""
from pydantic import BaseModel, Field, model_validator
from typing import Optional

from mcp_server.config import (
    DEFAULT_WAVELENGTH_NM,
    DEFAULT_POLARIZATION,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WIDTH_RANGE_UM,
    DEFAULT_HEIGHT_RANGE_NM,
    DEFAULT_TAPER_LENGTH_UM,
    DEFAULT_LAYER,
)


# --- Mode Solver Schemas ---
class ModeSolverInput(BaseModel):
    """Input parameters for the solve_waveguide_mode tool."""
    width_um: float = Field(..., gt=0, description="Waveguide core width in microns")
    height_nm: float = Field(..., gt=0, description="Waveguide core height in nanometers")
    core_material: str = Field(default="SiN", description="Core material: SiN, Si3N4, Si")
    cladding_material: str = Field(default="SiO2", description="Cladding material: SiO2, Air, SiN")
    wavelength_nm: float = Field(default=DEFAULT_WAVELENGTH_NM, gt=0, description="Operating wavelength in nm")
    polarization: str = Field(default=DEFAULT_POLARIZATION, pattern="^(TE|TM)$", description="TE or TM polarization")
    num_modes: int = Field(default=1, ge=1, le=10, description="Number of modes to solve")


class ModeSolverOutput(BaseModel):
    """Output from the solve_waveguide_mode tool."""
    n_eff: float = Field(description="Effective refractive index of the fundamental mode")
    confinement_factor: float = Field(description="Optical confinement factor (0 to 1)")
    mfd_um: float = Field(description="Mode field diameter in microns at 1/e² intensity")
    group_index: float = Field(description="Group index n_g")
    te_fraction: float = Field(
        description="Fraction of transverse E-field energy in Ex (near 1 for TE, near 0 for TM)"
    )
    mode_profile_path: Optional[str] = Field(default=None, description="Path to saved mode profile image")


# --- Inverse Design Schemas ---
# Metrics the analytic model expresses as fractions; their targets must lie in (0, 1].
FRACTIONAL_METRICS = ("confinement", "coupling_efficiency")


class InverseDesignInput(BaseModel):
    """Input parameters for the optimize_waveguide tool."""
    target_metric: str = Field(
        ...,
        pattern="^(insertion_loss|coupling_efficiency|propagation_loss|confinement)$",
        description="Target metric to optimize"
    )
    target_value: float = Field(..., description="Desired value for the target metric")
    wavelength_nm: float = Field(default=DEFAULT_WAVELENGTH_NM, gt=0, description="Operating wavelength in nm")
    polarization: str = Field(
        default=DEFAULT_POLARIZATION, pattern="^(TE|TM)$",
        description="TE or TM polarization (recorded; the analytic model is polarization-independent)",
    )
    constraints: dict = Field(
        default_factory=dict,
        description="Fixed process parameters e.g. {'deposition': 'LPCVD'} (recorded; not used by the analytic model)",
    )
    width_range_um: tuple[float, float] = Field(default=DEFAULT_WIDTH_RANGE_UM, description="Width search bounds in µm")
    height_range_nm: tuple[float, float] = Field(default=DEFAULT_HEIGHT_RANGE_NM, description="Height search bounds in nm")
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1, le=5000, description="Max optimization iterations")
    learning_rate: float = Field(default=DEFAULT_LEARNING_RATE, gt=0, description="Gradient descent learning rate")

    @model_validator(mode="after")
    def _target_within_model_range(self):
        if self.target_metric in FRACTIONAL_METRICS:
            if not 0.0 < self.target_value <= 1.0:
                raise ValueError(
                    f"{self.target_metric} target must be in (0, 1], got {self.target_value}"
                )
        elif self.target_value <= 0.0:
            raise ValueError(f"{self.target_metric} target must be positive, got {self.target_value}")
        for name, (low, high) in (("width_range_um", self.width_range_um),
                                  ("height_range_nm", self.height_range_nm)):
            if not 0.0 < low < high:
                raise ValueError(f"{name} must satisfy 0 < min < max, got ({low}, {high})")
        return self


class InverseDesignOutput(BaseModel):
    """Output from the optimize_waveguide tool."""
    optimized_width_um: float = Field(description="Optimized waveguide width in microns")
    optimized_height_nm: float = Field(description="Optimized waveguide height in nanometers")
    achieved_value: float = Field(description="Achieved value of the target metric")
    convergence_history: list[float] = Field(description="Loss value at each iteration")
    iterations: int = Field(description="Number of iterations run")
    converged: bool = Field(
        description="True if the target was reached within tolerance before max_iterations"
    )


# --- Mask Generation Schemas ---
class MaskGenInput(BaseModel):
    """Input parameters for the generate_mask tool."""
    width_um: float = Field(..., gt=0, description="Waveguide width in microns")
    height_nm: float = Field(..., gt=0, description="Waveguide height in nm (stored as metadata)")
    length_mm: float = Field(..., gt=0, description="Total waveguide length in mm")
    io_type: str = Field(default="edge_coupler", pattern="^(edge_coupler|grating_coupler)$")
    taper_length_um: float = Field(default=DEFAULT_TAPER_LENGTH_UM, gt=0, description="Taper length in microns")
    layer: tuple[int, int] = Field(default=DEFAULT_LAYER, description="GDS layer and datatype")
    output_filename: str = Field(
        default="waveguide_design.gds",
        # Bare filename only — path separators would let a caller write
        # outside the configured GDS output directory.
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gds$",
        description="Output GDS filename (bare filename, no directories)",
    )


class MaskGenOutput(BaseModel):
    """Output from the generate_mask tool."""
    gds_file_path: str = Field(description="Path to the generated GDSII file")
    cell_name: str = Field(description="Top-level cell name")
    total_length_um: float = Field(description="Total physical length in microns")
    bounding_box: tuple[tuple[float, float], tuple[float, float]] = Field(
        description="Bounding box as ((x_min, y_min), (x_max, y_max)) in µm"
    )
