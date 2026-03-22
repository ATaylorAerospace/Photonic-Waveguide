"""Waveguide mode solver using modesolverpy."""
import numpy as np
import modesolverpy.mode_solver as ms
import modesolverpy.structure as st
from mcp_server.config import (
    MATERIAL_INDEX,
    DEFAULT_X_STEP_UM,
    DEFAULT_Y_STEP_UM,
    DEFAULT_BOUNDARY_UM,
)
from mcp_server.schemas.waveguide import ModeSolverInput, ModeSolverOutput


class WaveguideSolver:
    """Wraps modesolverpy to solve waveguide eigenmodes."""

    def __init__(self, x_step: float = DEFAULT_X_STEP_UM, y_step: float = DEFAULT_Y_STEP_UM):
        self.x_step = x_step
        self.y_step = y_step

    def _get_refractive_index(self, material: str) -> float:
        key = material.strip()
        if key not in MATERIAL_INDEX:
            raise ValueError(f"Unknown material: {material}. Available: {list(MATERIAL_INDEX.keys())}")
        return MATERIAL_INDEX[key]

    def solve(self, params: ModeSolverInput) -> ModeSolverOutput:
        """Run fully vectorial eigenmode expansion for the given waveguide geometry."""
        n_core = self._get_refractive_index(params.core_material)
        n_clad = self._get_refractive_index(params.cladding_material)
        wl_um = params.wavelength_nm / 1000.0
        height_um = params.height_nm / 1000.0

        structure = st.RidgeWaveguide(
            wavelength=wl_um,
            x_step=self.x_step,
            y_step=self.y_step,
            wg_height=height_um,
            wg_width=params.width_um,
            sub_height=DEFAULT_BOUNDARY_UM,
            sub_width=params.width_um + 2 * DEFAULT_BOUNDARY_UM,
            clad_height=DEFAULT_BOUNDARY_UM,
            n_sub=n_clad,
            n_wg=n_core,
            n_clad=n_clad,
            film_thickness=height_um,
            angle=90.0,
        )

        solver = ms.ModeSolverFullyVectorial(params.num_modes)
        solver.solve(structure)

        n_eff = float(np.real(solver.n_effs[0]))

        mode_field = solver.fields["Ex" if params.polarization == "TE" else "Ey"][0]
        intensity = np.abs(mode_field) ** 2
        total_power = np.sum(intensity)

        x_pts = structure.x
        y_pts = structure.y
        core_x_mask = np.abs(x_pts) <= params.width_um / 2
        core_y_mask = (y_pts >= 0) & (y_pts <= height_um)
        core_intensity = intensity[np.ix_(core_y_mask, core_x_mask)]
        confinement = float(np.sum(core_intensity) / total_power) if total_power > 0 else 0.0

        peak = np.max(intensity)
        threshold = peak / (np.e ** 2)
        above_threshold = intensity >= threshold
        x_extent = np.sum(np.any(above_threshold, axis=0)) * self.x_step
        y_extent = np.sum(np.any(above_threshold, axis=1)) * self.y_step
        mfd_um = float(np.sqrt(x_extent * y_extent))

        delta_wl = 0.001
        structure_plus = st.RidgeWaveguide(
            wavelength=wl_um + delta_wl,
            x_step=self.x_step, y_step=self.y_step,
            wg_height=height_um, wg_width=params.width_um,
            sub_height=DEFAULT_BOUNDARY_UM,
            sub_width=params.width_um + 2 * DEFAULT_BOUNDARY_UM,
            clad_height=DEFAULT_BOUNDARY_UM,
            n_sub=n_clad, n_wg=n_core, n_clad=n_clad,
            film_thickness=height_um, angle=90.0,
        )
        solver_plus = ms.ModeSolverFullyVectorial(1)
        solver_plus.solve(structure_plus)
        n_eff_plus = float(np.real(solver_plus.n_effs[0]))
        dn_dwl = (n_eff_plus - n_eff) / delta_wl
        group_index = float(n_eff - wl_um * dn_dwl)

        return ModeSolverOutput(
            n_eff=n_eff,
            confinement_factor=confinement,
            mfd_um=mfd_um,
            group_index=group_index,
            mode_profile_path=None,
        )
