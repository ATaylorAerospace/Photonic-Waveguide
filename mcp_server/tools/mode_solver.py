"""Waveguide mode solver using modesolverpy."""
import numpy as np

from mcp_server.tools import _modesolverpy_compat

_modesolverpy_compat.install()

import modesolverpy.mode_solver as ms  # noqa: E402  (must follow the shim)
import modesolverpy.structure as st  # noqa: E402

from mcp_server.config import (  # noqa: E402
    MATERIAL_INDEX,
    DEFAULT_X_STEP_UM,
    DEFAULT_Y_STEP_UM,
    DEFAULT_BOUNDARY_UM,
)
from mcp_server.schemas.waveguide import ModeSolverInput, ModeSolverOutput  # noqa: E402

# modesolverpy labels each solved mode by its dominant transverse E component.
_MODE_LABEL = {"TE": "qTE", "TM": "qTM"}
_DOMINANT_FIELD = {"TE": "Ex", "TM": "Ey"}
# Wavelength step for the finite-difference dn_eff/dλ in the group index.
_GROUP_INDEX_DELTA_UM = 0.001


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

    def _build_structure(self, params: ModeSolverInput, wavelength_um: float,
                         n_core: float, n_clad: float) -> st.RidgeWaveguide:
        height_um = params.height_nm / 1000.0
        return st.RidgeWaveguide(
            wavelength=wavelength_um,
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

    def _solve_polarized(self, structure: st.RidgeWaveguide, polarization: str, num_modes: int):
        """Solve and return (solver, index of the lowest-order mode of this polarization).

        Modes come back sorted by n_eff, so index 0 is simply the fundamental
        mode (normally quasi-TE); a TM request has to look further down the list.
        """
        solver = ms.ModeSolverFullyVectorial(max(num_modes, 2))
        solver.solve(structure)
        wanted = _MODE_LABEL[polarization]
        for index, (label, _) in enumerate(solver.mode_types):
            if label == wanted:
                return solver, index
        raise ValueError(
            f"No {polarization} mode among the {len(solver.mode_types)} lowest-order "
            f"modes of this geometry; increase num_modes"
        )

    def solve(self, params: ModeSolverInput) -> ModeSolverOutput:
        """Run fully vectorial eigenmode expansion for the given waveguide geometry."""
        n_core = self._get_refractive_index(params.core_material)
        n_clad = self._get_refractive_index(params.cladding_material)
        wl_um = params.wavelength_nm / 1000.0
        height_um = params.height_nm / 1000.0

        structure = self._build_structure(params, wl_um, n_core, n_clad)
        solver, index = self._solve_polarized(structure, params.polarization, params.num_modes)
        n_eff = float(np.real(solver.n_effs[index]))
        te_fraction = float(solver.fraction_te[index])

        intensity = np.abs(solver.modes[index].fields[_DOMINANT_FIELD[params.polarization]]) ** 2
        # Fields are sampled on the cell-centre grids (structure.xc / .yc), one
        # point shorter than the edge grids on each axis, and indexed [y, x].
        xc, yc = structure.xc, structure.yc
        if intensity.shape != (yc.size, xc.size):
            raise ValueError(
                f"Mode field shape {intensity.shape} does not match the cell-centre "
                f"grid ({yc.size}, {xc.size})"
            )
        total_power = float(np.sum(intensity))
        # The core is centred horizontally and sits on the substrate slab of
        # thickness DEFAULT_BOUNDARY_UM.
        core_x = np.abs(xc - structure.x_ctr) <= params.width_um / 2
        core_y = (yc >= DEFAULT_BOUNDARY_UM) & (yc <= DEFAULT_BOUNDARY_UM + height_um)
        core_power = float(np.sum(intensity[np.ix_(core_y, core_x)]))
        confinement = core_power / total_power if total_power > 0 else 0.0

        threshold = np.max(intensity) / (np.e ** 2)
        above_threshold = intensity >= threshold
        x_extent = np.sum(np.any(above_threshold, axis=0)) * self.x_step
        y_extent = np.sum(np.any(above_threshold, axis=1)) * self.y_step
        mfd_um = float(np.sqrt(x_extent * y_extent))

        structure_plus = self._build_structure(params, wl_um + _GROUP_INDEX_DELTA_UM, n_core, n_clad)
        solver_plus, index_plus = self._solve_polarized(
            structure_plus, params.polarization, params.num_modes
        )
        n_eff_plus = float(np.real(solver_plus.n_effs[index_plus]))
        dn_dwl = (n_eff_plus - n_eff) / _GROUP_INDEX_DELTA_UM
        group_index = float(n_eff - wl_um * dn_dwl)

        return ModeSolverOutput(
            n_eff=n_eff,
            confinement_factor=confinement,
            mfd_um=mfd_um,
            group_index=group_index,
            te_fraction=te_fraction,
            mode_profile_path=None,
        )
