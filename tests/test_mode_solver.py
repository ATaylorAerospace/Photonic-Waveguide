"""Tests for the modesolverpy-backed mode solver (skipped when modesolverpy is absent)."""
import pytest

mode_solver = pytest.importorskip("mcp_server.tools.mode_solver")

from mcp_server.config import MATERIAL_INDEX  # noqa: E402
from mcp_server.schemas.waveguide import ModeSolverInput  # noqa: E402

# A coarse grid keeps each eigen-solve well under a second.
COARSE_STEP_UM = 0.1


@pytest.fixture(scope="module")
def results():
    solver = mode_solver.WaveguideSolver(x_step=COARSE_STEP_UM, y_step=COARSE_STEP_UM)
    return {
        pol: solver.solve(ModeSolverInput(width_um=1.5, height_nm=400, polarization=pol))
        for pol in ("TE", "TM")
    }


def test_te_mode_is_physical(results):
    te = results["TE"]
    assert MATERIAL_INDEX["SiO2"] < te.n_eff < MATERIAL_INDEX["SiN"]
    assert 0.0 < te.confinement_factor < 1.0
    assert te.mfd_um > 0.0
    assert te.group_index > te.n_eff
    assert te.te_fraction > 0.9


def test_tm_request_returns_the_tm_mode(results):
    te, tm = results["TE"], results["TM"]
    assert tm.te_fraction < 0.1
    assert tm.n_eff < te.n_eff
    assert 0.0 < tm.confinement_factor < 1.0


def test_field_grid_matches_cell_centres():
    # The confinement masks are built on the solver's cell-centre grids; a
    # mismatch between those and the returned field arrays is a hard error.
    solver = mode_solver.WaveguideSolver(x_step=COARSE_STEP_UM, y_step=COARSE_STEP_UM)
    params = ModeSolverInput(width_um=1.0, height_nm=300)
    structure = solver._build_structure(params, 1.55, MATERIAL_INDEX["SiN"], MATERIAL_INDEX["SiO2"])
    inner, index = solver._solve_polarized(structure, "TE", 1)
    field = inner.modes[index].fields["Ex"]
    assert field.shape == (structure.yc.size, structure.xc.size)
