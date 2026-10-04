"""Inverse design engine: gradient descent with JAX autodiff on an analytic waveguide model."""
from functools import partial

import jax
import jax.numpy as jnp

from mcp_server.config import MATERIAL_INDEX
from mcp_server.schemas.waveguide import InverseDesignInput, InverseDesignOutput

# Squared error below which the target counts as reached.
CONVERGENCE_TOL = 1e-8


def _waveguide_model(params: dict, wavelength_um) -> dict:
    """Differentiable analytic waveguide model (closed-form V-number heuristics)."""
    width = params["width_um"]
    height_um = params["height_nm"] / 1000.0
    n_core = MATERIAL_INDEX["SiN"]
    n_clad = MATERIAL_INDEX["SiO2"]

    V = (2 * jnp.pi / wavelength_um) * height_um * jnp.sqrt(n_core**2 - n_clad**2)
    n_eff_approx = n_clad + (n_core - n_clad) * (1 - jnp.exp(-V / 2))

    V_width = (2 * jnp.pi / wavelength_um) * width * jnp.sqrt(n_core**2 - n_clad**2)
    confinement = 1 - jnp.exp(-V_width / 2)

    prop_loss = 0.1 + 2.0 * (1 - confinement) ** 2
    insertion_loss = 0.5 * (1 - confinement) + prop_loss * 0.1
    coupling_eff = confinement * jnp.exp(-insertion_loss / 10)

    return {
        "n_eff": n_eff_approx,
        "confinement": confinement,
        "propagation_loss": prop_loss,
        "insertion_loss": insertion_loss,
        "coupling_efficiency": coupling_eff,
    }


def _denormalize(p, bounds):
    """Map [0, 1]^2 optimizer coordinates onto the (width_um, height_nm) bounds."""
    return {
        "width_um": bounds[0] + p[0] * (bounds[1] - bounds[0]),
        "height_nm": bounds[2] + p[1] * (bounds[3] - bounds[2]),
    }


def _loss(target_metric: str, p, target_value, wavelength_um, bounds):
    predicted = _waveguide_model(_denormalize(p, bounds), wavelength_um)[target_metric]
    return (predicted - target_value) ** 2


class InverseDesigner:
    """Differentiable waveguide surrogate optimized with JAX gradients."""

    def __init__(self):
        # One compiled value-and-gradient function per metric. Target, wavelength
        # and bounds are traced arguments, so later requests reuse the
        # compilation instead of paying for it on every call.
        self._compiled = {}

    def _value_and_grad(self, target_metric: str):
        if target_metric not in self._compiled:
            self._compiled[target_metric] = jax.jit(
                jax.value_and_grad(partial(_loss, target_metric))
            )
        return self._compiled[target_metric]

    def optimize(self, inputs: InverseDesignInput) -> InverseDesignOutput:
        """Run gradient-based inverse design optimization."""
        wavelength_um = inputs.wavelength_nm / 1000.0
        # Optimizing in [0, 1]-normalized coordinates lets both parameters share
        # one well-conditioned learning rate regardless of their units.
        bounds = jnp.array([*inputs.width_range_um, *inputs.height_range_nm])
        value_and_grad = self._value_and_grad(inputs.target_metric)

        p = jnp.array([0.5, 0.5])
        convergence_history = []
        converged = False
        iterations = inputs.max_iterations

        for i in range(inputs.max_iterations):
            loss_val, grads = value_and_grad(p, inputs.target_value, wavelength_um, bounds)
            loss_val = float(loss_val)
            convergence_history.append(loss_val)
            if loss_val < CONVERGENCE_TOL:
                converged = True
                iterations = i + 1
                break
            p = jnp.clip(p - inputs.learning_rate * grads, 0.0, 1.0)

        params = _denormalize(p, bounds)
        achieved = float(_waveguide_model(params, wavelength_um)[inputs.target_metric])

        return InverseDesignOutput(
            optimized_width_um=float(params["width_um"]),
            optimized_height_nm=float(params["height_nm"]),
            achieved_value=achieved,
            convergence_history=convergence_history,
            iterations=iterations,
            converged=converged,
        )
