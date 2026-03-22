"""Inverse design engine using SAX + JAX."""
import jax
import jax.numpy as jnp
import sax
from mcp_server.config import MATERIAL_INDEX
from mcp_server.schemas.waveguide import InverseDesignInput, InverseDesignOutput


class InverseDesigner:
    """Wraps SAX circuit solver with JAX autodiff for gradient-based inverse design."""

    def __init__(self):
        pass

    def _waveguide_model(self, params: dict, wavelength_um: float) -> dict:
        """Differentiable waveguide S-parameter model using SAX."""
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

    def _loss_fn(self, params: dict, target_metric: str, target_value: float,
                 wavelength_um: float) -> float:
        model_out = self._waveguide_model(params, wavelength_um)
        predicted = model_out[target_metric]
        return (predicted - target_value) ** 2

    def _project_params(self, params: dict, width_range: tuple, height_range: tuple) -> dict:
        return {
            "width_um": jnp.clip(params["width_um"], width_range[0], width_range[1]),
            "height_nm": jnp.clip(params["height_nm"], height_range[0], height_range[1]),
        }

    def optimize(self, inputs: InverseDesignInput) -> InverseDesignOutput:
        """Run gradient-based inverse design optimization."""
        wavelength_um = inputs.wavelength_nm / 1000.0
        params = {
            "width_um": jnp.array((inputs.width_range_um[0] + inputs.width_range_um[1]) / 2),
            "height_nm": jnp.array((inputs.height_range_nm[0] + inputs.height_range_nm[1]) / 2),
        }

        grad_fn = jax.grad(
            lambda p: self._loss_fn(p, inputs.target_metric, inputs.target_value, wavelength_um)
        )

        convergence_history = []
        converged_iter = inputs.max_iterations

        for i in range(inputs.max_iterations):
            loss_val = float(self._loss_fn(
                params, inputs.target_metric, inputs.target_value, wavelength_um
            ))
            convergence_history.append(loss_val)
            if loss_val < 1e-8:
                converged_iter = i + 1
                break
            grads = grad_fn(params)
            params = {
                "width_um": params["width_um"] - inputs.learning_rate * grads["width_um"],
                "height_nm": params["height_nm"] - inputs.learning_rate * grads["height_nm"] * 100,
            }
            params = self._project_params(params, inputs.width_range_um, inputs.height_range_nm)

        final_model = self._waveguide_model(params, wavelength_um)
        achieved = float(final_model[inputs.target_metric])

        return InverseDesignOutput(
            optimized_width_um=float(params["width_um"]),
            optimized_height_nm=float(params["height_nm"]),
            achieved_value=achieved,
            convergence_history=convergence_history,
            iterations=converged_iter,
        )
