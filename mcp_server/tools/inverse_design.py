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

    def optimize(self, inputs: InverseDesignInput) -> InverseDesignOutput:
        """Run gradient-based inverse design optimization."""
        wavelength_um = inputs.wavelength_nm / 1000.0
        w_lo, w_hi = inputs.width_range_um
        h_lo, h_hi = inputs.height_range_nm

        # Optimize in [0, 1]-normalized coordinates so both parameters share
        # one well-conditioned learning rate regardless of their units.
        def denormalize(p):
            return {
                "width_um": w_lo + p[0] * (w_hi - w_lo),
                "height_nm": h_lo + p[1] * (h_hi - h_lo),
            }

        value_and_grad = jax.jit(jax.value_and_grad(
            lambda p: self._loss_fn(
                denormalize(p), inputs.target_metric, inputs.target_value, wavelength_um
            )
        ))

        p = jnp.array([0.5, 0.5])
        convergence_history = []
        converged_iter = inputs.max_iterations

        for i in range(inputs.max_iterations):
            loss_val, grads = value_and_grad(p)
            loss_val = float(loss_val)
            convergence_history.append(loss_val)
            if loss_val < 1e-8:
                converged_iter = i + 1
                break
            p = jnp.clip(p - inputs.learning_rate * grads, 0.0, 1.0)

        params = denormalize(p)
        final_model = self._waveguide_model(params, wavelength_um)
        achieved = float(final_model[inputs.target_metric])

        return InverseDesignOutput(
            optimized_width_um=float(params["width_um"]),
            optimized_height_nm=float(params["height_nm"]),
            achieved_value=achieved,
            convergence_history=convergence_history,
            iterations=converged_iter,
        )
