"""S-R4: Jaxley mechanisms, full single-cell backend and numerical trial.

The first-order voltage gate is an explicit kinetic model, not a fitted
replacement for an unspecified molecule. It is valid only for records that
select this model. More detailed Jaxley mechanisms can be supplied separately.
"""

import time
from functools import lru_cache

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from jaxley.channels import Channel, Leak


class MolecularChannel(Channel):
    def __init__(self, name="Mol"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{name}_g": 1e-4,
            f"{name}_e": 0.0,
            f"{name}_tau_ms": 10.0,
            f"{name}_vhalf": -35.0,
            f"{name}_slope": 10.0,
        }
        self.channel_states = {f"{name}_a": 0.1}
        self.current_name = f"i_{name}"

    def update_states(self, states, dt, v, params):
        name = self.name
        target = jax.nn.sigmoid((v - params[f"{name}_vhalf"]) / params[f"{name}_slope"])
        updated = target + (states[f"{name}_a"] - target) * jnp.exp(
            -dt / params[f"{name}_tau_ms"]
        )
        return {f"{name}_a": updated}

    def compute_current(self, states, v, params):
        name = self.name
        return params[f"{name}_g"] * states[f"{name}_a"] * (v - params[f"{name}_e"])

    def init_state(self, states, v, params, delta_t):
        name = self.name
        return {
            f"{name}_a": jax.nn.sigmoid(
                (v - params[f"{name}_vhalf"]) / params[f"{name}_slope"]
            )
        }


@lru_cache(maxsize=1)
def default_mechanism():
    return MolecularChannel()


def gate_step(record, voltage, old, dt_s, temperature_C):
    mechanism = default_mechanism()
    params = {
        **mechanism.channel_params,
        "Mol_tau_ms": record.time_constant(temperature_C) * 1000,
        "Mol_vhalf": record.parameter("v_half_mV", record.v_half_mV),
        "Mol_slope": record.parameter("slope_mV", record.slope_mV),
    }
    return mechanism.update_states({"Mol_a": old}, dt_s * 1000, voltage, params)[
        "Mol_a"
    ]


def build_single_cell(
    capacitance_pF=1.0,
    leak_nS=0.02,
    leak_mV=-60.0,
    channels=(),
    radius_um=1.0,
    length_um=10.0,
):
    """Build a full Jaxley cell once, outside the gradient path."""
    cell = jx.Compartment()
    area_cm2 = 2 * np.pi * radius_um * length_um * 1e-8
    cell.set("radius", radius_um)
    cell.set("length", length_um)
    cell.set("capacitance", capacitance_pF * 1e-6 / area_cm2)
    cell.set("v", leak_mV)
    cell.insert(Leak())
    cell.set("Leak_gLeak", leak_nS * 1e-9 / area_cm2)
    cell.set("Leak_eLeak", leak_mV)
    for index, record in enumerate(channels):
        name = f"Mol{index}"
        cell.insert(MolecularChannel(name))
        cell.set(f"{name}_tau_ms", record.time_constant(20) * 1000)
        cell.set(f"{name}_vhalf", record.v_half_mV)
        cell.set(f"{name}_slope", record.slope_mV)
        cell.set(
            f"{name}_e", record.reversal_mV if record.reversal_mV is not None else 0.0
        )
        cell.set(f"{name}_g", record.conductance * 1e-9 / area_cm2)
    cell.init_states()
    cell.record()
    return cell, area_cm2


def run_single_cell(cell, area_cm2, densities, channels, currents_pA, dt_s):
    """Rule outputs enter Jaxley with data_set inside one JAX program."""
    parameters = None
    for index, record in enumerate(channels):
        parameters = cell.data_set(
            f"Mol{index}_g",
            densities[index] * record.conductance * 1e-9 / area_cm2,
            parameters,
        )
    stimuli = cell.data_stimulate(jnp.asarray(currents_pA) / 1000)
    voltage = jx.integrate(
        cell,
        param_state=parameters,
        data_stimuli=stimuli,
        delta_t=dt_s * 1000,
        solver="bwd_euler",
    )
    return voltage[0, 1:]


def worm_backend_trial(
    reference,
    in_house,
    gradients_reference,
    gradients_in_house,
    implicit_gap=False,
    factorized_peptides=False,
    runtime_switching=False,
    repeats=3,
    tolerance=1e-5,
):
    """Measure available interfaces; unsupported capabilities fail explicitly."""
    a, b = np.asarray(reference()), np.asarray(in_house())
    ga, gb = np.asarray(gradients_reference()), np.asarray(gradients_in_house())
    trajectory_match = bool(np.allclose(a, b, atol=tolerance, rtol=tolerance))
    gradient_match = bool(np.allclose(ga, gb, atol=tolerance, rtol=tolerance))

    def benchmark(fn):
        jax.block_until_ready(fn())
        start = time.perf_counter()
        for _ in range(repeats):
            jax.block_until_ready(fn())
        return (time.perf_counter() - start) / repeats

    timings = {
        "jaxley_forward_s": benchmark(reference),
        "in_house_forward_s": benchmark(in_house),
        "jaxley_backward_s": benchmark(gradients_reference),
        "in_house_backward_s": benchmark(gradients_in_house),
    }
    ratio_forward = timings["jaxley_forward_s"] / max(
        timings["in_house_forward_s"], 1e-12
    )
    ratio_backward = timings["jaxley_backward_s"] / max(
        timings["in_house_backward_s"], 1e-12
    )
    checks = {
        "implicit_gap": bool(implicit_gap),
        "factorized_peptides": bool(factorized_peptides),
        "runtime_switching": bool(runtime_switching),
        "differentiable_parameter_bridge": gradient_match and trajectory_match,
        "within_2x_cost": ratio_forward <= 2 and ratio_backward <= 2,
    }
    return {
        "version": jx.__version__,
        "scope": "single_cell_reference_and_capability_trial",
        "backend": "jaxley" if all(checks.values()) else "in_house",
        "checks": checks,
        "failed_conditions": [k for k, v in checks.items() if not v],
        "timings": timings,
        "trajectory_match": trajectory_match,
        "gradient_match": gradient_match,
        "forward_ratio": ratio_forward,
        "backward_ratio": ratio_backward,
    }
