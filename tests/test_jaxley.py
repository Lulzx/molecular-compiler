from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from molecular_compiler import ResolutionPolicy, Stimulus, compile, simulate
from molecular_compiler.jaxley_backend import (
    build_single_cell,
    run_single_cell,
    worm_backend_trial,
)


def test_S_R4_jaxley_single_cell_voltage_and_gradient(system):
    graph, rules, kinetics = system
    policy = replace(ResolutionPolicy.default(), n_comp=1, solve_tolerance=1e-10)
    sim = compile(graph, rules, kinetics, resolution=policy)
    # Disable all network pathways to compare an isolated soma exactly.
    sim = replace(
        sim,
        gap={**sim.gap, "conductance": jnp.zeros_like(sim.gap["conductance"])},
        syn_params={**sim.syn_params, "gate": jnp.zeros_like(sim.syn_params["gate"])},
        neuromod={
            **sim.neuromod,
            "release": jnp.zeros_like(sim.neuromod["release"]),
            "sensitivity": jnp.zeros_like(sim.neuromod["sensitivity"]),
        },
    )
    record = kinetics.records["ca"]
    cell, area = build_single_cell(channels=(record,))
    currents = jnp.array([1.0, 1.0, 1.0, 1.0])

    def reference(density):
        return run_single_cell(
            cell, area, jnp.array([density]), (record,), currents, policy.dt_s
        )

    def in_house(density):
        changed = replace(
            sim,
            neuron_params={
                **sim.neuron_params,
                "channel_density": sim.neuron_params["channel_density"]
                .at[0, 0]
                .set(density),
            },
        )
        current = jnp.zeros((4, 4)).at[:, 0].set(currents)
        return simulate(changed, Stimulus(currents=current), 0.002).voltage[:, 0, 0]

    density = float(sim.neuron_params["channel_density"][0, 0])
    np.testing.assert_allclose(
        reference(density), in_house(density), atol=1e-5, rtol=1e-7
    )
    ga = jax.grad(lambda x: reference(x).sum())(density)
    gb = jax.grad(lambda x: in_house(x).sum())(density)
    np.testing.assert_allclose(ga, gb, atol=1e-5, rtol=1e-5)


def test_S_R4_trial_records_unsupported_conditions():
    forward = jax.jit(lambda: jnp.arange(4.0))
    backward = jax.jit(lambda: jnp.ones(4))
    report = worm_backend_trial(forward, forward, backward, backward)
    assert report["backend"] == "in_house"
    assert report["trajectory_match"] and report["gradient_match"]
    assert "implicit_gap" in report["failed_conditions"]
