from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler import ResolutionPolicy, compile
from molecular_compiler.schwarz import escalate_solver, with_schwarz
from molecular_compiler.solver import voltage_solve


@pytest.mark.parametrize("dense_limit", [512, 1])
def test_two_level_Schwarz_matches_dense_and_implicit_gradient(system, dense_limit):
    sim = compile(
        *system,
        resolution=replace(
            ResolutionPolicy.default(), solver="dense", solve_tolerance=1e-10
        ),
    )
    sparse = with_schwarz(sim, partitions=2, dense_limit=dense_limit, sweeps=20)
    diagonal = jnp.ones((4, 3))
    rhs = jnp.arange(12.0).reshape(4, 3)
    exact, _, _ = voltage_solve(sim, diagonal, rhs, jnp.zeros_like(rhs))
    result, residual, _ = voltage_solve(sparse, diagonal, rhs, jnp.zeros_like(rhs))
    np.testing.assert_allclose(result, exact, atol=1e-8)
    assert residual < 1e-10
    gradient = jax.grad(
        lambda b: voltage_solve(sparse, diagonal, b, jnp.zeros_like(b))[0].sum()
    )(rhs)
    direct_gradient = jax.grad(
        lambda b: voltage_solve(sim, diagonal, b, jnp.zeros_like(b))[0].sum()
    )(rhs)
    np.testing.assert_allclose(gradient, direct_gradient, atol=1e-8)


def test_profile_thresholds_select_escalation(system):
    sim = compile(*system)
    assert escalate_solver(sim, [1, 2], 0.1) is sim
    assert escalate_solver(sim, [31, 32], 0.1).schwarz_layout is not None
    assert escalate_solver(sim, [1, 2], 0.6).schwarz_layout is not None
