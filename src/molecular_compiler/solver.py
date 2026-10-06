"""M8 voltage solve: dense reference and matrix-free block-Jacobi PCG.

Capacitance pF, voltage mV, conductance nS, current pA, time ms.
Gradients use an adjoint solve, never differentiated CG iterations.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve


def coupling_edges(sim):
    n, c = sim.n_neurons, sim.resolution.n_comp
    base = jnp.arange(n, dtype=jnp.int32)[:, None] * c
    left = (base + jnp.arange(c - 1)[None]).ravel()
    right = left + 1
    return (
        jnp.concatenate([left, sim.gap["i"] * c]),
        jnp.concatenate([right, sim.gap["j"] * c]),
        jnp.concatenate(
            [jnp.full(left.shape, sim.resolution.axial_nS), sim.gap["conductance"]]
        ),
    )


def laplacian_product(x, i, j, g):
    flow = g * (x[i] - x[j])
    return jnp.zeros_like(x).at[i].add(flow).at[j].add(-flow)


def pcg(matvec, rhs, precondition, x0, tolerance, max_iterations):
    residual = rhs - matvec(x0)
    z = precondition(residual)
    rz = jnp.vdot(residual, z).real
    threshold = tolerance * jnp.maximum(jnp.linalg.norm(rhs), 1e-12)
    initial = (jnp.array(0), x0, residual, z, rz)

    def condition(carry):
        k, _, r, _, _ = carry
        return (k < max_iterations) & (jnp.linalg.norm(r) > threshold)

    def iteration(carry):
        k, x, r, p, old_rz = carry
        ap = matvec(p)
        alpha = old_rz / jnp.maximum(jnp.vdot(p, ap).real, 1e-30)
        xx, rr = x + alpha * p, r - alpha * ap
        zz = precondition(rr)
        new_rz = jnp.vdot(rr, zz).real
        beta = new_rz / jnp.maximum(old_rz, 1e-30)
        return k + 1, xx, rr, zz + beta * p, new_rz

    count, result, _, _, _ = jax.lax.while_loop(condition, iteration, initial)
    return result, count


def voltage_solve(sim, diagonal, rhs, old_voltage):
    if sim.partition_plan is not None:
        from .distributed import distributed_voltage_solve

        return distributed_voltage_solve(
            sim, sim.partition_plan, diagonal, rhs, old_voltage
        )
    policy = sim.resolution
    n, c = diagonal.shape
    size = n * c
    i, j, g = coupling_edges(sim)
    g = g.astype(diagonal.dtype)
    diag = diagonal.ravel()
    vector = rhs.ravel()

    def matvec(x):
        return diag * x + laplacian_product(x, i, j, g)

    method = policy.solver
    if method == "auto":
        method = "dense" if size <= policy.dense_limit else "pcg"
    if method == "dense":
        matrix = jnp.diag(diag)
        matrix = (
            matrix.at[i, i].add(g).at[j, j].add(g).at[i, j].add(-g).at[j, i].add(-g)
        )
        factor = jnp.linalg.cholesky(matrix)

        def solve(_, b):
            return cho_solve((factor, True), b), jnp.array(1)
    else:
        blocks = jax.vmap(jnp.diag)(diagonal)
        gap_diag = (
            jnp.zeros(size, dtype=diagonal.dtype)
            .at[sim.gap["i"] * c]
            .add(sim.gap["conductance"])
            .at[sim.gap["j"] * c]
            .add(sim.gap["conductance"])
        )
        blocks += jax.vmap(jnp.diag)(gap_diag.reshape(n, c))
        for k in range(c - 1):
            blocks = (
                blocks.at[:, k, k]
                .add(policy.axial_nS)
                .at[:, k + 1, k + 1]
                .add(policy.axial_nS)
            )
            blocks = (
                blocks.at[:, k, k + 1]
                .add(-policy.axial_nS)
                .at[:, k + 1, k]
                .add(-policy.axial_nS)
            )
        factors = jnp.linalg.cholesky(blocks)

        def precondition(r):
            return jax.vmap(lambda a, b: cho_solve((a, True), b))(
                factors, r.reshape(n, c)
            ).ravel()

        if sim.schwarz_layout is not None:
            from .schwarz import make_preconditioner

            precondition = make_preconditioner(sim, diagonal)

        def solve(_, b):
            return pcg(
                matvec,
                b,
                precondition,
                old_voltage.ravel(),
                policy.solve_tolerance,
                policy.max_cg_iterations,
            )

    result, iterations = jax.lax.custom_linear_solve(
        matvec, vector, solve=solve, symmetric=True, has_aux=True
    )
    residual = jnp.linalg.norm(matvec(result) - vector) / jnp.maximum(
        jnp.linalg.norm(vector), 1e-12
    )
    return result.reshape(n, c), residual, iterations
