"""Two-level overlapping additive Schwarz escalation for the voltage solver.

Small local/coarse systems use Cholesky. Large local systems use a fixed
symmetric Richardson polynomial, preserving a linear SPD preconditioner.
The layout is built outside the differentiable program; conductances remain
live JAX values inside the solve.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve

from .solver import coupling_edges


@dataclass(frozen=True)
class SchwarzBlock:
    indices: np.ndarray
    diagonal_rows: np.ndarray
    diagonal_edges: np.ndarray
    i: np.ndarray
    j: np.ndarray
    edge_indices: np.ndarray


@dataclass(frozen=True)
class SchwarzLayout:
    blocks: tuple
    owner: np.ndarray
    counts: np.ndarray
    coarse_i: np.ndarray
    coarse_j: np.ndarray
    coarse_edges: np.ndarray
    dense_limit: int = 512
    sweeps: int = 12


def build_layout(sim, partitions=4, dense_limit=512, sweeps=12):
    n, c = sim.n_neurons, sim.resolution.n_comp
    if partitions < 1 or dense_limit < 1 or sweeps < 1:
        raise ValueError("invalid Schwarz configuration")
    partitions = min(partitions, n)
    i, j, _ = coupling_edges(sim)
    i, j = np.asarray(i), np.asarray(j)
    neuron_owner = np.minimum(np.arange(n) * partitions // n, partitions - 1)
    owner = np.repeat(neuron_owner, c)
    blocks = []
    # One-hop overlap across all electrical couplings.
    for partition in range(partitions):
        core = set(np.flatnonzero(owner == partition))
        overlap = set(core)
        for a, b in zip(i, j):
            if a in core:
                overlap.add(int(b))
            if b in core:
                overlap.add(int(a))
        indices = np.asarray(sorted(overlap), dtype=np.int32)
        mapping = {int(node): local for local, node in enumerate(indices)}
        diagonal_rows, diagonal_edges, local_i, local_j, sources = [], [], [], [], []
        for edge, (a, b) in enumerate(zip(i, j)):
            for node in (a, b):
                if int(node) in mapping:
                    diagonal_rows.append(mapping[int(node)])
                    diagonal_edges.append(edge)
            if int(a) in mapping and int(b) in mapping:
                local_i.append(mapping[int(a)])
                local_j.append(mapping[int(b)])
                sources.append(edge)
        arrays = [
            np.asarray(v, dtype=np.int32)
            for v in (diagonal_rows, diagonal_edges, local_i, local_j, sources)
        ]
        blocks.append(SchwarzBlock(indices, *arrays))
    cross = np.flatnonzero(owner[i] != owner[j]).astype(np.int32)
    return SchwarzLayout(
        tuple(blocks),
        owner,
        np.bincount(owner, minlength=partitions),
        owner[i[cross]],
        owner[j[cross]],
        cross,
        dense_limit,
        sweeps,
    )


def make_preconditioner(sim, diagonal):
    layout = sim.schwarz_layout
    if layout is None:
        raise ValueError("Schwarz layout is required")
    _, _, conductance = coupling_edges(sim)
    conductance = conductance.astype(diagonal.dtype)
    diagonal = diagonal.ravel()
    local_solvers = []
    for block in layout.blocks:
        diag = (
            diagonal[block.indices]
            .at[block.diagonal_rows]
            .add(conductance[block.diagonal_edges])
        )
        g = conductance[block.edge_indices]

        def product(x, diag=diag, block=block, g=g):
            return (
                (diag * x)
                .at[block.i]
                .add(-g * x[block.j])
                .at[block.j]
                .add(-g * x[block.i])
            )

        if len(block.indices) <= layout.dense_limit:
            matrix = (
                jnp.diag(diag).at[block.i, block.j].add(-g).at[block.j, block.i].add(-g)
            )
            factor = jnp.linalg.cholesky(matrix)
            solve = lambda b, factor=factor: cho_solve((factor, True), b)
        else:

            def solve(b, product=product, diag=diag):
                def sweep(_, x):
                    return x + 0.5 * (b - product(x)) / diag

                return jax.lax.fori_loop(0, layout.sweeps, sweep, jnp.zeros_like(b))

        local_solvers.append((block.indices, solve))
    count = len(layout.counts)
    normalization = jnp.sqrt(jnp.asarray(layout.counts, dtype=diagonal.dtype))
    coarse_diagonal = (
        jax.ops.segment_sum(diagonal, jnp.asarray(layout.owner), count)
        / normalization**2
    )
    ci, cj, cg = (
        jnp.asarray(layout.coarse_i),
        jnp.asarray(layout.coarse_j),
        conductance[layout.coarse_edges],
    )
    coarse_diag_full = (
        coarse_diagonal.at[ci]
        .add(cg / normalization[ci] ** 2)
        .at[cj]
        .add(cg / normalization[cj] ** 2)
    )
    cross = cg / (normalization[ci] * normalization[cj])

    def coarse_product(x):
        return (
            (coarse_diag_full * x).at[ci].add(-cross * x[cj]).at[cj].add(-cross * x[ci])
        )

    if count <= layout.dense_limit:
        matrix = (
            jnp.diag(coarse_diag_full).at[ci, cj].add(-cross).at[cj, ci].add(-cross)
        )
        factor = jnp.linalg.cholesky(matrix)
        coarse_solve = lambda b: cho_solve((factor, True), b)
    else:

        def coarse_solve(b):
            return jax.lax.fori_loop(
                0,
                layout.sweeps,
                lambda _, x: x + 0.5 * (b - coarse_product(x)) / coarse_diag_full,
                jnp.zeros_like(b),
            )

    def precondition(residual):
        result = jnp.zeros_like(residual)
        for indices, solve in local_solvers:
            result = result.at[indices].add(solve(residual[indices]))
        coarse_rhs = jax.ops.segment_sum(
            residual / normalization[layout.owner], jnp.asarray(layout.owner), count
        )
        coarse = coarse_solve(coarse_rhs)
        return result + coarse[layout.owner] / normalization[layout.owner]

    return precondition


def with_schwarz(sim, partitions=4, dense_limit=512, sweeps=12):
    layout = build_layout(sim, partitions, dense_limit, sweeps)
    return replace(
        sim,
        schwarz_layout=layout,
        resolution=replace(sim.resolution, solver="pcg"),
        metadata={
            **sim.metadata,
            "preconditioner": "two_level_additive_schwarz",
            "schwarz_partitions": len(layout.blocks),
        },
    )


def escalate_solver(sim, solve_iterations, solve_fraction, partitions=4):
    if np.median(solve_iterations) > 30 or solve_fraction > 0.5:
        return with_schwarz(sim, partitions)
    return sim
