"""Measured costs and timestep/compartment convergence gates."""

import time
from dataclasses import asdict, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .compiler import ResolutionPolicy, compile
from .fixtures import synthetic_system
from .simulation import initial_state, step
from .solver import voltage_solve


def timed(fn, repeats=10):
    start = time.perf_counter()
    jax.block_until_ready(fn())
    first_call = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn())
    return {
        "first_call_s": first_call,
        "steady_call_s": (time.perf_counter() - start) / repeats,
    }


def scaling_benchmark(sizes=(8, 32, 128), repeats=10):
    rows = []
    if repeats < 1 or not sizes or any(n < 2 for n in sizes):
        raise ValueError("benchmark needs positive repeats and graph sizes >= 2")
    for n in sizes:
        graph, rules, kinetics = synthetic_system(n)
        sim = compile(
            graph,
            rules,
            kinetics,
            resolution=replace(ResolutionPolicy.default(), solver="pcg"),
        )
        state = initial_state(sim)
        current = jnp.zeros(n)
        compiled_step = jax.jit(lambda old, drive, sim=sim: step(sim, old, drive))
        run_step = partial(compiled_step, state, current)
        compiled_backward = jax.jit(
            jax.grad(
                lambda drive, sim=sim, state=state: step(sim, state, drive)[
                    0
                ].voltage.sum()
            )
        )
        backward_step = partial(compiled_backward, current)
        post_flat = (
            sim.syn_post_idx * sim.resolution.n_comp + sim.syn_params["compartment"]
        )
        compiled_accumulation = jax.jit(
            lambda density, gate, post_flat=post_flat, n=n, sim=sim: (
                jax.ops.segment_sum(
                    density * gate[:, None],
                    post_flat,
                    num_segments=n * sim.resolution.n_comp,
                )
            )
        )
        synaptic_accumulation = partial(
            compiled_accumulation, sim.syn_params["density"], sim.syn_params["gate"]
        )
        diagonal = (
            sim.neuron_params["capacitance"] / (sim.resolution.dt_s * 1000)
            + sim.neuron_params["leak"]
        )
        rhs = diagonal * state.voltage + jnp.linspace(
            -1, 1, n * sim.resolution.n_comp
        ).reshape(state.voltage.shape)
        compiled_solve = jax.jit(
            lambda diag, b, old, sim=sim: voltage_solve(sim, diag, b, old)
        )
        solve = partial(compiled_solve, diagonal, rhs, state.voltage)
        solve_timing = timed(solve, repeats)
        _, residual, iterations = solve()
        # Count static parameter arrays only, excluding state and JAX allocator usage.
        arrays = [
            v
            for group in (sim.neuron_params, sim.syn_params, sim.gap)
            for v in group.values()
            if hasattr(v, "nbytes")
        ]
        rows.append(
            {
                "neurons": n,
                "synapses": len(sim.syn_pre_idx),
                "gap_contacts": len(sim.gap["i"]),
                "parameter_count": rules.parameter_count,
                "step": timed(run_step, repeats),
                "backward_step": timed(backward_step, repeats),
                "synaptic_accumulation": timed(synaptic_accumulation, repeats),
                "voltage_solve": solve_timing,
                "solve_iterations": int(iterations),
                "solve_residual": float(residual),
                "static_parameter_array_bytes": sum(a.nbytes for a in arrays),
            }
        )
    return {
        "data_kind": "synthetic",
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "solver": "matrix_free_pcg",
        "backward_gradient_target": "external_current_pA",
        "memory_scope": "neuron/synapse/gap parameter arrays; excludes state, rules and allocator peak",
        "repeats": repeats,
        "rows": rows,
        "largest_available_connectome_verified": False,
    }


def surrogate_execution_benchmark(sizes=(128, 256, 512), repeats=5, steps=20):
    """Steady-state per-step cost: hybrid reference vs fast surrogate execution.

    Every neuron carries a synthetic relaxation surrogate inside its envelope, so the
    fast path never falls back. Timing is of a jitted scan, divided by `steps`.
    """
    from .fixtures import synthetic_surrogates
    from .surrogates import plan_fast_execution

    if repeats < 1 or steps < 1 or not sizes or any(n < 2 for n in sizes):
        raise ValueError("benchmark needs positive repeats, steps and sizes >= 2")
    rows = []
    for n in sizes:
        graph, rules, kinetics = synthetic_system(n)
        base = synthetic_surrogates(compile(graph, rules, kinetics))
        fast_sim = plan_fast_execution(base)
        # Nonzero drive so the iterative solve does real work every step.
        currents = jnp.tile(jnp.linspace(-5.0, 5.0, n), (steps, 1))
        results, timings = {}, {}
        for name, sim in (("hybrid", base), ("fast", fast_sim)):
            run = jax.jit(
                lambda c, sim=sim: jax.lax.scan(
                    lambda s, x: step(sim, s, x), initial_state(sim), c
                )
            )
            timings[name] = timed(partial(run, currents), repeats)
            timings[name]["per_step_s"] = timings[name]["steady_call_s"] / steps
            results[name] = run(currents)[1][0]
        rows.append(
            {
                "neurons": n,
                "surrogate_neurons": len(fast_sim.fast_plan.surrogate_idx),
                "hybrid": timings["hybrid"],
                "fast": timings["fast"],
                "speedup": timings["hybrid"]["per_step_s"]
                / timings["fast"]["per_step_s"],
                "soma_max_abs_difference_mV": float(
                    jnp.max(
                        jnp.abs(results["hybrid"][..., 0] - results["fast"][..., 0])
                    )
                ),
            }
        )
    return {
        "data_kind": "synthetic",
        "backend": jax.default_backend(),
        "steps_per_call": steps,
        "repeats": repeats,
        "rows": rows,
    }


def resolution_convergence(
    policy, metrics_fn, noise_ceilings, threshold=0.02, max_attempts=6
):
    if not noise_ceilings or any(v <= 0 for v in noise_ceilings.values()):
        raise ValueError("convergence requires positive per-metric noise ceilings")
    history = []
    for _ in range(max_attempts):
        baseline = metrics_fn(policy)
        finer = metrics_fn(replace(policy, dt_s=policy.dt_s / 2))
        extra = metrics_fn(replace(policy, n_comp=policy.n_comp + 2))
        if (
            set(baseline) != set(noise_ceilings)
            or set(finer) != set(baseline)
            or set(extra) != set(baseline)
        ):
            raise ValueError(
                "all registered metrics must participate in resolution convergence"
            )
        dt_changes = {
            m: abs(finer[m] - baseline[m]) / noise_ceilings[m] for m in baseline
        }
        comp_changes = {
            m: abs(extra[m] - baseline[m]) / noise_ceilings[m] for m in baseline
        }
        if not all(
            np.isfinite(v) for v in [*dt_changes.values(), *comp_changes.values()]
        ):
            raise ValueError("nonfinite reference convergence metrics")
        dt_ok, comp_ok = (
            max(dt_changes.values()) < threshold,
            max(comp_changes.values()) < threshold,
        )
        history.append(
            {
                "resolution": asdict(policy),
                "timestep_changes": dt_changes,
                "compartment_changes": comp_changes,
            }
        )
        if dt_ok and comp_ok:
            return policy, {
                "passed": True,
                "frozen": True,
                "threshold": threshold,
                "history": history,
            }
        policy = replace(
            policy,
            dt_s=policy.dt_s if dt_ok else policy.dt_s / 2,
            n_comp=policy.n_comp if comp_ok else policy.n_comp + 1,
        )
    raise RuntimeError(
        "resolution did not converge within the configured reference budget"
    )
