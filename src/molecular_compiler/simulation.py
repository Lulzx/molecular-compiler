"""M8 differentiable scan and experimental time-parallel fixed-point mode."""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .kinetics import binding_step, rush_larsen
from .solver import voltage_solve


@dataclass(frozen=True)
class Stimulus:
    pulses: tuple = ()  # (target neuron_id, t_on_s, t_off_s, current_pA)
    currents: object | None = None  # [L,N], overrides pulses
    opsin_expression: dict = field(default_factory=dict)
    drive: dict = field(default_factory=dict)

    def build(self, neuron_ids, steps, dt_s):
        n = len(neuron_ids)
        if self.currents is not None:
            if np.shape(self.currents) != (steps, n):
                raise ValueError("stimulus currents require [steps, neurons]")
            return jnp.asarray(self.currents)
        indices = {int(neuron): i for i, neuron in enumerate(neuron_ids)}
        times = jnp.arange(steps) * dt_s
        result = jnp.zeros((steps, n))
        for target, onset, offset, amplitude in self.pulses:
            if (
                target not in indices
                or onset < 0
                or offset <= onset
                or not np.isfinite(amplitude)
            ):
                raise ValueError("invalid stimulation target or pulse")
            gain = self.opsin_expression.get(target, self.drive.get(target, 1.0))
            result = result.at[:, indices[target]].add(
                amplitude * gain * ((times >= onset) & (times < offset))
            )
        return result


class SimState(NamedTuple):
    voltage: jax.Array
    gates: jax.Array
    receptors: jax.Array
    resources: jax.Array
    facilitation: jax.Array
    peptide_open: jax.Array
    peptide_field: jax.Array
    signaling: jax.Array
    calcium: jax.Array
    surrogate_state: jax.Array
    spike_above: jax.Array
    event_time: jax.Array
    last_release_time: jax.Array
    synaptic_conductance: jax.Array
    synaptic_reversal_current: jax.Array


@dataclass
class Trajectory:
    t: jax.Array
    voltage: jax.Array
    calcium: jax.Array
    signaling: jax.Array
    solve_residual: jax.Array
    solve_iterations: jax.Array
    surrogate_fallback: jax.Array
    neuron_ids: object
    metadata: dict
    final_state: SimState


def initial_state(sim):
    n, c, e, p = (
        sim.n_neurons,
        sim.resolution.n_comp,
        len(sim.syn_pre_idx),
        sim.neuromod["release"].shape[1],
    )
    dtype = sim.neuron_params["capacitance"].dtype
    v = jnp.full((n, c), sim.resolution.leak_reversal_mV, dtype=dtype)
    width = max((r.state_size for r in sim.channel_records), default=1)
    gates = (
        jnp.stack([r.initial_gates(v, width) for r in sim.channel_records], axis=2)
        if sim.channel_records
        else jnp.zeros((n, c, 0, width), dtype=dtype)
    )
    field_shape = (
        (p,)
        if sim.neuromod["mode"] == "global"
        else (p, math.prod(sim.resolution.grid_shape))
    )
    k = len(sim.neuromod["signaling0"])
    return SimState(
        v,
        gates,
        jnp.zeros((e, len(sim.receptor_records)), dtype=dtype),
        jnp.ones(e, dtype=dtype),
        sim.syn_params["U"],
        jnp.zeros((n, p), dtype=dtype),
        jnp.zeros(field_shape, dtype=dtype),
        jnp.broadcast_to(sim.neuromod["signaling0"], (n, k)),
        jnp.zeros(n, dtype=dtype),
        jnp.column_stack([v[:, 0], jnp.zeros(n), jnp.zeros(n), jnp.zeros(n)]),
        jnp.zeros(n, dtype=bool),
        jnp.array(0.0, dtype=dtype),
        jnp.zeros(e, dtype=dtype),
        jnp.zeros((n, c, len(sim.receptor_records)), dtype=dtype),
        jnp.zeros((n, c, len(sim.receptor_records)), dtype=dtype),
    )


def grid_release(sim, released):
    """Periodic screened-diffusion kernel. No N x N pair matrix."""
    policy = sim.resolution
    p = released.shape[1]
    grid_size = math.prod(policy.grid_shape)
    grid = (
        jnp.zeros((grid_size, p), dtype=released.dtype)
        .at[sim.neuromod["grid_index"]]
        .add(released)
    )
    frequency = jnp.meshgrid(
        *[jnp.fft.fftfreq(s, d=policy.grid_spacing_um) for s in policy.grid_shape],
        indexing="ij",
    )
    frequency2 = sum(f * f for f in frequency)
    kernel = 1 / (1 + (2 * jnp.pi * sim.neuromod["decay_length_um"]) ** 2 * frequency2)
    values = grid.T.reshape((p,) + policy.grid_shape)
    axes = tuple(range(1, 1 + len(policy.grid_shape)))
    return jnp.fft.ifftn(
        jnp.fft.fftn(values, axes=axes) * kernel, axes=axes
    ).real.reshape(p, grid_size)


def event_synapses(sim, old, spikes, now):
    """Lazy STP/receptor updates only on edges reached by spikes.

    Aggregated compartment conductances decay analytically per receptor;
    synapse states are touched only during outgoing event traversal.
    """
    dt = sim.resolution.dt_s
    tau = jnp.asarray(
        [r.time_constant(sim.resolution.temperature_C) for r in sim.receptor_records]
    )
    decay = jnp.exp(-dt / tau)
    initial = (
        old.receptors,
        old.resources,
        old.facilitation,
        old.last_release_time,
        old.synaptic_conductance * decay,
        old.synaptic_reversal_current * decay,
    )
    if not len(sim.syn_pre_idx):
        return initial
    indices = jnp.nonzero(spikes, size=sim.n_neurons, fill_value=-1)[0]

    def neuron_event(index, carry):
        neuron = indices[index]

        def traverse(values):
            start, stop = sim.syn_pre_ptr[neuron], sim.syn_pre_ptr[neuron + 1]

            def edge_event(position, states):
                (
                    receptors,
                    resources,
                    facilitation,
                    timestamps,
                    aggregate_g,
                    aggregate_b,
                ) = states
                edge = sim.syn_pre_edges[position]
                elapsed = now - timestamps[edge]
                base_u = sim.syn_params["U"][edge]
                u = rush_larsen(
                    facilitation[edge], base_u, sim.syn_params["tau_fac"][edge], elapsed
                )
                u = u + base_u * (1 - u)
                x = rush_larsen(
                    resources[edge], 1.0, sim.syn_params["tau_rec"][edge], elapsed
                )
                released = u * x
                receptor_before = receptors[edge] * jnp.exp(-elapsed / tau)
                receptor_delta = released * (1 - receptor_before)
                receptor_after = receptor_before + receptor_delta
                g_delta = (
                    receptor_delta
                    * sim.syn_params["density"][edge]
                    * sim.syn_params["gbar"]
                    * sim.syn_params["gate"][edge]
                )
                post, comp = sim.syn_post_idx[edge], sim.syn_params["compartment"][edge]
                return (
                    receptors.at[edge].set(receptor_after),
                    resources.at[edge].set(x - released),
                    facilitation.at[edge].set(u),
                    timestamps.at[edge].set(now),
                    aggregate_g.at[post, comp].add(g_delta),
                    aggregate_b.at[post, comp].add(
                        g_delta * sim.syn_params["reversal"][edge]
                    ),
                )

            return jax.lax.fori_loop(start, stop, edge_event, values)

        return jax.lax.cond(neuron >= 0, traverse, lambda values: values, carry)

    return jax.lax.fori_loop(0, sim.n_neurons, neuron_event, initial)


def synaptic_update(sim, state, params, source_events, source_activity):
    """Dense STP/receptor update for any subset of synapses (also used per device)."""
    dt = sim.resolution.dt_s
    old_facilitation, old_resources, old_receptors = state
    tau_rec, tau_fac, base_u = (params[k] for k in ("tau_rec", "tau_fac", "U"))
    if sim.resolution.release_mode == "spiking":
        u_before = rush_larsen(old_facilitation, base_u, tau_fac, dt)
        x_before = rush_larsen(old_resources, 1.0, tau_rec, dt)
        facilitation = jnp.where(
            source_events, u_before + base_u * (1 - u_before), u_before
        )
        release = jnp.where(source_events, facilitation * x_before, 0.0)
        resources = x_before - release
        receptor_tau = jnp.asarray(
            [
                r.time_constant(sim.resolution.temperature_C)
                for r in sim.receptor_records
            ]
        )
        decayed = old_receptors * jnp.exp(-dt / receptor_tau)
        receptors = decayed + release[:, None] * (1 - decayed)
    else:
        rate = 20 * source_activity
        target_u = (base_u / tau_fac + base_u * rate) / (1 / tau_fac + base_u * rate)
        facilitation = rush_larsen(
            old_facilitation, target_u, 1 / (1 / tau_fac + base_u * rate), dt
        )
        resource_tau = 1 / (1 / tau_rec + facilitation * rate)
        resources = rush_larsen(old_resources, resource_tau / tau_rec, resource_tau, dt)
        release = facilitation * resources * source_activity
        receptors = (
            jnp.stack(
                [
                    binding_step(
                        old_receptors[:, i],
                        release,
                        0.2,
                        r.time_constant(sim.resolution.temperature_C),
                        dt,
                    )
                    for i, r in enumerate(sim.receptor_records)
                ],
                axis=1,
            )
            if sim.receptor_records
            else old_receptors
        )
    return facilitation, resources, receptors


def step(sim, old, current, event=False, synapses=None):
    """`synapses`: opt-in callable replacing synaptic update + accumulation
    (see distributed.distributed_synapses); None keeps the single-device path."""
    dt, n, c = sim.resolution.dt_s, sim.n_neurons, sim.resolution.n_comp
    v = old.voltage
    # Graded release is a voltage-dependent rate, not a fixed synaptic sign.
    activity = jax.nn.sigmoid((v[:, 0] + 35) / 5)
    spike_above = v[:, 0] >= sim.resolution.spike_threshold_mV
    spikes = spike_above & ~old.spike_above
    event_time = old.event_time + dt
    last_release_time = old.last_release_time
    cached_g, cached_b = old.synaptic_conductance, old.synaptic_reversal_current
    if synapses is not None:
        if event:
            raise ValueError("distributed synapses do not support event mode")
        (
            receptors,
            resources,
            facilitation,
            syn_by_receptor,
            syn_reversal_by_receptor,
        ) = synapses(sim, old, spikes, activity)
    elif event and sim.resolution.release_mode == "spiking":
        receptors, resources, facilitation, last_release_time, cached_g, cached_b = (
            event_synapses(sim, old, spikes, event_time)
        )
    else:
        facilitation, resources, receptors = synaptic_update(
            sim,
            (old.facilitation, old.resources, old.receptors),
            sim.syn_params,
            spikes[sim.syn_pre_idx],
            activity[sim.syn_pre_idx],
        )
    gates = (
        jnp.stack(
            [
                r.step_gates(v, old.gates[:, :, i], dt, sim.resolution.temperature_C)
                for i, r in enumerate(sim.channel_records)
            ],
            axis=2,
        )
        if sim.channel_records
        else old.gates
    )
    released_peptides = sim.neuromod["release"] * activity[:, None]
    if sim.neuromod["mode"] == "global":
        field_target = released_peptides.sum(axis=0)
        peptide_field = rush_larsen(
            old.peptide_field, field_target, sim.neuromod["tau_s"], dt
        )
        ligand = jnp.broadcast_to(peptide_field, sim.neuromod["sensitivity"].shape)
    else:
        field_target = grid_release(sim, released_peptides)
        peptide_field = rush_larsen(
            old.peptide_field, field_target, sim.neuromod["tau_s"][:, None], dt
        )
        ligand = peptide_field[:, sim.neuromod["grid_index"]].T
    peptide_open = binding_step(
        old.peptide_open,
        jnp.maximum(ligand, 0),
        sim.neuromod["ec50_nM"],
        sim.neuromod["tau_s"],
        dt,
    )
    peptide_drive = (peptide_open * sim.neuromod["sensitivity"]).sum(axis=1)
    concentration = sim.neuromod["concentrations"]
    slow_target = sim.neuromod["mod_sensitivity"] * (
        concentration.sum() + released_peptides.sum() / max(n, 1)
    )
    signaling = rush_larsen(
        old.signaling,
        jnp.broadcast_to(slow_target[:, None], old.signaling.shape),
        60.0,
        dt,
    )
    modulation = (
        jnp.exp(jnp.clip(0.1 * signaling.mean(axis=1), -3, 3))
        if signaling.shape[1]
        else jnp.ones(n)
    )
    open_channels = (
        jnp.stack(
            [
                r.open_probability(gates[:, :, i])
                for i, r in enumerate(sim.channel_records)
            ],
            axis=2,
        )
        if sim.channel_records
        else jnp.zeros((n, c, 0))
    )
    channel_g = (
        sim.neuron_params["channel_density"][:, None, :]
        * sim.neuron_params["channel_gbar"]
        * open_channels
        * modulation[:, None, None]
    )
    if synapses is not None:
        pass
    elif event and sim.resolution.release_mode == "spiking":
        syn_g = None
        syn_by_receptor, syn_reversal_by_receptor = cached_g, cached_b
    else:
        syn_g = (
            sim.syn_params["density"]
            * sim.syn_params["gbar"]
            * receptors
            * sim.syn_params["gate"][:, None]
        )
        post_flat = sim.syn_post_idx * c + sim.syn_params["compartment"]
        syn_by_receptor = jax.ops.segment_sum(
            syn_g, post_flat, num_segments=n * c
        ).reshape(n, c, len(sim.receptor_records))
        syn_reversal_by_receptor = jax.ops.segment_sum(
            syn_g * sim.syn_params["reversal"], post_flat, num_segments=n * c
        ).reshape(n, c, len(sim.receptor_records))
    syn_diagonal = syn_by_receptor.sum(axis=2)
    syn_rhs = syn_reversal_by_receptor.sum(axis=2)
    capacitance_dt = sim.neuron_params["capacitance"] / (dt * 1000)
    diagonal = (
        capacitance_dt
        + sim.neuron_params["leak"]
        + channel_g.sum(axis=2)
        + syn_diagonal
    )
    drive = (
        current
        + peptide_drive * sim.neuromod["effector"]
        + sim.neuron_params["transporter_current"]
    )
    rhs = (
        capacitance_dt * v
        + sim.neuron_params["leak"] * sim.resolution.leak_reversal_mV
        + (channel_g * sim.neuron_params["channel_reversal"][:, None]).sum(axis=2)
        + syn_rhs
    )
    rhs = rhs.at[:, 0].add(drive)
    voltage, residual, iterations = voltage_solve(sim, diagonal, rhs, v)
    ca_mask = jnp.asarray(
        [r.ion_selectivity.get("Ca", 0) > 0 for r in sim.channel_records]
    )
    calcium_current = jnp.maximum(
        (
            channel_g
            * (sim.neuron_params["channel_reversal"][:, None] - voltage[:, :, None])
            * ca_mask
        ).sum(axis=(1, 2)),
        0,
    )
    rec_ca = jnp.asarray(
        [r.ion_selectivity.get("Ca", 0) > 0 for r in sim.receptor_records]
    )
    syn_ca_current = (
        (syn_reversal_by_receptor - syn_by_receptor * voltage[:, :, None]) * rec_ca
    ).sum(axis=(1, 2))
    calcium_current += jnp.maximum(syn_ca_current, 0)
    calcium = rush_larsen(old.calcium, 0.001 * calcium_current * 0.5, 0.5, dt)
    inputs = jnp.concatenate(
        [
            drive[:, None],
            jnp.broadcast_to(concentration, (n, len(concentration))),
            signaling,
            peptide_drive[:, None],
        ],
        axis=1,
    )
    surrogate_state = jnp.column_stack(
        [
            voltage[:, 0],
            calcium,
            gates.mean(axis=(1, 2, 3)) if gates.shape[2] else jnp.zeros(n),
            signaling.mean(axis=1) if signaling.shape[1] else jnp.zeros(n),
        ]
    )
    fallback = jnp.zeros(n, dtype=bool)
    for type_id, surrogate in enumerate(sim.surrogates):
        if surrogate.family == "full":
            continue
        selected = sim.neuron_params["type_index"] == type_id
        valid = surrogate.inside(inputs) & (
            surrogate.conditioned | (not sim.neuromod["nondefault"])
        )
        fallback |= selected & ~valid
        dimensions = surrogate.state_dimension
        predicted = old.surrogate_state.at[:, :dimensions].set(
            old.surrogate_state[:, :dimensions]
            + dt * surrogate.derivative(old.surrogate_state[:, :dimensions], inputs)
        )
        active = selected & valid
        surrogate_state = jnp.where(active[:, None], predicted, surrogate_state)
        voltage = voltage.at[:, 0].set(
            jnp.where(active, predicted[:, 0], voltage[:, 0])
        )
        calcium = jnp.where(active, jnp.maximum(predicted[:, 1], 0), calcium)
    new = SimState(
        voltage,
        gates,
        receptors,
        resources,
        facilitation,
        peptide_open,
        peptide_field,
        signaling,
        calcium,
        surrogate_state,
        spike_above,
        event_time,
        last_release_time,
        cached_g,
        cached_b,
    )
    return new, (voltage, calcium, signaling, residual, iterations, fallback)


def _sequential(sim, state0, currents, event=False):
    return jax.lax.scan(
        jax.checkpoint(lambda state, current: step(sim, state, current, event=event)),
        state0,
        currents,
    )


def _parallel(sim, state0, currents):
    """Time-parallel Jacobi waveform iteration with exact sequential fallback.

    This is an experimental fixed-point accelerator, not DEER. All timesteps
    evaluate concurrently; no dense state Jacobian is built.
    """
    steps = len(currents)
    guess = jax.tree.map(lambda a: jnp.broadcast_to(a, (steps,) + a.shape), state0)

    def iteration(_, previous):
        shifted = jax.tree.map(
            lambda a, x: jnp.concatenate([a[None], x[:-1]], axis=0), state0, previous
        )
        new, _ = jax.vmap(lambda state, current: step(sim, state, current))(
            shifted, currents
        )
        return new

    result = jax.lax.fori_loop(0, sim.resolution.parallel_iterations, iteration, guess)
    shifted = jax.tree.map(
        lambda a, x: jnp.concatenate([a[None], x[:-1]], axis=0), state0, result
    )
    checked, diagnostic = jax.vmap(lambda state, current: step(sim, state, current))(
        shifted, currents
    )
    errors = [
        jnp.max(jnp.not_equal(a, b).astype(jnp.float32))
        if a.dtype == jnp.bool_
        else jnp.max(jnp.abs(a - b))
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(checked))
        if a.size
    ]
    error = jnp.max(jnp.stack(errors))
    converged = error <= 1e-6

    def accept(_):
        return jax.tree.map(lambda a: a[-1], checked), diagnostic

    final, output = jax.lax.cond(
        converged, accept, lambda _: _sequential(sim, state0, currents), operand=None
    )
    return final, output, ~converged


def simulate(
    sim, stimulus, duration_s, mode="sequential", body=None, seed=0, state0=None
):
    if mode not in {"sequential", "event", "parallel"}:
        raise ValueError("unknown execution mode")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration must be positive and finite")
    steps = round(duration_s / sim.resolution.dt_s)
    if steps < 1 or not np.isclose(
        steps * sim.resolution.dt_s, duration_s, atol=1e-10, rtol=1e-8
    ):
        raise ValueError("duration must be an integer multiple of timestep")
    if body is not None and mode != "sequential":
        raise ValueError("body coupling requires sequential evaluation")
    neuron_ids = sim.neuron_params["neuron_ids"]
    currents = stimulus.build(neuron_ids, steps, sim.resolution.dt_s)
    state0 = initial_state(sim) if state0 is None else state0
    parallel_fallback = False
    if body is not None:
        sensory = body.reset({"seed": seed})
        state, output = state0, []
        for current in currents:
            sensory_current = jnp.asarray(sensory)
            if sensory_current.shape != (sim.n_neurons,):
                raise ValueError(
                    "body sensory mapping must return one current per neuron"
                )
            state, result = step(
                sim, state, current + jax.lax.stop_gradient(sensory_current)
            )
            output.append(result)
            sensory = body.step(
                np.asarray(jax.lax.stop_gradient(state.voltage[:, 0])),
                sim.resolution.dt_s,
            )
        final = state
        values = jax.tree.map(lambda *xs: jnp.stack(xs), *output)
    elif mode == "parallel":
        final, values, parallel_fallback = _parallel(sim, state0, currents)
    else:
        final, values = _sequential(sim, state0, currents, event=mode == "event")
    if mode == "event" and sim.resolution.release_mode == "spiking":
        elapsed = final.event_time - final.last_release_time
        tau = jnp.asarray(
            [
                r.time_constant(sim.resolution.temperature_C)
                for r in sim.receptor_records
            ]
        )
        final = final._replace(
            receptors=final.receptors * jnp.exp(-elapsed[:, None] / tau),
            resources=rush_larsen(
                final.resources, 1.0, sim.syn_params["tau_rec"], elapsed
            ),
            facilitation=rush_larsen(
                final.facilitation,
                sim.syn_params["U"],
                sim.syn_params["tau_fac"],
                elapsed,
            ),
            last_release_time=jnp.full_like(final.last_release_time, final.event_time),
        )
    voltage, calcium, signaling, residual, iterations, fallback = values
    metadata = {
        **sim.metadata,
        "boundary_condition": "closed_loop" if body else "open_loop",
        "mode": mode,
        "seed": seed,
        "dt_s": sim.resolution.dt_s,
        "parallel_fallback": parallel_fallback,
        "event_accelerated": mode == "event"
        and sim.resolution.release_mode == "spiking",
        "precision": str(voltage.dtype),
        "grid_boundary": "periodic" if sim.neuromod["mode"] == "grid" else None,
    }
    if not isinstance(residual, jax.core.Tracer):
        if not np.all(np.isfinite(np.asarray(voltage))) or np.any(
            np.asarray(residual) > sim.resolution.solve_tolerance
        ):
            raise FloatingPointError(
                "voltage solve failed residual tolerance or trajectory became nonfinite"
            )
        metadata.update(
            max_solve_residual=float(jnp.max(residual)),
            median_solve_iterations=float(jnp.median(iterations)),
            surrogate_fallback_count=int(jnp.sum(fallback)),
            parallel_fallback=bool(parallel_fallback),
        )
    return Trajectory(
        jnp.arange(1, steps + 1) * sim.resolution.dt_s,
        voltage,
        calcium,
        signaling,
        residual,
        iterations,
        fallback,
        neuron_ids,
        metadata,
        final,
    )
