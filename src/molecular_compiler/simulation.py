"""M8 differentiable scan and experimental time-parallel fixed-point mode."""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .body_ladder import BodyOutput
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


def _take(array, rows):
    return array if rows is None else array[rows]


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


def _synapse_state(sim, old, activity, spikes, event_time, event):
    """Presynaptic STP and receptor updates for every edge."""
    if event and sim.resolution.release_mode == "spiking":
        return event_synapses(sim, old, spikes, event_time)
    facilitation, resources, receptors = synaptic_update(
        sim,
        (old.facilitation, old.resources, old.receptors),
        sim.syn_params,
        spikes[sim.syn_pre_idx],
        activity[sim.syn_pre_idx],
    )
    return (
        receptors,
        resources,
        facilitation,
        old.last_release_time,
        old.synaptic_conductance,
        old.synaptic_reversal_current,
    )


def _modulatory(sim, old, activity, current, concentration=None):
    """Peptide, slow signaling, drive and surrogate inputs; no channel kinetics.

    `concentration` (opt-in, L4) overrides the compiled global modulator c."""
    dt, n = sim.resolution.dt_s, sim.n_neurons
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
    if concentration is None:
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
    drive = (
        current
        + peptide_drive * sim.neuromod["effector"]
        + sim.neuron_params["transporter_current"]
    )
    inputs = jnp.concatenate(
        [
            drive[:, None],
            jnp.broadcast_to(concentration, (n, len(concentration))),
            signaling,
            peptide_drive[:, None],
        ],
        axis=1,
    )
    return peptide_open, peptide_field, signaling, modulation, drive, inputs


def _channel_conductance(sim, gates, modulation, rows=None):
    open_channels = (
        jnp.stack(
            [
                r.open_probability(gates[:, :, i])
                for i, r in enumerate(sim.channel_records)
            ],
            axis=2,
        )
        if sim.channel_records
        else jnp.zeros(gates.shape[:2] + (0,))
    )
    params = sim.neuron_params
    return (
        _take(params["channel_density"], rows)[:, None, :]
        * params["channel_gbar"]
        * open_channels
        * _take(modulation, rows)[:, None, None]
    )


def _synaptic_conductance(sim, receptors):
    n, c = sim.n_neurons, sim.resolution.n_comp
    syn_g = (
        sim.syn_params["density"]
        * sim.syn_params["gbar"]
        * receptors
        * sim.syn_params["gate"][:, None]
    )
    post_flat = sim.syn_post_idx * c + sim.syn_params["compartment"]
    by_receptor = jax.ops.segment_sum(syn_g, post_flat, num_segments=n * c).reshape(
        n, c, len(sim.receptor_records)
    )
    reversal_by_receptor = jax.ops.segment_sum(
        syn_g * sim.syn_params["reversal"], post_flat, num_segments=n * c
    ).reshape(n, c, len(sim.receptor_records))
    return by_receptor, reversal_by_receptor


def _assemble(sim, v, channel_g, syn_by, syn_reversal, drive, rows=None):
    capacitance = _take(sim.neuron_params["capacitance"], rows)
    leak = _take(sim.neuron_params["leak"], rows)
    capacitance_dt = capacitance / (sim.resolution.dt_s * 1000)
    diagonal = capacitance_dt + leak + channel_g.sum(axis=2) + syn_by.sum(axis=2)
    reversal = _take(sim.neuron_params["channel_reversal"], rows)
    rhs = (
        capacitance_dt * v
        + leak * sim.resolution.leak_reversal_mV
        + (channel_g * reversal[:, None]).sum(axis=2)
        + syn_reversal.sum(axis=2)
    )
    return diagonal, rhs.at[:, 0].add(drive)


def _calcium_update(sim, old_calcium, channel_g, voltage, syn_by, syn_reversal, rows):
    ca_mask = jnp.asarray(
        [r.ion_selectivity.get("Ca", 0) > 0 for r in sim.channel_records]
    )
    reversal = _take(sim.neuron_params["channel_reversal"], rows)
    calcium_current = jnp.maximum(
        (channel_g * (reversal[:, None] - voltage[:, :, None]) * ca_mask).sum(
            axis=(1, 2)
        ),
        0,
    )
    rec_ca = jnp.asarray(
        [r.ion_selectivity.get("Ca", 0) > 0 for r in sim.receptor_records]
    )
    syn_ca_current = ((syn_reversal - syn_by * voltage[:, :, None]) * rec_ca).sum(
        axis=(1, 2)
    )
    calcium_current += jnp.maximum(syn_ca_current, 0)
    return rush_larsen(
        old_calcium, 0.001 * calcium_current * 0.5, 0.5, sim.resolution.dt_s
    )


def _surrogate_predictions(sim, old, inputs):
    """Surrogate state update, active mask and M7-R2/M7-R4 fallback flags."""
    dt, n = sim.resolution.dt_s, sim.n_neurons
    predicted_state = old.surrogate_state
    active = jnp.zeros(n, dtype=bool)
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
        chosen = selected & valid
        predicted_state = jnp.where(chosen[:, None], predicted, predicted_state)
        active |= chosen
    return predicted_state, active, fallback


def _front(sim, old, event, synapses=None):
    """Shared first half of a step: spike events and synapse updates.

    With `synapses` (distributed.distributed_synapses), the update and the
    accumulation come from the callable; the cached event conductances stay.
    """
    v = old.voltage
    # Graded release is a voltage-dependent rate, not a fixed synaptic sign.
    activity = jax.nn.sigmoid((v[:, 0] + 35) / 5)
    spike_above = v[:, 0] >= sim.resolution.spike_threshold_mV
    spikes = spike_above & ~old.spike_above
    event_time = old.event_time + sim.resolution.dt_s
    if synapses is not None:
        receptors, resources, facilitation, syn_by, syn_reversal = synapses(
            sim, old, spikes, activity
        )
        result = (receptors, resources, facilitation, old.last_release_time)
        result += (old.synaptic_conductance, old.synaptic_reversal_current)
        return activity, spike_above, event_time, result, (syn_by, syn_reversal)
    state = _synapse_state(sim, old, activity, spikes, event_time, event)
    return activity, spike_above, event_time, state, None


def _hybrid_step(sim, old, current, event=False, synapses=None, concentration=None):
    """Reference: full kinetics always run; surrogate output replaces the result."""
    dt, n = sim.resolution.dt_s, sim.n_neurons
    v = old.voltage
    activity, spike_above, event_time, synapses, accumulated = _front(
        sim, old, event, synapses
    )
    receptors, resources, facilitation, last_release_time, cached_g, cached_b = synapses
    peptide_open, peptide_field, signaling, modulation, drive, inputs = _modulatory(
        sim, old, activity, current, concentration
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
    channel_g = _channel_conductance(sim, gates, modulation)
    if accumulated is not None:
        syn_by, syn_reversal = accumulated
    elif event and sim.resolution.release_mode == "spiking":
        syn_by, syn_reversal = cached_g, cached_b
    else:
        syn_by, syn_reversal = _synaptic_conductance(sim, receptors)
    diagonal, rhs = _assemble(sim, v, channel_g, syn_by, syn_reversal, drive)
    voltage, residual, iterations = voltage_solve(sim, diagonal, rhs, v)
    calcium = _calcium_update(
        sim, old.calcium, channel_g, voltage, syn_by, syn_reversal, None
    )
    surrogate_state = jnp.column_stack(
        [
            voltage[:, 0],
            calcium,
            gates.mean(axis=(1, 2, 3)) if gates.shape[2] else jnp.zeros(n),
            signaling.mean(axis=1) if signaling.shape[1] else jnp.zeros(n),
        ]
    )
    predicted_state, active, fallback = _surrogate_predictions(sim, old, inputs)
    surrogate_state = jnp.where(active[:, None], predicted_state, surrogate_state)
    voltage = voltage.at[:, 0].set(
        jnp.where(active, predicted_state[:, 0], voltage[:, 0])
    )
    calcium = jnp.where(active, jnp.maximum(predicted_state[:, 1], 0), calcium)
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


def _surrogate_step(sim, old, current, concentration=None):
    """Fast branch, taken only when no eligible neuron left its envelope.

    Eligible neurons take the surrogate update and skip gate stepping, channel
    conductances, calcium kinetics and the voltage solve. The solve runs on the
    full-kinetics set only; a gap contact to an eligible neuron enters as a
    Dirichlet boundary at the surrogate soma voltage. Eligible non-soma
    compartments are set to that voltage, and their gates are held until the
    neuron next runs full kinetics.
    """
    plan = sim.fast_plan
    dt, n, c = sim.resolution.dt_s, sim.n_neurons, sim.resolution.n_comp
    v, full, surr = old.voltage, plan.full_idx, plan.surrogate_idx
    activity, spike_above, event_time, synapses, _ = _front(sim, old, False)
    receptors, resources, facilitation, last_release_time = synapses[:4]
    peptide_open, peptide_field, signaling, modulation, drive, inputs = _modulatory(
        sim, old, activity, current, concentration
    )
    predicted_state, _, fallback = _surrogate_predictions(sim, old, inputs)
    gates = old.gates
    voltage = jnp.zeros_like(v)
    calcium = jnp.zeros(n, dtype=v.dtype)
    surrogate_state = predicted_state
    residual = jnp.zeros((), dtype=v.dtype)
    iterations = jnp.zeros((), dtype=jnp.int32)
    if len(full):
        if sim.channel_records:
            stepped = jnp.stack(
                [
                    r.step_gates(
                        v[full],
                        old.gates[full][:, :, i],
                        dt,
                        sim.resolution.temperature_C,
                    )
                    for i, r in enumerate(sim.channel_records)
                ],
                axis=2,
            )
            gates = old.gates.at[full].set(stepped)
        channel_g = _channel_conductance(sim, gates[full], modulation, full)
        syn_by, syn_reversal = (a[full] for a in _synaptic_conductance(sim, receptors))
        diagonal, rhs = _assemble(
            sim, v[full], channel_g, syn_by, syn_reversal, drive[full], full
        )
        gap = sim.gap
        both, (contact, local, other) = plan.gap_full_full, plan.gap_full_surrogate
        sub = SimpleNamespace(
            n_neurons=len(full),
            resolution=sim.resolution,
            gap={
                "i": jnp.asarray(plan.local[np.asarray(gap["i"])[both]]),
                "j": jnp.asarray(plan.local[np.asarray(gap["j"])[both]]),
                "conductance": gap["conductance"][both],
            },
            partition_plan=None,
            schwarz_layout=None,
            neuron_params={
                k: sim.neuron_params[k][full]
                for k in ("axial",)
                if k in sim.neuron_params
            },
        )
        boundary_g = gap["conductance"][contact].astype(diagonal.dtype)
        diagonal = diagonal.at[local, 0].add(boundary_g)
        rhs = rhs.at[local, 0].add(boundary_g * predicted_state[other, 0])
        solved, solve_residual, solve_iterations = voltage_solve(
            sub, diagonal, rhs, v[full]
        )
        residual = solve_residual.astype(v.dtype)
        iterations = solve_iterations.astype(jnp.int32)
        calcium_full = _calcium_update(
            sim, old.calcium[full], channel_g, solved, syn_by, syn_reversal, full
        )
        voltage = voltage.at[full].set(solved)
        calcium = calcium.at[full].set(calcium_full)
        state_full = jnp.column_stack(
            [
                solved[:, 0],
                calcium_full,
                gates[full].mean(axis=(1, 2, 3))
                if gates.shape[2]
                else jnp.zeros(len(full)),
                signaling[full].mean(axis=1)
                if signaling.shape[1]
                else jnp.zeros(len(full)),
            ]
        )
        surrogate_state = surrogate_state.at[full].set(state_full)
    soma = predicted_state[surr, 0]
    voltage = voltage.at[surr].set(jnp.broadcast_to(soma[:, None], (len(surr), c)))
    calcium = calcium.at[surr].set(jnp.maximum(predicted_state[surr, 1], 0))
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
        old.synaptic_conductance,
        old.synaptic_reversal_current,
    )
    return new, (voltage, calcium, signaling, residual, iterations, fallback)


def step(sim, old, current, event=False, synapses=None, concentration=None):
    """One timestep. `synapses` is an opt-in callable replacing the synaptic
    update and accumulation (distributed.distributed_synapses). `concentration`
    is an opt-in time-varying modulator vector c(t) (L4 interoception)."""
    plan = sim.fast_plan
    if synapses is not None:
        if event:
            raise ValueError("distributed synapses do not support event mode")
        if plan is not None and len(plan.surrogate_idx):
            raise ValueError("distributed synapses do not support fast surrogates")
        return _hybrid_step(
            sim, old, current, synapses=synapses, concentration=concentration
        )
    if plan is None or not len(plan.surrogate_idx):
        return _hybrid_step(sim, old, current, event=event, concentration=concentration)
    if event:
        raise ValueError("fast surrogate execution does not support event mode")
    # Per-neuron masking cannot skip work under static shapes, so neurons are
    # partitioned at plan time. A runtime envelope exit (M7-R2) sends that whole
    # step down the hybrid branch, where every neuron runs full kinetics and the
    # exit is logged. lax.cond differentiates through the branch taken, and
    # becomes a select (both branches run) under vmap.
    activity = jax.nn.sigmoid((old.voltage[:, 0] + 35) / 5)
    inputs = _modulatory(sim, old, activity, current, concentration)[-1]
    exits = jnp.zeros((), dtype=bool)
    for type_id, surrogate in enumerate(sim.surrogates):
        if surrogate.family == "full":
            continue
        mine = (sim.neuron_params["type_index"] == type_id) & jnp.asarray(plan.eligible)
        exits |= jnp.any(mine & ~surrogate.inside(inputs))

    def hybrid(_):
        return _hybrid_step(sim, old, current, concentration=concentration)

    shapes = jax.eval_shape(hybrid, None)

    def fast(_):
        out = _surrogate_step(sim, old, current, concentration)
        return jax.tree.map(lambda x, s: x.astype(s.dtype), out, shapes)

    return jax.lax.cond(exits, hybrid, fast, None)


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
    sim,
    stimulus,
    duration_s,
    mode="sequential",
    body=None,
    seed=0,
    state0=None,
    modulator_tau_s=1.0,
):
    """Run the simulation. A closed-loop `body` returning BodyOutput (L4) also
    drives the global modulator concentrations: c relaxes exponentially toward
    the body's targets with time constant `modulator_tau_s` (c starts at the
    compiled concentrations). Bodies returning only sensory currents leave c
    at its compiled constant."""
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
        c = None
        c_fixed = sim.neuromod["concentrations"]
        for current in currents:
            if isinstance(sensory, BodyOutput):
                target = jnp.asarray(sensory.modulator_targets, dtype=c_fixed.dtype)
                if target.shape != c_fixed.shape:
                    raise ValueError(
                        "body modulator targets must match compiled concentrations"
                    )
                c = rush_larsen(
                    c_fixed if c is None else c,
                    jax.lax.stop_gradient(target),
                    modulator_tau_s,
                    sim.resolution.dt_s,
                )
                sensory = sensory.sensory
            sensory_current = jnp.asarray(sensory)
            if sensory_current.shape != (sim.n_neurons,):
                raise ValueError(
                    "body sensory mapping must return one current per neuron"
                )
            state, result = step(
                sim,
                state,
                current + jax.lax.stop_gradient(sensory_current),
                concentration=c,
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
        "surrogate_execution": "hybrid" if sim.fast_plan is None else "fast",
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
