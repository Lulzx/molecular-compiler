"""C-R3: graph partitions, boundary-only halo exchange and sharded PCG.

Synapses are owned by the device of their postsynaptic neuron: per-synapse
update and conductance accumulation run per device under shard_map, with
presynaptic voltage-derived signals (activity, spike) arriving through the
same boundary halo exchange as the voltage solve.

Verified on virtual CPU devices in one process only (two devices in the
tests). Multi-host deployment, real interconnect cost and mouse-scale
throughput require separate hardware evidence.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass
class PartitionPlan:
    mesh: Mesh
    n_neurons: int
    local_size: int
    send_indices: object
    send_mask: object
    row_indices: object
    neighbor_indices: object
    neighbor_owners: object
    halo_slots: object
    remote: object
    edge_conductance: object
    halo_neurons: int
    edge_source_index: object
    # Synapse ownership by postsynaptic device, padded to [partitions, slots].
    syn_edge_ids: object = None  # global synapse index, -1 padding
    syn_pre_local: object = None
    syn_pre_owner: object = None
    syn_pre_slot: object = None
    syn_pre_remote: object = None
    syn_post_local: object = None
    syn_counts: object = None  # host int array [partitions]

    @property
    def partitions(self):
        return self.mesh.size


def partition_graph(sim, devices=None):
    devices = tuple(devices or jax.devices())
    if not devices:
        raise ValueError("no devices available")
    d, n = len(devices), sim.n_neurons
    local_size = (n + d - 1) // d
    sends = [[[] for _ in range(d)] for _ in range(d)]
    # Halo catalog includes chemical edges, so it can carry presynaptic state
    # for distributed synaptic work as well as voltage-solver coupling.
    edges = list(zip(np.asarray(sim.gap["i"]), np.asarray(sim.gap["j"])))
    edges += list(zip(np.asarray(sim.syn_pre_idx), np.asarray(sim.syn_post_idx)))
    for i, j in edges:
        a, b = int(i) // local_size, int(j) // local_size
        if a != b:
            sends[a][b].append(int(i) % local_size)
            sends[b][a].append(int(j) % local_size)
    for source in range(d):
        for target in range(d):
            sends[source][target] = sorted(set(sends[source][target]))
    h = max(1, max(len(v) for row in sends for v in row))
    send_indices = np.zeros((d, d, h), dtype=np.int32)
    send_mask = np.zeros((d, d, h), dtype=bool)
    for source in range(d):
        for target in range(d):
            nodes = sends[source][target]
            send_indices[source, target, : len(nodes)] = nodes
            send_mask[source, target, : len(nodes)] = True
    row_edges = [[] for _ in range(d)]
    for source_index, (i, j, g) in enumerate(
        zip(
            np.asarray(sim.gap["i"]),
            np.asarray(sim.gap["j"]),
            np.asarray(sim.gap["conductance"]),
        )
    ):
        for row, neighbor in ((int(i), int(j)), (int(j), int(i))):
            owner, remote_owner = row // local_size, neighbor // local_size
            remote = owner != remote_owner
            slot = (
                sends[remote_owner][owner].index(neighbor % local_size) if remote else 0
            )
            row_edges[owner].append(
                (
                    row % local_size,
                    neighbor % local_size,
                    remote_owner,
                    slot,
                    remote,
                    g,
                    source_index,
                )
            )
    width = max(1, max(map(len, row_edges)))
    columns = [
        np.zeros((d, width), dtype=dtype)
        for dtype in (np.int32, np.int32, np.int32, np.int32, bool, float, np.int32)
    ]
    columns[-1][:] = -1
    for owner, values in enumerate(row_edges):
        for index, row in enumerate(values):
            for column, value in zip(columns, row):
                column[owner, index] = value
    pre, post = np.asarray(sim.syn_pre_idx), np.asarray(sim.syn_post_idx)
    owned = [[] for _ in range(d)]
    for edge, j in enumerate(post):
        owned[int(j) // local_size].append(edge)
    slots = max(1, max(map(len, owned)))
    syn = {
        k: np.zeros((d, slots), dtype=t)
        for k, t in (
            ("pre_local", np.int32),
            ("pre_owner", np.int32),
            ("pre_slot", np.int32),
            ("pre_remote", bool),
            ("post_local", np.int32),
        )
    }
    syn_ids = np.full((d, slots), -1, dtype=np.int32)
    for owner, edges in enumerate(owned):
        for k, edge in enumerate(edges):
            i, j = int(pre[edge]), int(post[edge])
            source = i // local_size
            syn_ids[owner, k] = edge
            syn["pre_local"][owner, k] = i % local_size
            syn["pre_owner"][owner, k] = source
            syn["pre_remote"][owner, k] = source != owner
            syn["pre_slot"][owner, k] = (
                sends[source][owner].index(i % local_size) if source != owner else 0
            )
            syn["post_local"][owner, k] = j % local_size
    mesh = Mesh(np.array(devices), ("device",))

    def place(array):
        return jax.device_put(
            array, NamedSharding(mesh, P("device", *([None] * (array.ndim - 1))))
        )

    return PartitionPlan(
        mesh,
        n,
        local_size,
        place(send_indices),
        place(send_mask),
        *[place(column) for column in columns[:-1]],
        sum(len(v) for row in sends for v in row),
        place(columns[-1]),
        place(syn_ids),
        *[
            place(syn[k])
            for k in ("pre_local", "pre_owner", "pre_slot", "pre_remote", "post_local")
        ],
        np.array([len(v) for v in owned]),
    )


def halo_exchange(plan, values):
    """Return [target,source,slot,features] with only boundary values packed."""
    d, local = plan.partitions, plan.local_size
    if values.shape[:2] != (d, local):
        raise ValueError(
            "partitioned halo values require [partitions, local_neurons, ...]"
        )
    features = values.shape[2:]

    @partial(
        jax.shard_map,
        mesh=plan.mesh,
        in_specs=(P("device"), P("device"), P("device")),
        out_specs=P("device"),
        check_vma=False,
    )
    def exchange(local_values, send_indices, send_mask):
        v = local_values.reshape((local,) + features)
        indices = send_indices.reshape((d, -1))
        mask = send_mask.reshape((d, -1) + (1,) * len(features))
        packed = jnp.where(mask, v[indices], 0)
        received = jax.lax.all_to_all(packed, "device", split_axis=0, concat_axis=0)
        return received[None]

    return exchange(values, plan.send_indices, plan.send_mask)


def distributed_voltage_solve(sim, plan, diagonal, rhs, old_voltage):
    from .solver import pcg

    if "axial" in sim.neuron_params:
        raise NotImplementedError(
            "distributed solve does not support per-neuron axial conductances"
        )
    d, local, c = plan.partitions, plan.local_size, sim.resolution.n_comp
    size = d * local

    def reshape(value, pad):
        value = jnp.pad(
            value, ((0, size - plan.n_neurons), (0, 0)), constant_values=pad
        )
        return jax.device_put(
            value.reshape(d, local, c), NamedSharding(plan.mesh, P("device"))
        )

    diagonal, rhs, old_voltage = (
        reshape(diagonal, 1),
        reshape(rhs, 0),
        reshape(old_voltage, 0),
    )
    edge_conductance = (
        jnp.where(
            plan.edge_source_index >= 0,
            sim.gap["conductance"][jnp.maximum(plan.edge_source_index, 0)],
            0.0,
        )
        if len(sim.gap["conductance"])
        else jnp.zeros_like(plan.edge_conductance)
    )
    input_specs = tuple(P("device") for _ in range(10))

    @partial(
        jax.shard_map,
        mesh=plan.mesh,
        in_specs=input_specs,
        out_specs=P("device"),
        check_vma=False,
    )
    def local_product(
        x, diag, sends, masks, rows, neighbors, owners, slots, remote, conductance
    ):
        x, diag = x.reshape(local, c), diag.reshape(local, c)
        packed = x[sends.reshape(d, -1)] * masks.reshape(d, -1, 1)
        halo = jax.lax.all_to_all(packed, "device", split_axis=0, concat_axis=0)
        rows, neighbors, owners, slots, remote, conductance = [
            v.reshape(-1) for v in (rows, neighbors, owners, slots, remote, conductance)
        ]
        neighbor_voltage = jnp.where(remote, halo[owners, slots, 0], x[neighbors, 0])
        gap = (
            jnp.zeros(local, dtype=x.dtype)
            .at[rows]
            .add(conductance * (x[rows, 0] - neighbor_voltage))
        )
        result = diag * x
        result = result.at[:, 0].add(gap)
        flow = sim.resolution.axial_nS * (x[:, :-1] - x[:, 1:])
        result = result.at[:, :-1].add(flow).at[:, 1:].add(-flow)
        return result[None]

    def matvec(x):
        return local_product(
            x,
            diagonal,
            plan.send_indices,
            plan.send_mask,
            plan.row_indices,
            plan.neighbor_indices,
            plan.neighbor_owners,
            plan.halo_slots,
            plan.remote,
            edge_conductance,
        )

    gap_diag = jnp.zeros((d, local), dtype=diagonal.dtype)
    gap_diag = gap_diag.at[jnp.arange(d)[:, None], plan.row_indices].add(
        edge_conductance
    )
    blocks = jax.vmap(jax.vmap(jnp.diag))(diagonal)
    blocks = blocks.at[:, :, 0, 0].add(gap_diag)
    for k in range(c - 1):
        blocks = (
            blocks.at[:, :, k, k]
            .add(sim.resolution.axial_nS)
            .at[:, :, k + 1, k + 1]
            .add(sim.resolution.axial_nS)
        )
        blocks = (
            blocks.at[:, :, k, k + 1]
            .add(-sim.resolution.axial_nS)
            .at[:, :, k + 1, k]
            .add(-sim.resolution.axial_nS)
        )
    from jax.scipy.linalg import cho_solve

    factors = jnp.linalg.cholesky(blocks)

    def precondition(r):
        return jax.vmap(jax.vmap(lambda factor, b: cho_solve((factor, True), b)))(
            factors, r
        )

    def solve(_, b):
        return pcg(
            matvec,
            b,
            precondition,
            old_voltage,
            sim.resolution.solve_tolerance,
            sim.resolution.max_cg_iterations,
        )

    voltage, iterations = jax.lax.custom_linear_solve(
        matvec, rhs, solve=solve, symmetric=True, has_aux=True
    )
    residual = jnp.linalg.norm(matvec(voltage) - rhs) / jnp.maximum(
        jnp.linalg.norm(rhs), 1e-12
    )
    return voltage.reshape(size, c)[: plan.n_neurons], residual, iterations


def synaptic_state_bytes(sim, plan):
    """Per-device bytes of owned per-synapse state and parameters (memory balance).

    Counts the state a device would hold under this plan: receptors, resources,
    facilitation, last-release time, plus every per-synapse parameter array.
    """
    e = len(sim.syn_pre_idx)
    dtype = sim.neuron_params["capacitance"].dtype
    per_edge = ((len(sim.receptor_records) + 3) * dtype.itemsize if e else 0) + sum(
        a.nbytes // e
        for a in sim.syn_params.values()
        if e and hasattr(a, "shape") and a.ndim and a.shape[0] == e
    )
    return [int(c) * per_edge for c in plan.syn_counts]


def distributed_synapses(sim, plan, old, spikes, activity):
    """Per-device synaptic update and accumulation; pass as `step(synapses=...)`.

    Returns (receptors, resources, facilitation, syn_by_receptor,
    syn_reversal_by_receptor) matching the single-device step. Global state
    arrays are gathered into the owner layout and scattered back each call;
    a persistent sharded state layout is not implemented. Verified on virtual
    CPU devices only.
    """
    d, local, c = plan.partitions, plan.local_size, sim.resolution.n_comp
    n, e = plan.n_neurons, len(sim.syn_pre_idx)
    ids = plan.syn_edge_ids
    valid, safe = ids >= 0, jnp.maximum(ids, 0)
    pad = d * local - n
    features = jnp.stack([activity, spikes.astype(activity.dtype)], axis=1)
    features = jnp.pad(features, ((0, pad), (0, 0))).reshape(d, local, 2)
    p = sim.syn_params
    gathered = (
        old.facilitation[safe],
        old.resources[safe],
        old.receptors[safe],
        p["tau_rec"][safe],
        p["tau_fac"][safe],
        p["U"][safe],
        p["density"][safe],
        p["gate"][safe],
        p["compartment"][safe],
        p["reversal"][safe],
    )
    structure = (
        plan.send_indices,
        plan.send_mask,
        plan.syn_pre_local,
        plan.syn_pre_owner,
        plan.syn_pre_slot,
        plan.syn_pre_remote,
        plan.syn_post_local,
        valid,
    )
    from .simulation import synaptic_update

    @partial(
        jax.shard_map,
        mesh=plan.mesh,
        in_specs=tuple(P("device") for _ in range(1 + len(structure) + len(gathered))),
        out_specs=P("device"),
        check_vma=False,
    )
    def local_synapses(feat, sends, masks, *rest):
        pre_local, pre_owner, pre_slot, pre_remote, post_local, ok = [
            v.reshape(v.shape[1:]) for v in rest[:6]
        ]
        fac, res, rec, tau_rec, tau_fac, base_u, density, gate, comp, reversal = [
            v.reshape(v.shape[1:]) for v in rest[6:]
        ]
        f = feat.reshape(local, 2)
        packed = f[sends.reshape(d, -1)] * masks.reshape(d, -1, 1)
        halo = jax.lax.all_to_all(packed, "device", split_axis=0, concat_axis=0)
        source = jnp.where(pre_remote[:, None], halo[pre_owner, pre_slot], f[pre_local])
        fac, res, rec = synaptic_update(
            sim,
            (fac, res, rec),
            {"tau_rec": tau_rec, "tau_fac": tau_fac, "U": base_u},
            source[:, 1] > 0.5,
            source[:, 0],
        )
        g = density * p["gbar"] * rec * gate[:, None] * ok[:, None]
        flat = post_local * c + comp
        total = jax.ops.segment_sum(g, flat, num_segments=local * c)
        rev = jax.ops.segment_sum(g * reversal, flat, num_segments=local * c)
        r = rec.shape[1]
        return (
            fac[None],
            res[None],
            rec[None],
            total.reshape(1, local, c, r),
            rev.reshape(1, local, c, r),
        )

    fac, res, rec, total, rev = local_synapses(
        features, *structure[:2], *structure[2:], *gathered
    )
    target = jnp.where(valid, ids, e).reshape(-1)

    def scatter(base, new):
        return base.at[target].set(new.reshape((-1,) + new.shape[2:]), mode="drop")

    r = len(sim.receptor_records)
    return (
        scatter(old.receptors, rec),
        scatter(old.resources, res),
        scatter(old.facilitation, fac),
        total.reshape(d * local, c, r)[:n],
        rev.reshape(d * local, c, r)[:n],
    )


def distributed_step(sim, plan, old, current):
    """Full step with partitioned synapses and the distributed voltage solve."""
    from dataclasses import replace

    from .simulation import step

    return step(
        replace(sim, partition_plan=plan),
        old,
        current,
        synapses=lambda s, o, sp, act: distributed_synapses(s, plan, o, sp, act),
    )


def with_partitions(sim, devices=None):
    from dataclasses import replace

    plan = partition_graph(sim, devices)
    return replace(
        sim,
        partition_plan=plan,
        metadata={
            **sim.metadata,
            "device_partitions": plan.partitions,
            "halo_neurons": plan.halo_neurons,
        },
    )
