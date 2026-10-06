"""Executable reference baselines; external fly jobs stay outside JAX training."""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .provenance import content_hash


@dataclass(frozen=True)
class LinearDynamics:
    weights: np.ndarray  # post, pre, per second
    bias: np.ndarray
    baseline_id: str
    training_hash: str | None = None

    def predict(self, initial, currents, dt_s):
        weights, bias = jnp.asarray(self.weights), jnp.asarray(self.bias)
        identity = jnp.eye(len(initial))

        # Semi-implicit linear baseline. No simulation sign invariant applies
        # to B0, which deliberately stores transmitter-derived signed weights.
        def step(previous, current):
            state = jnp.linalg.solve(
                identity - dt_s * weights, previous + dt_s * (bias + current)
            )
            return state, state

        _, result = jax.lax.scan(step, jnp.asarray(initial), jnp.asarray(currents))
        return result.T


def connectome_only(
    graph, decay_s=0.5, edge_gain=0.01, transmitter_signs=(1.0, -1.0, 1.0)
):
    n = len(graph.neuron_ids)
    indices = {int(v): i for i, v in enumerate(graph.neuron_ids)}
    w = np.zeros((n, n))
    for edge in graph.connectome.synapses.to_pylist():
        sign = 1.0
        if edge["nt_pred"] is not None:
            if len(edge["nt_pred"]) != len(transmitter_signs):
                raise ValueError("B0 transmitter alphabet differs from supplied signs")
            sign = float(np.asarray(edge["nt_pred"]) @ np.asarray(transmitter_signs))
        w[indices[edge["post_id"]], indices[edge["pre_id"]]] += (
            edge["size"] * edge_gain * sign
        )
    w -= np.diag(np.abs(w).sum(axis=1) + 1 / decay_s)
    return LinearDynamics(w, np.zeros(n), "B0")


def fit_linear(recording, dt_s, graph=None, ridge=1e-3, currents=None):
    """B2 unrestricted linear dynamics; B5 anatomy-constrained ridge fit.

    This is a connectome-constrained linear reimplementation, not a claim
    that published Creamer et al. results have been reproduced.
    """
    y = np.asarray(recording, dtype=float)
    if y.ndim != 2 or y.shape[1] < 3 or dt_s <= 0 or ridge <= 0:
        raise ValueError("linear fit requires a recording, timestep and positive ridge")
    n = len(y)
    allowed = np.ones((n, n), dtype=bool) if graph is None else np.eye(n, dtype=bool)
    if graph is not None:
        index = {int(v): i for i, v in enumerate(graph.neuron_ids)}
        if len(index) != n:
            raise ValueError("B5 recording must align to graph")
        for edge in graph.connectome.synapses.to_pylist():
            allowed[index[edge["post_id"]], index[edge["pre_id"]]] = True
        for edge in graph.connectome.contacts.to_pylist():
            if edge["gap_junction_observed"] is not False:
                i, j = index[edge["i_id"]], index[edge["j_id"]]
                allowed[i, j] = allowed[j, i] = True
    response = (y[:, 1:] - y[:, :-1]) / dt_s
    if currents is not None:
        if np.shape(currents) != response.T.shape:
            raise ValueError("baseline current shape differs")
        response -= np.asarray(currents).T
    weights, bias = np.zeros((n, n)), np.zeros(n)
    for post in range(n):
        columns = np.flatnonzero(allowed[post])
        design = np.column_stack([y[columns, :-1].T, np.ones(y.shape[1] - 1)])
        penalty = ridge * np.eye(design.shape[1])
        penalty[-1, -1] = 0
        coefficient = np.linalg.solve(
            design.T @ design + penalty, design.T @ response[post]
        )
        weights[post, columns] = coefficient[:-1]
        bias[post] = coefficient[-1]
    return LinearDynamics(
        weights,
        bias,
        "B2" if graph is None else "B5",
        content_hash(y, dt_s, ridge, currents),
    )


@dataclass(frozen=True)
class BlackBoxKernel:
    params: dict
    training_hash: str | None = None

    @classmethod
    def initialize(cls, input_dimension, hidden=16, seed=0):
        a, b = jax.random.split(jax.random.key(seed))
        return cls(
            {
                "w1": jax.random.normal(a, (input_dimension, hidden)) * 0.1,
                "b1": jnp.zeros(hidden),
                "w2": jax.random.normal(b, (hidden, 1)) * 0.1,
                "b2": jnp.zeros(1),
            }
        )

    def predict(self, features):
        p = self.params
        return (jnp.tanh(features @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]).ravel()

    def fit(self, features, target, steps=100, seed=0):
        from dataclasses import replace

        from .training import train

        x, y = jnp.asarray(features), jnp.asarray(target)
        updated, history = train(
            self.params,
            lambda p, _: jnp.mean((replace(self, params=p).predict(x) - y) ** 2),
            steps=steps,
            seed=seed,
        )
        return replace(
            self,
            params=updated,
            training_hash=content_hash(np.asarray(x), np.asarray(y)),
        ), history


def synapse_only(sim):
    from dataclasses import replace

    return replace(
        sim,
        neuromod={
            **sim.neuromod,
            "release": jnp.zeros_like(sim.neuromod["release"]),
            "sensitivity": jnp.zeros_like(sim.neuromod["sensitivity"]),
        },
        metadata={**sim.metadata, "baseline": "B4"},
    )


def external_predictions(path, baseline_id):
    """Read separate B6/fly-B0 jobs with source/version/training provenance."""
    if baseline_id not in {"B0_fly", "B6"}:
        raise ValueError("external adapter only accepts fly B0/B6")
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if not all(
        metadata.get(k) for k in ("source", "version", "training_hash", "split_hash")
    ):
        raise ValueError("external baseline lacks provenance")
    values = np.load(path, allow_pickle=False)["predictions"]
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite external baseline predictions")
    metadata["artifact_hash"] = content_hash(values)
    return values, metadata
