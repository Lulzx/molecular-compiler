"""Track I, Stage I1: restore test for a candidate durable state (spec 10.4).

A candidate d is sufficient at tolerance tau if running the model, discarding
everything except d, regenerating the fast state, and continuing reproduces the
uninterrupted run's response within tau times the noise ceiling. Candidates
are tried from small to large: d0 = {u} (compiled parameters only, no state
fields), then + c(t), + m_i(t), + per-synapse state.

The restored simulator is the same compiled object, so compiled parameters
(u_a) are always kept. The test is trivial while every slow variable is a
compile-time constant: d0 passes whenever the fast state relaxes to the
uninterrupted one. It becomes informative once c(t), m_i(t) or synaptic
state evolve during a run.

The fast state is regenerated either by `relax` (zero input from rest) or by
`infer`, a regularized least-squares fit of the observed history over every
float state field not kept in d, by gradient through the simulator in the
spirit of worm-sim INITIAL-STATE.md. The kept fields are held at their
origin values over the history, which assumes they change little there.
This is a point estimate, not a claim of identifiability.

A simulator is any object with `rest() -> state` and
`run(state, currents[L, N]) -> (state, response[L, M])`, state a NamedTuple.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .simulation import Stimulus, initial_state, simulate

# SimState fields by the slow quantity they carry (spec 10.4 ordering).
SLOW_GROUPS = {
    "c": ("peptide_field", "peptide_open"),
    "m": ("signaling",),
    "synapse": (
        "receptors",
        "resources",
        "facilitation",
        "last_release_time",
        "synaptic_conductance",
        "synaptic_reversal_current",
    ),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    keep: tuple = ()  # state fields retained; everything else is regenerated


def nested_candidates(groups=SLOW_GROUPS):
    """d0 = {u}, then cumulatively adding each slow group."""
    out, keep = [Candidate("u")], ()
    for i, fields in enumerate(groups.values()):
        keep += tuple(fields)
        out.append(Candidate("+".join(["u", *list(groups)[: i + 1]]), keep))
    return out


class CompiledSimulator:
    """Adapter from a compiled `simulate` model; response is the calcium proxy."""

    def __init__(self, sim, readout=lambda traj: traj.calcium):
        self.sim, self.readout = sim, readout

    def rest(self):
        return initial_state(self.sim)

    def run(self, state, currents):
        dt = self.sim.resolution.dt_s
        traj = simulate(
            self.sim, Stimulus(currents=currents), len(currents) * dt, state0=state
        )
        return traj.final_state, self.readout(traj)


def _merge(base, other, fields):
    """`base` with `fields` taken from `other`."""
    return base._replace(**{f: getattr(other, f) for f in fields})


def infer_fast_state(
    simulator,
    kept,
    rest,
    currents,
    observed,
    keep,
    prior_weight=1e-3,
    steps=100,
    learning_rate=0.02,
):
    """Fit the regenerated fields at history start to the observed responses.

    Loss = mean squared history error + prior_weight * mean squared distance of
    the fitted fields from rest (in units of each field's spread, at least 1).
    Returns (state at the end of the history, loss trace); the lowest-loss
    iterate is kept.
    """
    fields = [
        f
        for f in rest._fields
        if f not in keep and jnp.issubdtype(getattr(rest, f).dtype, jnp.floating)
    ]
    start = {f: getattr(rest, f) for f in fields}
    scale = {f: jnp.maximum(jnp.std(start[f]), 1.0) for f in fields}
    observed = jnp.asarray(observed)

    def loss(theta):
        _, response = simulator.run(kept._replace(**theta), currents)
        prior = sum(
            jnp.mean(((theta[f] - start[f]) / scale[f]) ** 2) for f in fields
        ) / max(len(fields), 1)
        return jnp.mean((response - observed) ** 2) + prior_weight * prior

    value_and_grad = jax.value_and_grad(loss)
    optimizer = optax.adam(learning_rate)
    theta, opt = start, optimizer.init(start)
    best, best_loss, trace = start, np.inf, []
    for _ in range(steps):
        value, grad = value_and_grad(theta)
        trace.append(float(value))
        if value < best_loss:
            best, best_loss = theta, float(value)
        updates, opt = optimizer.update(grad, opt)
        theta = optax.apply_updates(theta, updates)
    final, _ = simulator.run(kept._replace(**best), currents)
    return final, trace


def restore(
    simulator, state, candidate, rest, mode="relax", relax=None, history=None, **fit
):
    """State regenerated from `state` keeping only `candidate.keep`.

    `relax` is a zero-current array [L, N] for mode "relax"; `history` is
    (currents, observed response) for mode "infer". Returns (state, loss trace).
    """
    kept = _merge(rest, state, candidate.keep)
    if mode == "relax":
        if relax is not None and len(relax):
            kept, _ = simulator.run(kept, relax)
        return kept, []
    if mode != "infer":
        raise ValueError("mode must be 'relax' or 'infer'")
    fitted, trace = infer_fast_state(
        simulator, kept, rest, history[0], history[1], candidate.keep, **fit
    )
    return _merge(fitted, state, candidate.keep), trace


@dataclass
class RestoreResult:
    candidate: str
    error: float  # RMS response error over the probe, restored vs uninterrupted
    limit: float  # tau * noise ceiling
    sufficient: bool
    history_loss: list = field(default_factory=list)


@dataclass
class RestoreReport:
    results: list
    smallest: str | None  # first sufficient candidate in the tested order


def restore_test(
    simulator,
    candidates,
    drive,
    probe,
    tau,
    noise_ceiling,
    mode="relax",
    relax_steps=200,
    history_steps=100,
    stop_early=False,
    **fit,
):
    """Run the I1 restore test over `candidates`, ordered small to large.

    `drive` [Ld, N] is run from rest, then interrupted. `probe` [Lp, N] is the
    continuation whose response is compared. The observed history for
    mode "infer" is the last `history_steps` of the drive and its response
    (noiseless here; pass real noise through the drive response if needed).
    `noise_ceiling` is the RMS trial noise in response units. Returns the
    smallest sufficient candidate, or None when none is.
    """
    drive, probe = jnp.asarray(drive), jnp.asarray(probe)
    rest = simulator.rest()
    state, drive_response = simulator.run(rest, drive)
    _, reference = simulator.run(state, probe)
    history = (drive[-history_steps:], drive_response[-history_steps:])
    relax = jnp.zeros((relax_steps, drive.shape[1]), dtype=drive.dtype)
    results = []
    for candidate in candidates:
        restored, trace = restore(
            simulator, state, candidate, rest, mode, relax, history, **fit
        )
        _, got = simulator.run(restored, probe)
        error = float(jnp.sqrt(jnp.mean((got - reference) ** 2)))
        limit = tau * noise_ceiling
        results.append(
            RestoreResult(candidate.name, error, limit, error <= limit, trace)
        )
        if stop_early and results[-1].sufficient:
            break
    smallest = next((r.candidate for r in results if r.sufficient), None)
    return RestoreReport(results, smallest)
