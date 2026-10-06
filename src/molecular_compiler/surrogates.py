"""M7: held-out validated cascade, conditioned envelopes and safe fallback."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Surrogate:
    family: str
    lower: tuple
    upper: tuple
    coefficients: tuple = ()
    heldout_error: float | None = None
    conditioned: bool = False

    @property
    def state_dimension(self):
        if self.family == "fixed_ode":
            return len(self.coefficients[0])
        if self.family == "neural_ode":
            return len(self.coefficients[3])
        return 0

    def inside(self, inputs):
        return jnp.all(
            (inputs >= jnp.asarray(self.lower)) & (inputs <= jnp.asarray(self.upper)),
            axis=-1,
        )

    def derivative(self, state, inputs):
        if self.family == "fixed_ode":
            features = jnp.concatenate(
                [jnp.ones(state.shape[:-1] + (1,)), state, inputs], axis=-1
            )
            return features @ jnp.asarray(self.coefficients)
        if self.family == "neural_ode":
            features = jnp.concatenate([state, inputs], axis=-1)
            w1, b1, w2, b2 = [jnp.asarray(x) for x in self.coefficients]
            return jnp.tanh(features @ w1 + b1) @ w2 + b2
        raise ValueError("full-model fallback has no surrogate derivative")


def fit_surrogate(
    states,
    inputs,
    derivatives,
    heldout,
    tolerance=0.05,
    seed=0,
    hidden=8,
    conditioned=False,
):
    """Fit per molecular type to full-model derivatives; validate held-out derivatives.

    Callers must additionally validate rollout error against their full-model backend
    before attaching a candidate to a SimGraph. No fit is silently trusted.
    """
    s, u, d = map(np.asarray, (states, inputs, derivatives))
    hs, hu, hd = map(np.asarray, heldout)
    if s.ndim != 2 or not 2 <= s.shape[1] <= 4 or u.ndim != 2 or d.shape != s.shape:
        raise ValueError("surrogates require 2-4 states and aligned full-model samples")
    if (
        len(s) != len(u)
        or hs.shape != hd.shape
        or hu.shape[1] != u.shape[1]
        or not len(hs)
    ):
        raise ValueError("invalid held-out surrogate samples")
    lo, hi = tuple(u.min(axis=0)), tuple(u.max(axis=0))
    denominator = max(np.sqrt(np.mean(hd**2)), 1e-8)
    features = np.column_stack([np.ones(len(s)), s, u])
    coefficients = np.linalg.lstsq(features, d, rcond=None)[0]
    fixed = Surrogate(
        "fixed_ode", lo, hi, tuple(map(tuple, coefficients)), conditioned=conditioned
    )
    error = (
        np.sqrt(np.mean((np.asarray(fixed.derivative(hs, hu)) - hd) ** 2)) / denominator
    )
    if error <= tolerance:
        return Surrogate(
            "fixed_ode", lo, hi, fixed.coefficients, float(error), conditioned
        )
    rng = np.random.default_rng(seed)
    x = np.column_stack([s, u])
    hx = np.column_stack([hs, hu])
    shapes = [(x.shape[1], hidden), (hidden,), (hidden, s.shape[1]), (s.shape[1],)]
    sizes = [np.prod(shape) for shape in shapes]
    offsets = np.cumsum([0] + sizes)

    def unpack(p):
        return [
            p[offsets[i] : offsets[i + 1]].reshape(shape)
            for i, shape in enumerate(shapes)
        ]

    def predict(p, features):
        w1, b1, w2, b2 = unpack(p)
        return np.tanh(features @ w1 + b1) @ w2 + b2

    p0 = rng.normal(0, 0.1, offsets[-1])
    fit = least_squares(lambda p: (predict(p, x) - d).ravel(), p0, max_nfev=100)
    error = np.sqrt(np.mean((predict(fit.x, hx) - hd) ** 2)) / denominator
    if error <= tolerance:
        return Surrogate(
            "neural_ode",
            lo,
            hi,
            tuple(a.tolist() for a in unpack(fit.x)),
            float(error),
            conditioned,
        )
    return Surrogate(
        "full", lo, hi, heldout_error=float(error), conditioned=conditioned
    )


def validate_rollout(candidate, predicted, reference, tolerance):
    error = float(
        np.sqrt(np.mean((np.asarray(predicted) - reference) ** 2))
        / max(np.sqrt(np.mean(np.asarray(reference) ** 2)), 1e-8)
    )
    if error > tolerance:
        return Surrogate(
            "full",
            candidate.lower,
            candidate.upper,
            heldout_error=error,
            conditioned=candidate.conditioned,
        )
    return Surrogate(
        candidate.family,
        candidate.lower,
        candidate.upper,
        candidate.coefficients,
        error,
        candidate.conditioned,
    )


def reduce_graph(sim, samples, heldout_samples, rollout_validator, tolerance=0.05):
    """Compile-time per-type cascade validated against the full Jaxley backend.

    samples[type] and heldout_samples[type] are (states, inputs, derivatives)
    from full-model runs. rollout_validator(type, candidate) returns the
    candidate and full-model held-out trajectories. Missing data falls back
    to full kinetics; fitting runs exactly once per molecular type.
    """
    from dataclasses import replace

    surrogates = []
    for type_index, previous in enumerate(sim.surrogates):
        if type_index not in samples or type_index not in heldout_samples:
            surrogates.append(previous)
            continue
        candidate = fit_surrogate(
            *samples[type_index],
            heldout_samples[type_index],
            tolerance=tolerance,
            conditioned=True,
        )
        if candidate.family != "full":
            predicted, reference = rollout_validator(type_index, candidate)
            candidate = validate_rollout(candidate, predicted, reference, tolerance)
        surrogates.append(candidate)
    families = {
        name: sum(s.family == name for s in surrogates)
        for name in ("fixed_ode", "neural_ode", "full")
    }
    return replace(
        sim,
        surrogates=tuple(surrogates),
        metadata={
            **sim.metadata,
            "surrogate_families": families,
            "surrogate_rollout_validated": True,
        },
    )
