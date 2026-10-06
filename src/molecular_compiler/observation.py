"""M9: indicator dynamics and nuisance gains kept outside transferable rules."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .kinetics import rush_larsen

INDICATORS = {"GCaMP6s": 0.7, "GCaMP6f": 0.2, "GCaMP7f": 0.15}


def fix_gauge(log_values, type_ids, measured=None):
    """Center unmeasured log-gains within type; measured values remain fixed."""
    values = jnp.asarray(log_values)
    ids = jnp.asarray(type_ids)
    if values.shape != ids.shape:
        raise ValueError("gain and type shapes differ")
    measured = (
        jnp.zeros_like(values, dtype=bool)
        if measured is None
        else jnp.asarray(measured, dtype=bool)
    )
    # Unique types are compile-time data, not trainable identities.
    count = int(np.max(np.asarray(type_ids))) + 1 if len(type_ids) else 0
    sums = jax.ops.segment_sum(jnp.where(measured, 0, values), ids, count)
    counts = jax.ops.segment_sum((~measured).astype(values.dtype), ids, count)
    centered = values - sums[ids] / jnp.maximum(counts[ids], 1)
    return jnp.where(measured, values, centered)


@dataclass(frozen=True)
class ObservationModel:
    indicator: str = "GCaMP6s"
    log_gain: object | None = None
    type_ids: object | None = None
    measured_gain: object | None = None
    y_ref: object | None = None
    hill_coefficient: float = 2.0
    half_saturation: float = 0.001
    noise_sd: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.indicator not in INDICATORS:
            raise ValueError("unknown indicator; register its fixed kernel first")
        if self.hill_coefficient <= 0 or self.half_saturation <= 0 or self.noise_sd < 0:
            raise ValueError("invalid observation parameters")
        if self.log_gain is not None and self.type_ids is None:
            raise ValueError(
                "unmeasured gains require molecular types for gauge fixing"
            )


@dataclass
class Recording:
    y: jax.Array  # [N_obs,L]
    t: jax.Array
    neuron_map: object
    match_confidence: object
    indicator: str
    prep: str
    animal_state: dict
    y_ref: object | None = None
    metadata: dict | None = None

    def validate(self):
        y, t = np.asarray(self.y), np.asarray(self.t)
        if (
            y.ndim != 2
            or y.shape != (len(self.neuron_map), len(t))
            or len(self.match_confidence) != len(self.neuron_map)
        ):
            raise ValueError("recording shapes differ")
        if (
            not np.all(np.isfinite(y))
            or not np.all(np.isfinite(t))
            or np.any(np.diff(t) <= 0)
        ):
            raise ValueError("recording values or timestamps invalid")
        if not np.all(
            (np.asarray(self.match_confidence) >= 0)
            & (np.asarray(self.match_confidence) <= 1)
        ):
            raise ValueError("recording match confidence outside [0,1]")
        if self.prep not in {"immobilized", "freely_moving"}:
            raise ValueError("unknown preparation")
        if self.y_ref is not None and np.shape(self.y_ref) != y.shape:
            raise ValueError("reference channel shape differs")

    def training_mask(self, threshold=0.95):
        return jnp.asarray(self.match_confidence) >= threshold


def observe(traj, obs):
    n = traj.calcium.shape[1]
    dt = traj.metadata["dt_s"]

    def filter_step(previous, ca):
        current = rush_larsen(previous, ca, INDICATORS[obs.indicator], dt)
        return current, current

    _, filtered = jax.lax.scan(
        filter_step, jnp.zeros(n, dtype=traj.calcium.dtype), traj.calcium
    )
    log_gain = jnp.zeros(n) if obs.log_gain is None else jnp.asarray(obs.log_gain)
    if log_gain.shape != (n,):
        raise ValueError("observation gains require one value per neuron")
    gauge = "unit_gain"
    if obs.y_ref is not None:
        reference = jnp.asarray(obs.y_ref)
        if reference.shape != (n, len(traj.t)) or np.any(np.asarray(reference) <= 0):
            raise ValueError("reference fluorescence must be positive and aligned")
        log_gain = jnp.log(reference.mean(axis=1))
        gauge = "ratiometric"
    elif obs.type_ids is not None:
        log_gain = fix_gauge(log_gain, obs.type_ids, obs.measured_gain)
        gauge = "within_type_mean_log_gain_zero"
    calcium_power = jnp.maximum(filtered, 0) ** obs.hill_coefficient
    signal = (
        jnp.exp(log_gain)[None]
        * calcium_power
        / (obs.half_saturation**obs.hill_coefficient + calcium_power)
    )
    noise = obs.noise_sd * jax.random.normal(
        jax.random.key(obs.seed), signal.shape, dtype=signal.dtype
    )
    metadata = {
        **traj.metadata,
        "gauge": gauge,
        "noise_sd": obs.noise_sd,
        "amplitude_identifiability": "across_types"
        if gauge == "ratiometric"
        else "within_types",
    }
    return Recording(
        (signal + noise).T,
        traj.t,
        traj.neuron_ids,
        jnp.ones(n),
        obs.indicator,
        "freely_moving"
        if traj.metadata["boundary_condition"] == "closed_loop"
        else "immobilized",
        {},
        obs.y_ref,
        metadata,
    )
