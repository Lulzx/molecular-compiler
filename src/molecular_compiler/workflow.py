"""M9 training workflow: per-animal opsin drives and indicator gains under the 6.3 gauge."""

import jax
import jax.numpy as jnp
import numpy as np

from .observation import fix_gauge
from .training import train, trajectory_nll


def _measured_mask(values, n):
    return jnp.zeros(n, dtype=bool) if values is None else ~jnp.isnan(values)


def _gauged(raw, type_ids, measured_values):
    """Centered log-values; NaN-marked-as-measured entries are held exactly fixed."""
    if measured_values is None:
        return fix_gauge(raw, type_ids)
    values = jnp.asarray(measured_values)
    measured = ~jnp.isnan(values)
    return fix_gauge(jnp.where(measured, values, raw), type_ids, measured)


def _type_mean_residual(values, type_ids, measured):
    """Max |within-type mean| over unmeasured entries (zero up to float error)."""
    ids = jnp.asarray(type_ids)
    count = int(np.max(type_ids)) + 1
    free = ~measured
    sums = jax.ops.segment_sum(jnp.where(free, values, 0.0), ids, count)
    sizes = jax.ops.segment_sum(free.astype(values.dtype), ids, count)
    return float(jnp.max(jnp.abs(sums / jnp.maximum(sizes, 1))))


def fit_animal_nuisance(
    recording,
    forward,
    gain_type_ids,
    drive_type_ids,
    measured_log_gain=None,
    measured_log_drive=None,
    fit_gain=True,
    steps=200,
    learning_rate=0.05,
    noise_sd=0.1,
    seed=0,
):
    """Fit log-drives (one per stimulated neuron) and log-gains jointly (M9, 6.3).

    `forward(log_drive, log_gain) -> [N_obs, L]` is the simulator plus observation
    model. It only ever receives gauged values: unmeasured entries have zero mean
    within each molecular type; measured entries (non-NaN in `measured_log_*`) are
    held at their measured values. With `fit_gain=False` (ratiometric or measured
    gains) gains are not fitted and stay at zero or their measured values.
    """
    gain_ids, drive_ids = np.asarray(gain_type_ids), np.asarray(drive_type_ids)
    n_gain, n_drive = len(gain_ids), len(drive_ids)
    observed = jnp.asarray(recording.y)
    mask = recording.training_mask()
    if observed.shape[0] != n_gain:
        raise ValueError("one gain type per observed neuron required")
    measured_gain = (
        None if measured_log_gain is None else jnp.asarray(measured_log_gain)
    )
    measured_drive = (
        None if measured_log_drive is None else jnp.asarray(measured_log_drive)
    )

    def gauge(raw):
        gain_raw = raw["gain"] if fit_gain else jnp.zeros(n_gain)
        return (
            _gauged(raw["drive"], drive_ids, measured_drive),
            _gauged(gain_raw, gain_ids, measured_gain),
        )

    def loss(raw, _):
        drive, gain = gauge(raw)
        return trajectory_nll(forward(drive, gain), observed, noise_sd, mask)

    raw = {"drive": jnp.zeros(n_drive), "gain": jnp.zeros(n_gain)}
    initial = float(loss(raw, 0))
    raw, history = train(
        raw, loss, steps=steps, learning_rate=learning_rate, weight_decay=0, seed=seed
    )
    drive, gain = gauge(raw)
    return {
        "log_drive": drive,
        "log_gain": gain,
        "initial_loss": initial,
        "final_loss": float(loss(raw, 0)),
        "history": history,
        "gauge": "within_type_mean_log_drive_and_log_gain_zero",
        "gauge_residual": {
            "drive": _type_mean_residual(
                drive, drive_ids, _measured_mask(measured_drive, n_drive)
            ),
            "gain": _type_mean_residual(
                gain, gain_ids, _measured_mask(measured_gain, n_gain)
            ),
        },
        "amplitude_identifiability": "within_types",
    }
