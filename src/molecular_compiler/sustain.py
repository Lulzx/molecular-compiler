"""M13-R3 sustain criterion and M13-R4 ladder climb."""

from dataclasses import dataclass

import numpy as np

from molecular_compiler.body_ladder import (
    RUNGS,
    BodyNotConfigured,
    ClosedLoopRecord,
)
from molecular_compiler.training import statistic_features

SUSTAINS_THIS = "sustains_this_worm"
SUSTAINS_A = "sustains_a_worm"
FAILS = "fails"
OCCUPANCY_BINS = 5  # trailing columns of statistic_features


def _summaries(activity, n_samples):
    """Correlation matrix, per-neuron PSD and occupancy from the first n_samples."""
    y = np.asarray(activity, dtype=float)[:, :n_samples]
    n = y.shape[0]
    features = np.asarray(statistic_features(y))
    return (
        features[:, :n],
        features[:, n:-OCCUPANCY_BINS],
        features[:, -OCCUPANCY_BINS:],
    )


def _distances(a, b):
    """(1 - correlation-matrix similarity, PSD L1 distance, occupancy total variation)."""
    iu = np.triu_indices(a[0].shape[0], k=1)
    # A constant (e.g. silent) matrix has undefined correlation: treat as none.
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.nan_to_num(np.corrcoef(a[0][iu], b[0][iu])[0, 1])
    return np.array(
        [
            1.0 - similarity,
            0.5 * np.abs(a[1] - b[1]).sum(axis=1).mean(),
            0.5 * np.abs(a[2] - b[2]).sum(axis=1).mean(),
        ]
    )


METRICS = ("correlation_dissimilarity", "psd_distance", "occupancy_divergence")


def silent_saturated(
    activity,
    reference_scale,
    bounds=None,
    silent_rel=0.05,
    runaway_factor=10.0,
    saturated_time_frac=0.5,
    bound_tol=0.01,
):
    """Boolean (silent, saturated) masks per neuron.

    Silent: temporal std below silent_rel * reference_scale. Saturated: any
    nonfinite sample; std above runaway_factor * reference_scale; or, when
    bounds=(lo, hi) is given, at least saturated_time_frac of samples within
    bound_tol of the range of either bound.
    """
    y = np.asarray(activity, dtype=float)
    finite = np.all(np.isfinite(y), axis=1)
    std = np.std(np.where(np.isfinite(y), y, 0.0), axis=1)
    silent = finite & (std < silent_rel * reference_scale)
    saturated = ~finite | (std > runaway_factor * reference_scale)
    if bounds is not None:
        lo, hi = bounds
        tol = bound_tol * (hi - lo)
        pinned = (y <= lo + tol) | (y >= hi - tol)
        saturated |= pinned.mean(axis=1) >= saturated_time_frac
    return silent, saturated


@dataclass(frozen=True)
class SpontaneousVerdict:
    passed: bool
    distances: dict  # metric -> emulation distance to the recorded animals
    limits: dict  # metric -> leave-one-animal-out quantile limit
    n_silent: int
    n_saturated: int
    max_silent_in_data: int
    max_saturated_in_data: int


def spontaneous_activity_check(
    emulation, recordings, quantile=1.0, n_samples=None, bounds=None, **neuron_rule
):
    """M13-R3 (i). Neurons must be aligned across emulation and recordings.

    Per metric, d(x, y) is the Section 9.1 distance between two activity arrays.
    Each recorded animal k gets a leave-one-animal-out distance r_k, its mean d to
    the other animals; the emulation gets e, its mean d to all recorded animals.
    The emulation passes a metric when e <= the `quantile` of {r_k} (default 1.0,
    the maximum, i.e. no farther from the population than its most atypical
    animal; closer is never penalized). All three metrics must pass, and the
    emulation may have no more silent and no more saturated neurons than the
    worst recorded animal. All arrays are truncated to a common length
    (n_samples, default the shortest) so PSDs are comparable.
    """
    recordings = [np.asarray(r, dtype=float) for r in recordings]
    if len(recordings) < 3:
        raise ValueError("across-animal spread needs at least three recorded animals")
    emulation = np.asarray(emulation, dtype=float)
    n_samples = n_samples or min(r.shape[1] for r in [emulation, *recordings])
    if any(r.shape[0] != emulation.shape[0] for r in recordings):
        raise ValueError("neuron sets differ between emulation and recordings")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    data = [_summaries(r, n_samples) for r in recordings]
    emu = _summaries(emulation, n_samples)
    pair = np.zeros((len(data), len(data), len(METRICS)))
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            pair[i, j] = pair[j, i] = _distances(data[i], data[j])
    loo = pair.sum(axis=1) / (len(data) - 1)  # (animals, metrics)
    score = np.mean([_distances(emu, d) for d in data], axis=0)
    limit = np.quantile(loo, quantile, axis=0)
    scale = float(np.median([np.std(r, axis=1) for r in recordings]))
    counts = np.array(
        [
            [
                m.sum()
                for m in silent_saturated(
                    r[:, :n_samples], scale, bounds, **neuron_rule
                )
            ]
            for r in recordings
        ]
    )
    n_silent, n_sat = (
        int(m.sum())
        for m in silent_saturated(
            emulation[:, :n_samples], scale, bounds, **neuron_rule
        )
    )
    passed = bool(
        np.all(score <= limit)
        and n_silent <= counts[:, 0].max()
        and n_sat <= counts[:, 1].max()
    )
    return SpontaneousVerdict(
        passed,
        dict(zip(METRICS, score.tolist(), strict=True)),
        dict(zip(METRICS, limit.tolist(), strict=True)),
        n_silent,
        n_sat,
        int(counts[:, 0].max()),
        int(counts[:, 1].max()),
    )


@dataclass(frozen=True)
class IdentityVerdict:
    passed: bool
    checkpoints_s: tuple
    margins: tuple  # (nearest other animal distance) - (distance to u_a); > 0 is a pass


def identity_check(activity, dt_s, decoder, u_a, other_u, checkpoints_s, window_s):
    """M13-R3 (ii). At every checkpoint the decoded latent must be closer to u_a.

    activity is (n, T) sampled every dt_s. The window for a checkpoint c is the
    window_s of activity ending at c (the first window_s of the run for earlier
    checkpoints); decoder maps that (n, w) window to a latent vector.
    """
    activity = np.asarray(activity, dtype=float)
    u_a, other_u = np.asarray(u_a, float), np.asarray(other_u, float)
    if other_u.ndim != 2 or len(other_u) < 1 or len(checkpoints_s) < 1:
        raise ValueError("identity check needs other animals and checkpoints")
    width = round(window_s / dt_s)
    margins = []
    for c in checkpoints_s:
        end = min(max(round(c / dt_s), width), activity.shape[1])
        if end - width < 0 or end > activity.shape[1]:
            raise ValueError("checkpoint window exceeds the emulated activity")
        u = np.asarray(decoder(activity[:, end - width : end]), float)
        margins.append(
            float(np.min(np.linalg.norm(other_u - u, axis=1)) - np.linalg.norm(u_a - u))
        )
    return IdentityVerdict(
        bool(np.all(np.asarray(margins) > 0)), tuple(checkpoints_s), tuple(margins)
    )


@dataclass(frozen=True)
class SustainVerdict:
    outcome: str  # sustains_this_worm | sustains_a_worm | fails
    spontaneous: SpontaneousVerdict
    identity: IdentityVerdict

    @property
    def passed(self):
        return self.outcome == SUSTAINS_THIS


def sustain(
    emulation,
    recordings,
    decoder,
    u_a,
    other_u,
    dt_s,
    checkpoints_s,
    window_s,
    quantile=1.0,
    **kwargs,
):
    """M13-R3. Part (i) is the population-spread check, part (ii) the identity check.

    Passing (i) and (ii) sustains this worm; (i) only sustains a worm but not this
    worm; failing (i) fails regardless of (ii). kwargs go to part (i).
    """
    spontaneous = spontaneous_activity_check(
        emulation, recordings, quantile=quantile, **kwargs
    )
    identity = identity_check(
        emulation, dt_s, decoder, u_a, other_u, checkpoints_s, window_s
    )
    if not spontaneous.passed:
        outcome = FAILS
    else:
        outcome = SUSTAINS_THIS if identity.passed else SUSTAINS_A
    return SustainVerdict(outcome, spontaneous, identity)


@dataclass(frozen=True)
class RungResult:
    rung: str
    status: str  # pass | fail | unavailable
    record: ClosedLoopRecord
    verdict: object = None
    note: str = ""


@dataclass(frozen=True)
class LadderReport:
    results: tuple
    minimum_rung: str | None
    body_model_defects: tuple  # rungs that failed after a lower rung passed


def climb_ladder(rungs, run_fn, evaluate_fn, horizon_s, check_higher=True):
    """M13-R4. Test rungs in ladder order and report each one up to the first pass.

    run_fn(rung) returns the emulation; evaluate_fn(rung, run) returns a verdict
    (a bool, or an object with `.passed`). A rung failing after a lower rung passed
    is a body-model defect, not a smaller minimum; with check_higher (default) the
    rungs above the first pass are still tested so such defects can be seen, and
    the minimum stays the first pass. A rung whose backend is not configured
    (BodyNotConfigured) is "unavailable", neither a pass nor a defect.
    """
    order = [RUNGS.index(r) for r in rungs]  # unknown rungs raise ValueError
    if order != sorted(set(order)):
        raise ValueError("rungs must be given in increasing ladder order")
    results, minimum, defects = [], None, []
    for rung in rungs:
        if minimum is not None and not check_higher:
            break
        boundary = "open_loop" if rung == "L0" else "closed_loop"
        record = ClosedLoopRecord(boundary, rung, horizon_s)
        try:
            verdict = evaluate_fn(rung, run_fn(rung))
        except BodyNotConfigured as error:
            results.append(RungResult(rung, "unavailable", record, note=str(error)))
            continue
        passed = bool(getattr(verdict, "passed", verdict))
        if passed and minimum is None:
            minimum = rung
        if not passed and minimum is not None:
            defects.append(rung)
        results.append(RungResult(rung, "pass" if passed else "fail", record, verdict))
    return LadderReport(tuple(results), minimum, tuple(defects))
