"""Track I Stage I2: minimum virtual body pipeline (spec 10.4, M13 ladder, M13-R2..R4).

For each recorded animal and horizon H the M13 ladder is climbed with
`sustain.climb_ladder`. Recordings (`atanas.Recording`, original fluorescence)
and the compiled simulator are put on one grid and one neuron set:

- Neurons: canonical names present in the simulator and in every animal.
- Time: each recording is linearly interpolated (NaN gaps included) onto
  a grid of dt_grid_s = k * dt_sim; the emulation's calcium is read out every
  k steps (value at the end of the bin) and converted to fluorescence.
- Units: both sides are dF/F0 with F0 the neuron's 10th percentile over the
  evaluated window [0, H], so a window never reads samples past H. The
  emulation's F is the fluorescence readout of the simulator, referenced to
  the rest state's F.

Part (i) of M13-R3 is `sustain.spontaneous_activity_check` against all
recorded animals with duration >= H (the animal itself included in the
reference, as the criterion is a population spread). Part (ii) needs a decoder
and per-animal latents; without them it is reported `not_evaluable`, never
as a pass or a fail. The ladder is climbed twice over the same cached runs:
once on part (i) alone ("lowest rung that sustains a worm") and, when part
(ii) is evaluable, once on both ("sustains this worm"), each with its M13-R4
body-model defects.

Declared approximations, all echoed in the report:

- L1 tonic drive: one per-neuron current vector for the whole population,
  fitted by damped diagonal-Jacobian relaxation of the population's mean
  recorded dF/F0 level against the emulation's, from the rest state.
- L2 replay: the animal's head angle (normalized by the population median
  absolute value) drives SMDD (+) and SMDV (-), and its absolute value DVA,
  with gain sensory_gain_pA. Velocity, angular velocity and pumping are
  carried but have zero default gain. Illustrative, not an anatomical map.
- L3 uses `ProprioceptiveLoop` (default illustrative territories).
- L4 feeds the animal's recorded pumping back into the modulator
  concentrations c(t) through `simulate`'s BodyOutput path. The map from
  pumping to each modulator (`l4["gain"]`, one row per compiled
  concentration) has no default, so L4 is `unavailable` until one is given.
  The baseline defaults to the compiled concentrations.
- L5 is `unavailable` unless a body factory is supplied.
- The emulation is the population model; no per-animal parameter enters it,
  so rungs other than L2 and L4 are emulated once and shared by all animals.
"""

import json
import time
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from molecular_compiler import atanas
from molecular_compiler.body_ladder import (
    RUNGS,
    BodyNotConfigured,
    InteroceptiveLoop,
    ProprioceptiveLoop,
    Replay,
)
from molecular_compiler.simulation import Stimulus, simulate
from molecular_compiler.sustain import climb_ladder, identity_check
from molecular_compiler.sustain import spontaneous_activity_check as part_i_check

OUTCOMES = (
    "sustains_this_worm",
    "sustains_a_worm",
    "sustains_a_worm_identity_not_evaluable",
    "fails",
)
NOT_EVALUABLE_REASON = (
    "no decoder: Stage I0 found no usable individual latent (spec 10.4, Revision 13)"
)
CHANNELS = ("head_angle", "abs_head_angle", "velocity", "angular_velocity", "pumping")
DEFAULT_SENSORY_MAP = {
    "head_angle": {"SMDD*": 1.0, "SMDV*": -1.0},
    "abs_head_angle": {"DVA": 1.0},
}
MIN_SAMPLES = 8
BASELINE_PERCENTILE = 10.0


class Diverged:
    """A run whose simulator raised FloatingPointError (nonfinite or unsolved)."""

    def __init__(self, message):
        self.message = message


def parse_horizons(values):
    """['60', 300, 'full'] -> [(label, seconds or None)]; None means full recording."""
    out = []
    for v in values:
        if str(v) == "full":
            out.append(("full", None))
        else:
            seconds = float(v)
            if not np.isfinite(seconds) or seconds <= 0:
                raise ValueError("horizons must be positive and finite, or 'full'")
            out.append((f"{seconds:g}", seconds))
    return out


@dataclass
class AlignedAnimal:
    animal: str
    duration_s: float
    n_grid: int
    fluorescence: np.ndarray  # [common, n_grid] on the common grid
    channels: dict  # CHANNELS name -> [n_grid], raw units
    missing_channels: tuple
    sha256: str


def _regrid(t, x, grid):
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return None
    return np.interp(grid, t[ok], x[ok])


def align_recordings(recordings, sim_names, dt_grid_s, min_neurons=3):
    """Common-neuron, common-grid view of the recordings. Returns (names, sim_index, animals)."""
    if len(set(sim_names)) != len(sim_names):
        raise ValueError("simulator neuron names must be unique")
    usable = []
    for rec in recordings:
        t = rec.t - rec.t[0]
        grid = np.arange(int(t[-1] / dt_grid_s + 1e-9) + 1) * dt_grid_s
        cols = {}
        for k, name in enumerate(rec.neurons):
            if name in sim_names:
                y = _regrid(t, rec.traces[:, k], grid)
                if y is not None:
                    cols[name] = y
        usable.append((rec, t, grid, cols))
    common = set(sim_names)
    for _, _, _, cols in usable:
        common &= set(cols)
    names = [n for n in sim_names if n in common]
    if len(names) < min_neurons:
        raise ValueError(
            f"only {len(names)} canonical neurons are shared by the simulator and "
            f"every animal; at least {min_neurons} are needed"
        )
    index = np.array([sim_names.index(n) for n in names])
    animals = []
    for rec, t, grid, cols in usable:
        channels, missing = {}, []
        raw = {
            name: _regrid(t, rec.behavior[name], grid)
            for name in atanas.BEHAVIOR
            if name in rec.behavior
        }
        for name in ("head_angle", "velocity", "angular_velocity", "pumping"):
            if raw.get(name) is None:
                channels[name] = np.zeros(len(grid))
                missing.append(name)
            else:
                channels[name] = raw[name]
        channels["abs_head_angle"] = np.abs(channels["head_angle"])
        animals.append(
            AlignedAnimal(
                rec.animal,
                float(t[-1]),
                len(grid),
                np.stack([cols[n] for n in names]),
                channels,
                tuple(missing),
                rec.sha256,
            )
        )
    return names, index, animals


def horizon_samples(animal, seconds, dt_grid_s):
    """Samples n_H in the window [0, H] of an animal; 'full' (None) uses all of it."""
    n = int(animal.duration_s / dt_grid_s + 1e-9)
    return n if seconds is None else round(seconds / dt_grid_s)


def to_dff(fluorescence, n):
    """dF/F0 over the first n samples, F0 = 10th percentile of that window."""
    f = np.asarray(fluorescence, dtype=float)[:, :n]
    f0 = np.percentile(f, BASELINE_PERCENTILE, axis=1, keepdims=True)
    if np.any(f0 <= 0):
        raise ValueError("fluorescence baseline must be positive to form dF/F0")
    return f / f0 - 1.0


def channel_scales(animals):
    """Population median absolute value per channel (1 where it is 0)."""
    out = {}
    for name in CHANNELS:
        pooled = np.concatenate([np.abs(a.channels[name]) for a in animals])
        scale = float(np.median(pooled))
        out[name] = scale if scale > 0 else 1.0
    return out


def sensory_gain(neuron_names, sensory_map, gain_pA):
    """(n_neurons, len(CHANNELS)) gain matrix from {channel: {pattern: weight}}."""
    gain = np.zeros((len(neuron_names), len(CHANNELS)))
    for channel, patterns in sensory_map.items():
        c = CHANNELS.index(channel)
        for i, name in enumerate(neuron_names):
            for pattern, weight in patterns.items():
                if fnmatchcase(name, pattern):
                    gain[i, c] += gain_pA * weight
    return gain


def replay_series(animal, scales, n_samples):
    """(T, len(CHANNELS)) normalized channel series, padded by holding the last value."""
    series = np.stack(
        [animal.channels[c][:n_samples] / scales[c] for c in CHANNELS], axis=1
    )
    return np.nan_to_num(series)


def _subsample(calcium, offset, k):
    """Rows of calcium (steps, n) whose global step index + 1 is a multiple of k."""
    first = (-offset - 1) % k
    return np.asarray(calcium)[first::k]


class ReadoutDecoder:
    """Decoder from an `encoder.Readout` on per-neuron window means (dF/F0).

    `decoder_neurons` is the neuron order the readout was fitted in. `bind`
    fixes the row order of the windows it will receive (the aligned names).
    """

    def __init__(self, readout, decoder_neurons):
        self.readout, self.decoder_neurons, self.order = readout, decoder_neurons, None

    def bind(self, names):
        missing = [n for n in self.decoder_neurons if n not in names]
        if missing:
            raise ValueError(
                f"decoder needs neurons absent from the alignment: {missing}"
            )
        self.order = [list(names).index(n) for n in self.decoder_neurons]
        return self

    def __call__(self, window):
        return self.readout.decode(np.asarray(window)[self.order].mean(axis=1))


def load_decoder(path):
    """(ReadoutDecoder, latents) from an npz with weights, bias, mean, neurons, animals, latents."""
    from molecular_compiler.encoder import Readout

    z = np.load(path, allow_pickle=False)
    readout = Readout(z["weights"], z["bias"], z["mean"])
    latents = {str(a): u for a, u in zip(z["animals"], z["latents"], strict=True)}
    return ReadoutDecoder(readout, [str(n) for n in z["neurons"]]), latents


def load_directory(directory, labels_path, manifest=None, canonical=None):
    """Recordings under `directory`: manifest-verified, or every *.h5/*.hdf5 by file stem."""
    if manifest is not None:
        return atanas.load_baseline(directory, manifest, labels_path, canonical)
    labels = atanas.read_labels(labels_path)
    files = sorted(p for p in Path(directory).iterdir() if p.suffix in (".h5", ".hdf5"))
    if not files:
        raise ValueError(f"no .h5 or .hdf5 recordings under {directory}")
    missing = [p.stem for p in files if p.stem not in labels]
    if missing:
        raise ValueError(f"missing animal labels: {missing}")
    return [atanas.load_recording(p, labels[p.stem], p.stem, canonical) for p in files]


@dataclass
class I2Verdict:
    outcome: str
    part_i: dict
    part_ii: dict
    note: str = ""

    @property
    def passed(self):
        """For the 'a worm' climb: part (i) alone."""
        return self.part_i["passed"]

    @property
    def passed_this(self):
        return self.part_ii["status"] == "evaluated" and self.part_ii["passed"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class MinimumBody:
    """Runs and caches the emulations; see module docstring."""

    def __init__(
        self,
        sim,
        neuron_names,
        recordings,
        fluorescence,
        warmup_s=20.0,
        dt_grid_s=None,
        sensory_gain_pA=1.0,
        sensory_map=None,
        l1_iterations=5,
        l1_step_pA=5.0,
        l1_fit_s=60.0,
        l4=None,
        l5=None,
        min_neurons=3,
        chunk_s=60.0,
    ):
        self.sim, self.fluorescence = sim, fluorescence
        self.names = list(neuron_names)
        if len(self.names) != sim.n_neurons:
            raise ValueError("one canonical name per simulator neuron is required")
        dt = sim.resolution.dt_s
        if dt_grid_s is None:
            steps = [np.median(np.diff(r.t)) for r in recordings]
            dt_grid_s = max(1, round(float(np.median(steps)) / dt)) * dt
        self.k = round(dt_grid_s / dt)
        if self.k < 1 or not np.isclose(self.k * dt, dt_grid_s, rtol=1e-6):
            raise ValueError("dt_grid_s must be a multiple of the simulator timestep")
        self.dt_grid_s = self.k * dt
        self.common, self.index, self.animals = align_recordings(
            recordings, self.names, self.dt_grid_s, min_neurons
        )
        self.scales = channel_scales(self.animals)
        self.gain = sensory_gain(
            self.names, sensory_map or DEFAULT_SENSORY_MAP, sensory_gain_pA
        )
        self.sensory_gain_pA = sensory_gain_pA
        self.l1_cfg = (l1_iterations, l1_step_pA, l1_fit_s)
        self.l4, self.l5, self.chunk_s = l4, l5, chunk_s
        self.rest = simulate(sim, Stimulus(), warmup_s).final_state
        self.rest_f = np.asarray(fluorescence(np.asarray(self.rest.calcium)))[
            self.index
        ]
        self._tonic = None
        self.l1_fit = None
        self.cache, self.need = {}, {}

    # --- emulation ---------------------------------------------------------

    def _dff(self, calcium):
        """(common, T) dF/F0-style trace of subsampled calcium (T, n): F / F_rest - 1."""
        f = np.asarray(self.fluorescence(np.asarray(calcium)))[:, self.index].T
        return f / self.rest_f[:, None] - 1.0

    def _open_loop(self, current_fn, n_samples):
        steps_total, state, parts, done = n_samples * self.k, self.rest, [], 0
        chunk = max(self.k, round(self.chunk_s / self.sim.resolution.dt_s))
        while done < steps_total:
            n = min(chunk, steps_total - done)
            currents = current_fn(done, n)
            traj = simulate(
                self.sim,
                Stimulus(currents=currents),
                n * self.sim.resolution.dt_s,
                state0=state,
            )
            state = traj.final_state
            parts.append(_subsample(traj.calcium, done, self.k))
            done += n
        return self._dff(np.concatenate(parts))

    def _closed_loop(self, body, n_samples):
        traj = simulate(
            self.sim,
            Stimulus(),
            n_samples * self.k * self.sim.resolution.dt_s,
            body=body,
            state0=self.rest,
        )
        return self._dff(_subsample(traj.calcium, 0, self.k))

    def _constant(self, current):
        n = self.sim.n_neurons

        def fn(done, count):
            return jnp.broadcast_to(jnp.asarray(current), (count, n))

        return fn

    def tonic_drive(self, targets):
        """L1 current (pA, per simulator neuron), fitted once; cached.

        Damped relaxation: I_i += step_pA * clip(target_i - level_i, -1, 1),
        where level_i is the emulation's mean dF/F0 over the second half of a
        fit_s run under constant I from rest.
        """
        if self._tonic is not None:
            return self._tonic
        iterations, step_pA, fit_s = self.l1_cfg
        n = max(MIN_SAMPLES, int(fit_s / self.dt_grid_s))
        current = np.zeros(self.sim.n_neurons)

        def level(current):
            dff = self._open_loop(self._constant(current), n)
            return dff[:, n // 2 :].mean(axis=1)

        before = float(np.sqrt(np.mean((targets - level(current)) ** 2)))
        for _ in range(iterations):
            error = targets - level(current)
            current[self.index] += step_pA * np.clip(error, -1.0, 1.0)
        after = float(np.sqrt(np.mean((targets - level(current)) ** 2)))
        self.l1_fit = {
            "iterations": iterations,
            "step_pA": step_pA,
            "fit_duration_s": n * self.dt_grid_s,
            "rms_error_before": before,
            "rms_error_after": after,
            "current_pA_mean_abs": float(np.abs(current).mean()),
            "current_pA_max_abs": float(np.abs(current).max()),
            "n_fitted_neurons": len(self.index),
        }
        self._tonic = current
        return current

    def emulate(self, rung, animal, n_samples, targets):
        """dF/F0 (common, n_samples) for rung and animal; one run per key, at `need`."""
        shared = rung not in ("L2", "L4", "L5")
        key = (rung, None if shared else animal.animal)
        if key not in self.cache:
            self.cache[key] = self._run(
                rung, animal, max(n_samples, self.need.get(key, 0)), targets
            )
        hit = self.cache[key]
        if not isinstance(hit, Diverged) and hit.shape[1] < n_samples:
            raise ValueError("cached emulation is shorter than the requested horizon")
        return hit if isinstance(hit, Diverged) else hit[:, :n_samples]

    def _run(self, rung, animal, n, targets):
        n_neurons, dt = self.sim.n_neurons, self.sim.resolution.dt_s
        try:
            if rung == "L0":
                return self._open_loop(self._constant(np.zeros(n_neurons)), n)
            if rung == "L1":
                return self._open_loop(self._constant(self.tonic_drive(targets)), n)
            if rung == "L2":
                series = replay_series(animal, self.scales, animal.n_grid)
                replay = Replay(series, self.gain, self.dt_grid_s, hold_last=True)

                def fn(done, count):
                    out = np.empty((count, n_neurons))
                    replay.t = done * dt
                    cur = replay._current()
                    for i in range(count):
                        out[i] = cur
                        cur = replay.step(None, dt)
                    return out

                return self._open_loop(fn, n)
            if rung == "L3":
                loop = self._loop(ProprioceptiveLoop)
                return self._closed_loop(loop, n)
            if rung == "L4":
                if self.l4 is None:
                    raise BodyNotConfigured(
                        "L4 needs a pumping-to-modulator map (`l4['gain']`); "
                        "there is no default"
                    )
                baseline = self.l4.get("baseline")
                if baseline is None:
                    baseline = np.asarray(self.sim.neuromod["concentrations"])
                pump = np.nan_to_num(
                    animal.channels["pumping"] / self.scales["pumping"]
                )

                def intero(t, state):
                    return np.array([pump[min(int(t / self.dt_grid_s), len(pump) - 1)]])

                loop = self._loop(
                    InteroceptiveLoop,
                    interoception_fn=intero,
                    gain=self.l4["gain"],
                    baseline=baseline,
                )
                return self._closed_loop(loop, n)
            if rung == "L5":
                if self.l5 is None:
                    raise BodyNotConfigured("L5 external body not configured")
                return self._closed_loop(self.l5(animal.animal), n)
        except FloatingPointError as error:
            return Diverged(str(error))
        raise ValueError(f"unknown rung {rung}")

    def _loop(self, cls, **kwargs):
        try:
            return cls(self.names, sensory_gain_pA=self.sensory_gain_pA, **kwargs)
        except ValueError as error:
            raise BodyNotConfigured(
                f"{cls.rung} body cannot be mapped: {error}"
            ) from error


def _part_i_dict(v):
    return {
        "passed": bool(v.passed),
        "distances": v.distances,
        "limits": v.limits,
        "n_silent": v.n_silent,
        "n_saturated": v.n_saturated,
        "max_silent_in_data": v.max_silent_in_data,
        "max_saturated_in_data": v.max_saturated_in_data,
    }


def _record(result, climb_this):
    rec = result.record
    out = {
        "boundary_condition": rec.boundary_condition,
        "rung": rec.rung,
        "horizon_s": rec.horizon_s,
        "status": result.status,
        "note": result.note,
    }
    if result.verdict is not None:
        out.update(
            outcome=result.verdict.outcome,
            part_i=result.verdict.part_i,
            part_ii=result.verdict.part_ii,
        )
        if result.verdict.note:
            out["note"] = result.verdict.note
    if climb_this is None:
        out["status_this_worm"] = "not_evaluable"
    else:
        out["status_this_worm"] = climb_this[rec.rung]
    return out


def track_i2(
    sim,
    neuron_names,
    recordings,
    fluorescence,
    horizons=(60, 300, "full"),
    rungs=RUNGS,
    decoder=None,
    latents=None,
    window_s=30.0,
    quantile=1.0,
    provenance=None,
    log=lambda message: None,
    **options,
):
    """Stage I2 report (restricted). `options` go to `MinimumBody`.

    fluorescence maps simulator calcium (any leading shape, last axis neurons)
    to fluorescence. decoder: a callable window (common, w) -> latent, or a
    builder names -> callable (see `decoder_from_readout`); latents: {animal id:
    u}. Without both, part (ii) is `not_evaluable`.
    """
    start = time.perf_counter()
    horizons = parse_horizons(horizons)
    body = MinimumBody(sim, neuron_names, recordings, fluorescence, **options)
    dt, animals = body.dt_grid_s, body.animals
    if len(animals) < 3:
        raise ValueError("across-animal spread needs at least three recorded animals")
    if decoder is not None and hasattr(decoder, "bind"):
        decoder = decoder.bind(body.common)
    evaluable = decoder is not None and latents is not None

    # Samples per (animal, horizon); None when the horizon cannot be evaluated.
    plan = {}
    for a in animals:
        for label, seconds in horizons:
            n = horizon_samples(a, seconds, dt)
            ok = n >= MIN_SAMPLES and n <= horizon_samples(a, None, dt)
            plan[(a.animal, label)] = n if ok else None
    # One run per cache key at its longest horizon, sliced for shorter ones.
    for rung in rungs:
        for a in animals:
            longest = max((plan[(a.animal, label)] or 0) for label, _ in horizons)
            shared = rung not in ("L2", "L4", "L5")
            key = (rung, None if shared else a.animal)
            body.need[key] = max(body.need.get(key, 0), longest)
    common_n = min(horizon_samples(a, None, dt) for a in animals)
    targets = np.mean(
        [to_dff(a.fluorescence, common_n).mean(axis=1) for a in animals], axis=0
    )

    references, report_animals = {}, {}
    for a in animals:
        report_animals[a.animal] = entry = {
            "duration_s": a.duration_s,
            "n_grid_samples": a.n_grid,
            "missing_behavior_channels": list(a.missing_channels),
            "sha256": a.sha256,
            "horizons": {},
        }
        for label, seconds in horizons:
            n = plan[(a.animal, label)]
            if n is None:
                entry["horizons"][label] = {
                    "status": "not_evaluable",
                    "reason": "horizon shorter than the minimum sample count "
                    f"({MIN_SAMPLES}) or longer than the recording",
                }
                continue
            if label not in references:
                references[label] = [
                    to_dff(b.fluorescence, horizon_samples(b, seconds, dt))
                    for b in animals
                    if plan[(b.animal, label)] is not None
                ]
            pool = references[label]
            if len(pool) < 3:
                entry["horizons"][label] = {
                    "status": "not_evaluable",
                    "reason": "fewer than three recorded animals cover this horizon",
                }
                continue
            horizon_s = n * dt
            log(f"{a.animal} H={label}")
            verdicts = {}

            def run_fn(rung, a=a, n=n):
                return body.emulate(rung, a, n, targets)

            def evaluate(rung, run, a=a, n=n, pool=pool, verdicts=verdicts):
                verdicts[rung] = _evaluate(
                    run,
                    pool,
                    a,
                    n,
                    dt,
                    decoder if evaluable else None,
                    latents,
                    window_s,
                    quantile,
                )
                return verdicts[rung]

            climb = climb_ladder(list(rungs), run_fn, evaluate, horizon_s)
            this = this_status = None
            if any(v.part_ii["status"] == "evaluated" for v in verdicts.values()):
                this = climb_ladder(
                    list(rungs),
                    run_fn,
                    lambda rung, run, v=verdicts: v[rung].passed_this,
                    horizon_s,
                )
                this_status = {r.rung: r.status for r in this.results}
            entry["horizons"][label] = {
                "status": "evaluated",
                "horizon_s": horizon_s,
                "records": [_record(r, this_status) for r in climb.results],
                "lowest_rung_sustaining_a_worm": climb.minimum_rung,
                "body_model_defects_a_worm": list(climb.body_model_defects),
                "lowest_rung_sustaining_this_worm": (
                    this.minimum_rung if this is not None else "not_evaluable"
                ),
                "body_model_defects_this_worm": (
                    list(this.body_model_defects)
                    if this is not None
                    else "not_evaluable"
                ),
                "n_reference_animals": len(pool),
            }
    summary = _summary(report_animals, horizons, evaluable)
    return _jsonable(
        {
            "stage": "I2",
            "spec": "10.4 Stage I2; M13-R2..R4",
            "data_tier": "restricted",
            "part_ii": {
                "status": "evaluated" if evaluable else "not_evaluable",
                "reason": None if evaluable else NOT_EVALUABLE_REASON,
            },
            "verdict_vocabulary": list(OUTCOMES),
            "config": {
                "rungs": list(rungs),
                "horizons": [label for label, _ in horizons],
                "dt_grid_s": dt,
                "sim_dt_s": sim.resolution.dt_s,
                "n_common_neurons": len(body.common),
                "common_neurons": body.common,
                "n_animals": len(animals),
                "window_s": window_s,
                "quantile": quantile,
                "warmup_s": options.get("warmup_s", 20.0),
                "sensory_gain_pA": body.sensory_gain_pA,
                "channel_scales": body.scales,
                "baseline": f"F0 = {BASELINE_PERCENTILE:g}th percentile over [0, H]",
                "trace_source": atanas.TRACE_KEY,
                **(provenance or {}),
            },
            "l1_fit": body.l1_fit,
            "declared_approximations": DECLARED_APPROXIMATIONS,
            "animals": report_animals,
            "summary": summary,
            "elapsed_s": time.perf_counter() - start,
        }
    )


DECLARED_APPROXIMATIONS = [
    (
        "L1 tonic drive: one per-neuron current vector fitted once on the population "
        "by damped diagonal-Jacobian relaxation of mean recorded dF/F0 against the "
        "emulation's, from the rest state; fit residual in l1_fit."
    ),
    (
        "L2 replay: head angle (population-median normalized) drives SMDD(+), SMDV(-); "
        "its absolute value drives DVA; other behavior channels have zero default gain. "
        "Illustrative map, not anatomical."
    ),
    "L3/L4 use the reduced 2D body with its illustrative territories and parameters.",
    (
        "L4: recorded pumping drives the modulator concentrations c(t) through a "
        "supplied linear map (no default); c relaxes toward the targets with "
        "simulate's modulator_tau_s."
    ),
    (
        "dF/F0 uses F0 = 10th percentile over [0, H] for recordings; the emulation is "
        "referenced to its rest-state fluorescence. Recordings are linearly interpolated "
        "onto the grid; the emulation is read out at the end of each grid bin."
    ),
    (
        "The emulation is the population model: no per-animal parameter enters it, "
        "so L0, L1 and L3 emulations are shared by all animals."
    ),
    "Part (i) reference includes the animal itself and every animal covering H.",
    (
        "'full' horizon: each animal's whole recording; part (i) compares on the "
        "shortest common length."
    ),
]


def _evaluate(run, pool, animal, n, dt, decoder, latents, window_s, quantile):
    if isinstance(run, Diverged):
        part_i = {
            "passed": False,
            "diverged": True,
            "message": run.message,
        }
        return I2Verdict(
            "fails",
            part_i,
            {"status": "not_evaluable", "reason": "emulation diverged"},
            "run diverged: " + run.message,
        )
    dff = np.asarray(run)[:, :n]
    spontaneous = part_i_check(dff, pool, quantile=quantile)
    part_i = _part_i_dict(spontaneous)
    if decoder is None:
        part_ii = {"status": "not_evaluable", "reason": NOT_EVALUABLE_REASON}
    else:
        others = [u for k, u in latents.items() if k != animal.animal]
        if animal.animal not in latents or not others:
            part_ii = {
                "status": "not_evaluable",
                "reason": "no latent for this animal or no other animal's latent",
            }
        else:
            window = min(window_s, n * dt)
            checkpoints = sorted(
                {round(window * i, 9) for i in range(1, int(n * dt / window) + 1)}
                | {round(n * dt, 9)}
            )
            identity = identity_check(
                dff,
                dt,
                decoder,
                latents[animal.animal],
                np.array(others),
                checkpoints,
                window,
            )
            part_ii = {
                "status": "evaluated",
                "passed": identity.passed,
                "checkpoints_s": list(identity.checkpoints_s),
                "margins": list(identity.margins),
            }
    if not part_i["passed"]:
        outcome = "fails"
    elif part_ii["status"] != "evaluated":
        outcome = "sustains_a_worm_identity_not_evaluable"
    else:
        outcome = "sustains_this_worm" if part_ii["passed"] else "sustains_a_worm"
    return I2Verdict(outcome, part_i, part_ii)


def _summary(animals, horizons, evaluable):
    out = {}
    for label, _ in horizons:
        counts = {}
        for entry in animals.values():
            h = entry["horizons"][label]
            key = (
                h["lowest_rung_sustaining_a_worm"] or "none"
                if h["status"] == "evaluated"
                else "not_evaluable"
            )
            counts[key] = counts.get(key, 0) + 1
        out[label] = {
            "lowest_rung_sustaining_a_worm": counts,
            "identity": "evaluated" if evaluable else "not_evaluable",
        }
    return out


def build_phase1_simulator(trained, project, variant=None):
    """Compiled simulator, names and fluorescence readout from a Phase 1 fold result.

    `variant`, if given, must equal the fold's variant. Reads restricted
    project data; used by the CLI only.
    """
    import jax

    from molecular_compiler import phase1_worm as P
    from molecular_compiler.compiler import compile

    params, half_saturation, fold = P.load_trained(trained)
    fold_variant = fold.get("variant", "base")
    if variant is not None and variant != fold_variant:
        raise ValueError(f"fold variant is {fold_variant}, not {variant}")
    options = P.VARIANTS[fold_variant]
    chloride = options.get("chloride_mM", P.CHLORIDE_MM)
    hill = options.get("hill", P.HILL)
    with jax.enable_x64(True):
        graph, manifest, _atlas, library, rules = P.setup(
            project, fold["parameter_budget"], chloride
        )
        sim = compile(
            graph,
            rules.with_params(params["rules"]),
            library,
            resolution=P.policy(chloride),
        )

    def fluorescence(calcium):
        power = np.maximum(calcium, 0) ** hill
        return P.F_BASAL + power / (half_saturation**hill + power)

    meta = {
        "trained": str(trained),
        "variant": fold_variant,
        "fold": fold.get("fold"),
        "split": fold.get("split"),
        "half_saturation": half_saturation,
        "hill": hill,
        "chloride_mM": chloride,
    }
    return sim, list(manifest["neuron_names"]), fluorescence, meta


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
