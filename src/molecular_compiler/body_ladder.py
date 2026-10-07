"""M13 body ladder L0-L5 as BodyModel adapters, and the M13-R2 evaluation record."""

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import ClassVar

import numpy as np

from molecular_compiler.body import MappedBodyAdapter
from molecular_compiler.worm_body import DEFAULT_PARAMS, WormBody

RUNGS = ("L0", "L1", "L2", "L3", "L4", "L5")
BOUNDARY_CONDITIONS = ("open_loop", "closed_loop")


class BodyNotConfigured(RuntimeError):
    """A rung needs an external backend that has not been configured."""


@dataclass(frozen=True)
class ClosedLoopRecord:
    """M13-R1/R2: every record states boundary condition, ladder rung and horizon H."""

    boundary_condition: str
    rung: str
    horizon_s: float
    results: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.boundary_condition not in BOUNDARY_CONDITIONS:
            raise ValueError(
                "record requires boundary_condition open_loop or closed_loop"
            )
        if self.rung not in RUNGS:
            raise ValueError(f"record requires a ladder rung in {RUNGS}")
        if not isinstance(self.horizon_s, int | float) or not (
            np.isfinite(self.horizon_s) and self.horizon_s > 0
        ):
            raise ValueError("record requires a positive finite horizon H in seconds")
        # L0 has no body to close a loop with; L3-L5 are closed by construction.
        # L1 and L2 may be labelled either way (simulate() labels any body closed_loop).
        if (self.rung == "L0" and self.boundary_condition != "open_loop") or (
            self.rung in ("L3", "L4", "L5") and self.boundary_condition != "closed_loop"
        ):
            raise ValueError(
                f"rung {self.rung} is inconsistent with {self.boundary_condition}"
            )


def record_from_trajectory(trajectory, rung, horizon_s, **results):
    """Build a record from a simulate() trajectory, taking its boundary condition."""
    return ClosedLoopRecord(
        trajectory.metadata.get("boundary_condition"), rung, horizon_s, results
    )


@dataclass
class NoBody:
    """L0: no feedback (immobilized)."""

    n_neurons: int
    rung: ClassVar[str] = "L0"

    def reset(self, state0):
        return np.zeros(self.n_neurons)

    def step(self, motor, dt_s):
        return np.zeros(self.n_neurons)


@dataclass
class TonicDrive:
    """L1: constant per-neuron current (pA), fitted elsewhere."""

    current_pA: np.ndarray
    rung: ClassVar[str] = "L1"

    def __post_init__(self):
        self.current_pA = np.asarray(self.current_pA, dtype=float)
        if self.current_pA.ndim != 1 or not np.all(np.isfinite(self.current_pA)):
            raise ValueError("tonic drive must be a finite 1D current vector")

    def reset(self, state0):
        return self.current_pA.copy()

    def step(self, motor, dt_s):
        return self.current_pA.copy()


class Replay:
    """L2: recorded posture/stimulus series converted to sensory current, open loop.

    series is (T, k) sampled every series_dt_s; gain_pA is (n_neurons, k). The motor
    input is ignored. Running past the recording raises unless hold_last is set.
    """

    rung = "L2"

    def __init__(self, series, gain_pA, series_dt_s, hold_last=False):
        self.series = np.asarray(series, dtype=float)
        self.gain = np.asarray(gain_pA, dtype=float)
        if (
            self.series.ndim != 2
            or self.gain.ndim != 2
            or self.gain.shape[1] != self.series.shape[1]
            or series_dt_s <= 0
            or not np.all(np.isfinite(self.series))
        ):
            raise ValueError("replay series and gain shapes are inconsistent")
        self.series_dt_s, self.hold_last, self.t = series_dt_s, hold_last, 0.0

    def _current(self):
        index = int(self.t / self.series_dt_s + 1e-9)
        if index >= len(self.series):
            if not self.hold_last:
                raise ValueError("replay ran past the end of the recording")
            index = len(self.series) - 1
        return self.gain @ self.series[index]

    def reset(self, state0):
        self.t = 0.0
        return self._current()

    def step(self, motor, dt_s):
        self.t += dt_s
        return self._current()


def _index(name, prefix):
    match = re.fullmatch(prefix + r"(\d+)", name)
    return int(match.group(1)) if match else None


def default_territories(names):
    """name -> (side, lo, hi): side +1 dorsal, -1 ventral, 0 either; lo, hi body fractions.

    Illustrative layout, head at 0: SMDD/SMDV cover the head, DB/VB neurons tile
    the rest in numerical order, DVA reads the whole body. Replace with an
    anatomical map by passing `territories` explicitly.
    """
    out = {}
    for prefix, side in (("DB", 1), ("VB", -1)):
        found = {n: _index(n, prefix) for n in names if _index(n, prefix)}
        top = max(found.values(), default=0)
        for name, k in found.items():
            out[name] = (side, 0.1 + 0.9 * (k - 1) / top, 0.1 + 0.9 * k / top)
    for name in names:
        if re.fullmatch(r"SMDD[LR]?", name):
            out[name] = (1, 0.0, 0.1)
        elif re.fullmatch(r"SMDV[LR]?", name):
            out[name] = (-1, 0.0, 0.1)
        elif name == "DVA":
            out[name] = (0, 0.0, 1.0)
    return out


class ProprioceptiveLoop:
    """L3: reduced 2D body closed with the brain.

    Motor neurons (side +-1) set dorsal or ventral muscle drive through the
    MappedBodyAdapter-style transform 0.5 * (1 + tanh((V + 40) / 10)) of voltage,
    averaged over the neurons covering each joint. Stretch-sensitive neurons
    (default DVA, SMDD*, DB*, VB*) receive sensory_gain_pA * side * curvature
    (1/mm, averaged over their territory); side 0 (DVA) reads |curvature|.
    """

    rung = "L3"
    stretch_patterns: ClassVar[tuple] = ("DVA", "SMDD*", "DB*", "VB*")
    motor_patterns: ClassVar[tuple] = ("SMDD*", "SMDV*", "DB*", "VB*")

    def __init__(
        self,
        neuron_names,
        params=DEFAULT_PARAMS,
        stretch_patterns=None,
        motor_patterns=None,
        territories=None,
        sensory_gain_pA=1.0,
        motor_gain=1.0,
    ):
        self.names = list(neuron_names)
        self.body = WormBody(params)
        self.sensory_gain_pA, self.motor_gain = sensory_gain_pA, motor_gain
        territories = territories or default_territories(self.names)
        self.n_joints = n_joints = params.n_joints

        def window(name):
            side, lo, hi = territories[name]
            a = min(int(lo * n_joints), n_joints - 1)
            return side, slice(a, max(a + 1, round(hi * n_joints)))

        def select(patterns):
            return [
                (i, *window(n))
                for i, n in enumerate(self.names)
                if n in territories and any(fnmatchcase(n, p) for p in patterns)
            ]

        self.stretch = select(stretch_patterns or self.stretch_patterns)
        self.motor = [m for m in select(motor_patterns or self.motor_patterns) if m[1]]
        if not self.motor:
            raise ValueError("no motor neuron maps to a body territory")

    def _sensory(self, kappa):
        out = np.zeros(len(self.names))
        for i, side, joints in self.stretch:
            seg = kappa[joints]
            out[i] = self.sensory_gain_pA * (
                np.abs(seg).mean() if side == 0 else side * seg.mean()
            )
        return out

    def _activation(self, voltage):
        voltage = np.asarray(voltage, dtype=float)
        if voltage.shape != (len(self.names),):
            raise ValueError("motor vector differs from neuron mapping")
        drive = np.zeros((2, self.n_joints))
        count = np.zeros((2, self.n_joints))
        for i, side, joints in self.motor:
            row = 0 if side > 0 else 1
            drive[row, joints] += 0.5 * (1 + np.tanh((voltage[i] + 40) / 10))
            count[row, joints] += 1
        return np.clip(self.motor_gain * drive / np.maximum(count, 1), 0, 1)

    def reset(self, state0):
        return self._sensory(self.body.reset(state0))

    def step(self, motor, dt_s):
        return self._sensory(self.body.step(self._activation(motor), dt_s))


@dataclass
class BodyOutput:
    """L4 two-part feedback: sensory currents (pA) and modulator-concentration targets c."""

    sensory: np.ndarray
    modulator_targets: np.ndarray


class InteroceptiveLoop(ProprioceptiveLoop):
    """L4: L3 plus an interoceptive channel.

    interoception_fn(t_s, body_state) returns the feeding/pumping state vector
    (k,); modulator targets are baseline + gain @ that vector (gain is
    (n_modulators, k)). reset/step return a BodyOutput; wrap with sensory_only to
    pass it to simulate(), which does not yet consume modulator targets.
    """

    rung = "L4"

    def __init__(self, *args, interoception_fn, gain, baseline, **kwargs):
        super().__init__(*args, **kwargs)
        self.interoception_fn = interoception_fn
        self.gain, self.baseline = np.asarray(gain, float), np.asarray(baseline, float)
        if self.gain.ndim != 2 or self.gain.shape[0] != self.baseline.shape[0]:
            raise ValueError("interoceptive gain and baseline shapes are inconsistent")
        self.t = 0.0

    def _output(self, sensory):
        intero = np.asarray(self.interoception_fn(self.t, self.body.state), float)
        if intero.shape != (self.gain.shape[1],) or not np.all(np.isfinite(intero)):
            raise ValueError("interoception vector differs from gain mapping")
        return BodyOutput(sensory, self.baseline + self.gain @ intero)

    def reset(self, state0):
        self.t = 0.0
        return self._output(super().reset(state0))

    def step(self, motor, dt_s):
        sensory = super().step(motor, dt_s)
        self.t += dt_s
        return self._output(sensory)


class sensory_only:
    """Expose only the sensory part of a BodyOutput-returning body, for simulate()."""

    def __init__(self, body):
        self.body, self.rung = body, body.rung
        self.last_modulator_targets = None

    def _take(self, out):
        self.last_modulator_targets = out.modulator_targets
        return out.sensory

    def reset(self, state0):
        return self._take(self.body.reset(state0))

    def step(self, motor, dt_s):
        return self._take(self.body.step(motor, dt_s))


class _JsonLinesBackend:
    """Talks to an external body process: one JSON object per line over stdin/stdout.

    Requests {"cmd": "reset", "state0": {...}} or {"cmd": "step", "actuators":
    [...], "dt_s": float}; each reply is {"sensors": [...]}.
    """

    def __init__(self, command):
        self.command, self.process = list(command), None

    def _call(self, request):
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("external body process closed its output")
        return np.asarray(json.loads(line)["sensors"], dtype=float)

    def reset(self, state0):
        return self._call({"cmd": "reset", "state0": state0})

    def step(self, actuators, dt_s):
        return self._call(
            {"cmd": "step", "actuators": np.asarray(actuators).tolist(), "dt_s": dt_s}
        )

    def close(self):
        if self.process is not None:
            self.process.kill()
            self.process.wait()


class ExternalProcessBody:
    """L5: BAAIWorm/Sibernetic (or any body) as an external process.

    Requires a command that exists on this machine and backend provenance
    (version, model); otherwise raises BodyNotConfigured. No backend is bundled
    and no HTTP endpoint transport exists. The command must speak the JSON-lines
    protocol of _JsonLinesBackend; mappings follow MappedBodyAdapter.
    """

    rung = "L5"

    def __init__(
        self,
        command,
        backend_provenance,
        n_neurons,
        motor_indices,
        sensory_indices,
        **gains,
    ):
        if not command:
            raise BodyNotConfigured("L5 needs an external executable; none configured")
        if shutil.which(command[0]) is None:
            raise BodyNotConfigured(f"L5 executable not found: {command[0]}")
        if not backend_provenance:
            raise BodyNotConfigured("L5 needs backend version and provenance")
        self.backend = _JsonLinesBackend(command)
        self.adapter = MappedBodyAdapter(
            self.backend,
            n_neurons,
            tuple(motor_indices),
            tuple(sensory_indices),
            backend_provenance=backend_provenance,
            **gains,
        )

    def reset(self, state0):
        return self.adapter.reset(state0)

    def step(self, motor, dt_s):
        return self.adapter.step(motor, dt_s)

    def close(self):
        self.backend.close()
