import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler import Stimulus, compile, simulate
from molecular_compiler.body import BodyModel
from molecular_compiler.body_ladder import (
    BodyNotConfigured,
    BodyOutput,
    ClosedLoopRecord,
    ExternalProcessBody,
    InteroceptiveLoop,
    NoBody,
    ProprioceptiveLoop,
    Replay,
    TonicDrive,
    record_from_trajectory,
    sensory_only,
)
from molecular_compiler.sustain import (
    FAILS,
    SUSTAINS_A,
    SUSTAINS_THIS,
    climb_ladder,
    identity_check,
    spontaneous_activity_check,
    sustain,
)
from molecular_compiler.worm_body import (
    BodyParams,
    WormBody,
    center_of_mass,
    forward_axis,
    initial_state,
    step,
)

P = BodyParams()
J = jnp.arange(P.n_joints)


def wave(t, direction):
    phase = 2 * jnp.pi * (1.5 * J / P.n_joints - direction * t)
    dorsal = 0.5 * (1 + jnp.sin(phase))
    return jnp.stack([dorsal, 1 - dorsal])


def rollout(direction, steps=150, dt=0.02):
    state = initial_state(P)
    for i in range(steps):
        state = step(state, wave(i * dt, direction), dt, P)
    return state


def test_M13_body_zero_activation_is_stationary():
    state = initial_state(P)
    for _ in range(20):
        state = step(state, jnp.zeros((2, P.n_joints)), 0.05, P)
    assert np.allclose(state.position, 0) and float(state.heading) == 0
    assert np.allclose(state.kappa, 0)


def test_M13_body_wave_propels_opposite_to_wave_direction():
    # A wave travelling head to tail moves the worm forward (toward the head).
    for direction in (1, -1):
        state = rollout(direction)
        displacement = center_of_mass(state, P) - center_of_mass(initial_state(P), P)
        forward = float(displacement @ forward_axis(initial_state(P)))
        assert direction * forward > 0.1 * P.length_mm


def test_M13_body_gradient_matches_finite_difference():
    def loss(gain):
        state = initial_state(P)
        for i in range(6):
            state = step(state, gain * wave(i * 0.02, 1), 0.02, P)
        return jnp.sum(center_of_mass(state, P) ** 2) + jnp.sum(state.kappa**2)

    g = jax.grad(loss)(0.7)
    eps = 1e-6
    fd = (loss(0.7 + eps) - loss(0.7 - eps)) / (2 * eps)
    assert np.isfinite(g) and np.isclose(g, fd, rtol=1e-4)


def test_M13_body_wrapper_implements_protocol():
    body = WormBody()
    assert isinstance(body, BodyModel)
    assert body.reset({}).shape == (P.n_joints,)
    curv = body.step(np.zeros(2 * P.n_joints), 0.02)
    assert curv.shape == (P.n_joints,)
    with pytest.raises(ValueError):
        body.step(np.zeros(3), 0.02)


NAMES = ["DVA", "SMDDL", "SMDVL", "DB1", "DB2", "VB1", "VB2", "AVAL"]


def test_M13_L0_L1_L2_rungs():
    assert (NoBody(3).rung, TonicDrive([1.0]).rung) == ("L0", "L1")
    assert not NoBody(3).reset({}).any() and isinstance(NoBody(3), BodyModel)
    tonic = TonicDrive([1.0, -2.0, 0.0])
    assert np.array_equal(tonic.step(np.zeros(3), 0.1), [1.0, -2.0, 0.0])
    replay = Replay([[0.0], [1.0], [2.0]], [[10.0], [0.0]], 0.1)
    assert replay.rung == "L2" and isinstance(replay, BodyModel)
    first = replay.reset({})
    # Open loop: motor output is ignored.
    a = replay.step(np.zeros(2), 0.1).copy()
    replay.reset({})
    b = replay.step(np.full(2, 99.0), 0.1)
    assert first[0] == 0 and np.array_equal(a, b) and a[0] == 10
    replay.step(np.zeros(2), 0.1)
    with pytest.raises(ValueError, match="past the end"):
        replay.step(np.zeros(2), 0.1)


def test_M13_L3_proprioceptive_loop_maps_curvature_and_motors():
    loop = ProprioceptiveLoop(NAMES, sensory_gain_pA=1.0)
    assert isinstance(loop, BodyModel) and loop.rung == "L3"
    assert loop.reset({}).shape == (len(NAMES),)
    # Dorsal drive (DB* depolarized, VB* hyperpolarized) bends dorsally.
    voltage = np.array([-70, -70, -70, 0, 0, -90, -90, -70.0])
    for _ in range(30):
        out = loop.step(voltage, 0.02)
    assert out[NAMES.index("DB1")] > 0 and out[NAMES.index("VB1")] < 0
    assert out[NAMES.index("DVA")] > 0  # |curvature|
    assert out[NAMES.index("AVAL")] == 0  # not stretch-sensitive
    assert np.isfinite(out).all()
    with pytest.raises(ValueError):
        loop.step(np.zeros(3), 0.02)
    with pytest.raises(ValueError, match="no motor"):
        ProprioceptiveLoop(["DVA", "AVAL"])


def test_M13_L3_runs_closed_loop_in_simulate(system):
    sim = compile(*system)
    loop = ProprioceptiveLoop(["DVA", "DB1", "VB1", "AVAL"], sensory_gain_pA=1.0)
    trajectory = simulate(sim, Stimulus(), 0.003, body=loop)
    assert trajectory.metadata["boundary_condition"] == "closed_loop"
    assert np.all(np.isfinite(trajectory.voltage))
    assert record_from_trajectory(trajectory, "L3", 0.003).rung == "L3"


def test_M13_L4_returns_sensory_and_modulator_targets():
    loop = InteroceptiveLoop(
        NAMES,
        interoception_fn=lambda t, state: np.array([min(t, 1.0), 1.0]),
        gain=[[2.0, 0.0], [0.0, 3.0]],
        baseline=[0.1, 0.2],
    )
    assert isinstance(loop, BodyModel) and loop.rung == "L4"
    out = loop.reset({})
    assert isinstance(out, BodyOutput) and out.sensory.shape == (len(NAMES),)
    assert np.allclose(out.modulator_targets, [0.1, 3.2])
    out = loop.step(np.full(len(NAMES), -50.0), 0.5)
    assert np.allclose(out.modulator_targets, [0.1 + 2 * 0.5, 3.2])
    wrapped = sensory_only(loop)
    assert wrapped.reset({}).shape == (len(NAMES),)
    assert wrapped.last_modulator_targets.shape == (2,)
    with pytest.raises(ValueError):
        InteroceptiveLoop(
            NAMES, interoception_fn=None, gain=[[1.0]], baseline=[0.0, 0.0]
        )


def _l4(system, feeding, tau_s=0.002):
    names = ["DVA", "DB1", "VB1", "AVAL"]
    return InteroceptiveLoop(
        names,
        sensory_gain_pA=1.0,
        interoception_fn=lambda t, state: np.array([feeding(t)]),
        gain=[[5.0], [5.0]],
        baseline=[0.0, 0.0],
    )


def test_M4_M13_L4_body_drives_modulator_concentrations(system):
    from molecular_compiler.compiler import ModulatoryState

    graph, rules, kinetics = system
    sim = compile(graph, rules, kinetics, state=ModulatoryState((0.0, 0.0)))
    stimulus, duration = Stimulus(), 0.006
    l3 = simulate(
        sim,
        stimulus,
        duration,
        body=ProprioceptiveLoop(["DVA", "DB1", "VB1", "AVAL"], sensory_gain_pA=1.0),
    )
    fed = _l4(system, lambda t: 0.0)
    flat = simulate(sim, stimulus, duration, body=fed, modulator_tau_s=0.002)
    # Targets equal the compiled constants: identical to the L3 body.
    np.testing.assert_array_equal(flat.signaling, l3.signaling)
    switch = lambda t: 0.0 if t < 0.003 else 1.0
    feeding = simulate(
        sim, stimulus, duration, body=_l4(system, switch), modulator_tau_s=0.002
    )
    assert np.all(np.isfinite(feeding.signaling))
    np.testing.assert_array_equal(feeding.signaling[:6], l3.signaling[:6])
    assert not np.allclose(feeding.signaling[-1], l3.signaling[-1])
    with pytest.raises(ValueError, match="modulator targets"):
        simulate(compile(graph, rules, kinetics), stimulus, duration, body=fed)


def test_M13_L4_without_body_output_leaves_concentrations_constant(system):
    sim = compile(*system)
    loop = ProprioceptiveLoop(["DVA", "DB1", "VB1", "AVAL"], sensory_gain_pA=1.0)
    a = simulate(sim, Stimulus(), 0.003, body=loop)
    b = simulate(sim, Stimulus(), 0.003, body=sensory_only(_l4(system, lambda t: 1.0)))
    np.testing.assert_array_equal(a.signaling, b.signaling)


def test_M13_L5_requires_configured_backend():
    args = {"n_neurons": 3, "motor_indices": (0,), "sensory_indices": (2,)}
    with pytest.raises(BodyNotConfigured, match="executable"):
        ExternalProcessBody(None, {"v": "1"}, **args)
    with pytest.raises(BodyNotConfigured, match="not found"):
        ExternalProcessBody(["/nonexistent/baaiworm"], {"v": "1"}, **args)
    with pytest.raises(BodyNotConfigured, match="provenance"):
        ExternalProcessBody([sys.executable], None, **args)


def test_M13_L5_json_lines_wire_protocol(tmp_path):
    # Protocol test double, not a worm: echoes the mean actuator as the sensor.
    script = tmp_path / "echo_body.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            for line in sys.stdin:
                req = json.loads(line)
                a = req.get("actuators", [0.0])
                print(json.dumps({"sensors": [sum(a) / len(a)]}), flush=True)
            """
        )
    )
    body = ExternalProcessBody(
        [sys.executable, str(script)],
        {"kind": "protocol-test", "version": "0"},
        3,
        (0,),
        (2,),
    )
    try:
        assert isinstance(body, BodyModel) and body.rung == "L5"
        assert body.reset({}).shape == (3,)
        out = body.step(np.array([-40.0, 0.0, 0.0]), 0.01)
        assert out[2] == 0.0 and out.shape == (3,)
    finally:
        body.close()


def test_M13_R2_record_requires_boundary_rung_and_horizon():
    ClosedLoopRecord("closed_loop", "L3", 60.0)
    ClosedLoopRecord("open_loop", "L0", 60.0)
    for bad in [
        (None, "L3", 60.0),
        ("closed_loop", None, 60.0),
        ("closed_loop", "L3", None),
        ("closed_loop", "L3", 0.0),
        ("closed_loop", "L3", float("inf")),
        ("closed_loop", "L9", 60.0),
        ("closed_loop", "L0", 60.0),
        ("open_loop", "L3", 60.0),
    ]:
        with pytest.raises(ValueError):
            ClosedLoopRecord(*bad)


# ---- sustain ----

N, T = 6, 256
LOAD = np.random.default_rng(3).normal(size=(N, 2))
OFFSET = np.random.default_rng(4).normal(size=(14, N)) * 5


def animal(offset, seed, noise=0.3):
    r = np.random.default_rng(seed)
    t = np.arange(T)
    latent = np.stack(
        [np.sin(0.15 * t + r.uniform(0, 6)), np.sin(0.05 * t + r.uniform(0, 6))]
    )
    return LOAD @ latent + noise * r.normal(size=(N, T)) + offset[:, None]


POP = [animal(OFFSET[k], 10 + k) for k in range(1, 13)]
OTHERS = OFFSET[1:13]


def mean_decoder(window):
    return window.mean(axis=1)


def test_M13_R3_part_i_population_spread():
    good = spontaneous_activity_check(animal(OFFSET[0], 51), POP)
    assert good.passed and good.n_silent == 0
    # A quiet fixed point is silent and has no correlation structure.
    quiet = spontaneous_activity_check(np.full((N, T), -70.0), POP)
    assert not quiet.passed and quiet.n_silent == N
    # Runaway is flagged as saturated.
    runaway = spontaneous_activity_check(animal(OFFSET[0], 51) * 1e3, POP)
    assert not runaway.passed and runaway.n_saturated > 0
    # Pinned at a bound counts as saturated when bounds are given.
    pinned = animal(OFFSET[0], 51)
    pinned[0] = 40.0
    pinned = spontaneous_activity_check(pinned, POP, bounds=(-90.0, 40.0))
    assert pinned.n_saturated >= 1
    with pytest.raises(ValueError):
        spontaneous_activity_check(POP[0], POP[:2])


def test_M13_R3_quantile_band_tightens_the_rule():
    scrambled = animal(OFFSET[0], 51)[np.random.default_rng(0).permutation(N)]
    strict = spontaneous_activity_check(scrambled, POP, quantile=0.5)
    loose = spontaneous_activity_check(scrambled, POP, quantile=1.0)
    assert strict.limits["psd_distance"] <= loose.limits["psd_distance"]
    assert not strict.passed  # scrambled neuron identity breaks correlation structure


def test_M13_R3_part_ii_identity_at_every_checkpoint():
    cps = [0, 100, 255]
    ok = identity_check(
        animal(OFFSET[0], 51), 1.0, mean_decoder, OFFSET[0], OTHERS, cps, 64
    )
    assert ok.passed and min(ok.margins) > 0
    # Drift toward another animal in the second half fails at the late checkpoint.
    drifted = animal(OFFSET[0], 51)
    drifted[:, 128:] += (OFFSET[1] - OFFSET[0])[:, None]
    bad = identity_check(drifted, 1.0, mean_decoder, OFFSET[0], OTHERS, cps, 64)
    assert not bad.passed and bad.margins[0] > 0 and bad.margins[-1] < 0
    with pytest.raises(ValueError):
        identity_check(drifted, 1.0, mean_decoder, OFFSET[0], OTHERS, [], 64)


def test_M13_R3_verdict_distinguishes_three_outcomes():
    kw = {"dt_s": 1.0, "checkpoints_s": [0, 128, 255], "window_s": 64}
    args = (POP, mean_decoder, OFFSET[0], OTHERS)
    this = sustain(animal(OFFSET[0], 51), *args, **kw)
    assert this.outcome == SUSTAINS_THIS and this.passed
    # A typical worm that holds another animal's durable state.
    other = sustain(animal(OFFSET[1], 51), *args, **kw)
    assert other.outcome == SUSTAINS_A and not other.passed
    quiet = sustain(np.full((N, T), -70.0), *args, **kw)
    assert quiet.outcome == FAILS


def test_M13_R4_climb_ladder_records_up_to_pass_and_flags_defects():
    def run(rung):
        if rung == "L5":
            raise BodyNotConfigured("no backend")
        return rung

    outcomes = {"L2": True, "L4": True}
    report = climb_ladder(
        ["L0", "L1", "L2", "L3", "L4", "L5"],
        run,
        lambda r, x: outcomes.get(r, False),
        60.0,
    )
    assert report.minimum_rung == "L2"
    assert report.body_model_defects == ("L3",)
    assert [r.status for r in report.results] == [
        "fail",
        "fail",
        "pass",
        "fail",
        "pass",
        "unavailable",
    ]
    assert all(r.record.horizon_s == 60.0 for r in report.results)
    assert report.results[0].record.boundary_condition == "open_loop"
    stopped = climb_ladder(
        ["L0", "L1", "L2", "L3"], run, lambda r, x: r == "L1", 60.0, check_higher=False
    )
    assert [r.rung for r in stopped.results] == ["L0", "L1"]
    assert stopped.minimum_rung == "L1" and not stopped.body_model_defects
    assert (
        climb_ladder(["L0", "L1"], run, lambda r, x: False, 60.0).minimum_rung is None
    )
    with pytest.raises(ValueError):
        climb_ladder(["L1", "L0"], run, lambda r, x: False, 60.0)
