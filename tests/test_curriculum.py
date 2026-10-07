import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler.curriculum import (
    STAGES,
    CurriculumData,
    ExitCondition,
    calibrate_ensemble,
    coverage,
    coverage_check,
    nominal_coverage,
    run_curriculum,
    train_ensemble,
)
from molecular_compiler.observation import Recording
from molecular_compiler.training import initialize_state_encoder, make_windows
from molecular_compiler.workflow import fit_animal_nuisance

N = 3


def _rollout(params, x0, steps):
    def step(x, _):
        x = 0.9 * x + 0.1 * jnp.tanh(params["m4"]["w"] @ x)
        return x, params["m9"]["s"] * x

    return jax.lax.scan(step, x0, None, length=steps)[1].T


def _true_params():
    w = jnp.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, -1.0, 0.0]])
    return {"m4": {"w": w}, "m9": {"s": jnp.ones(N)}, "m5": {"k": jnp.array(2.0)}}


def _init(seed):
    rng = np.random.default_rng(seed)
    return {
        "m4": {"w": jnp.asarray(rng.normal(size=(N, N)) * 0.1)},
        "m9": {"s": jnp.ones(N)},
        "m5": {"k": jnp.array(1.0 + 0.1 * seed)},
        "encoder": initialize_state_encoder(seed, N, N),
        "neuromodulation": {"b": jnp.zeros(N)},
        "species_adapters": {"a": jnp.zeros(N)},
        "peptidergic_heads": {"h": jnp.zeros(N)},
    }


def _data(second=True):
    x0 = jnp.array([1.0, -0.5, 0.3])
    y = _rollout(_true_params(), x0, 20)
    recording = Recording(
        y, jnp.arange(20) * 0.1, np.arange(N), jnp.ones(N), "GCaMP6s", "immobilized", {}
    )
    windows = make_windows(recording, jnp.zeros((20, N)), 10, 2, 10.0)

    def simulate(params):
        return _rollout(params, x0, 20)

    return CurriculumData(
        kinetics=lambda p: (p["m5"]["k"] - 2.0) ** 2,
        windows=windows,
        predict_window=lambda p, s0, w: _rollout(p, s0, w.observed.shape[1]),
        perturbations=lambda p: [(simulate(p).T[:6], y.T[:6])],
        long=(y, simulate),
        states={"a": (y, simulate)},
        second_species=(y, simulate) if second else None,
    )


def test_M10_curriculum_stage_table_and_order():
    assert [s.index for s in STAGES] == list(range(6))
    result = run_curriculum(_init(0), _data(), steps=3, learning_rate=1e-2)
    assert result.completed and result.stopped_at is None
    assert [r["stage"] for r in result.stages] == list(range(6))
    assert result.stages[0]["trained"] == ["m5"]
    assert "peptidergic_heads" not in result.stages[2]["trained"]
    assert result.stages[5]["trained"][-1] == "species_adapters"
    assert all(r["exit_met"] and "history" in r for r in result.stages)
    assert result.stages[2]["terms"]["perturbation"] >= 0


def test_M10_stage_trains_only_its_groups():
    init = _init(0)
    result = run_curriculum(init, _data(), last_stage=0, steps=5, learning_rate=0.1)
    assert float(result.params["m5"]["k"]) != 1.0
    np.testing.assert_array_equal(result.params["m4"]["w"], init["m4"]["w"])


def test_M10_missing_data_stops_with_reason():
    result = run_curriculum(_init(0), _data(second=False), steps=2)
    assert not result.completed and result.stopped_at == 5
    assert result.stop_reason == "missing data: second_species"
    assert len(result.stages) == 6 and result.stages[-1]["exit_met"] is False
    assert all(r["exit_met"] for r in result.stages[:5])


def test_M10_failed_exit_condition_stops_curriculum():
    exits = {1: ExitCondition(max_final_loss=-1e9)}
    result = run_curriculum(_init(0), _data(), steps=2, exits=exits)
    assert result.stopped_at == 1 and "above" in result.stop_reason
    assert [r["stage"] for r in result.stages] == [0, 1]


def test_M10_missing_parameter_group_stops():
    init = _init(0)
    del init["encoder"]
    result = run_curriculum(init, _data(), steps=2)
    assert result.stopped_at == 1 and "encoder" in result.stop_reason


def test_M10_peptidergic_ablation_reported_not_applied():
    result = run_curriculum(
        _init(0), _data(), last_stage=2, steps=2, peptidergic_ablation=True
    )
    ablation = result.stages[2]["ablation"]
    assert ablation["run"] and np.isfinite(ablation["final_loss"])
    np.testing.assert_array_equal(result.params["peptidergic_heads"]["h"], jnp.zeros(N))


def test_M10_ensemble_is_deterministic_per_seed():
    kw = {"last_stage": 1, "steps": 2, "learning_rate": 1e-2}
    a = train_ensemble(_init, _data(), k=2, seed=3, **kw)
    b = train_ensemble(_init, _data(), k=2, seed=3, **kw)
    c = train_ensemble(_init, _data(), k=2, seed=4, **kw)
    wa = [m["result"].params["m4"]["w"] for m in a]
    wb = [m["result"].params["m4"]["w"] for m in b]
    np.testing.assert_array_equal(wa[0], wb[0])
    np.testing.assert_array_equal(wa[1], wb[1])
    assert not np.allclose(wa[0], wa[1])
    assert not np.allclose(wa[0], c[0]["result"].params["m4"]["w"])
    assert [m["seed"] for m in a] == [3, 4]
    assert len(train_ensemble(_init, _data(), k=5, last_stage=0, steps=1)) == 5


def test_M10_coverage_on_constructed_arrays():
    assert nominal_coverage(5) == pytest.approx(2 / 3)
    preds = np.stack([np.zeros((4, 3)), np.ones((4, 3))])
    obs = np.zeros((4, 3))
    obs[0] = 2.0  # neuron 0 outside the range everywhere
    obs[1, :1] = -1.0  # one more outside
    overall, per_class = coverage(preds, obs, classes=np.array([0, 0, 1, 1]))
    assert overall == pytest.approx(8 / 12)
    assert per_class == {0: pytest.approx(2 / 6), 1: pytest.approx(1.0)}
    three = np.stack([preds[0], preds[1], preds[1]])
    check = coverage_check(three, obs)
    assert check["nominal"] == 0.5 and not check["passed"]


def test_M10_coverage_check_passes_near_nominal():
    # K=5: 4 of the 6 equally likely rank positions fall inside the range.
    k, n = 5, 600
    preds = np.stack([np.full((n, 1), float(i)) for i in range(k)])
    obs = (np.tile(np.arange(6), n // 6) - 0.5)[:, None]
    check = coverage_check(preds, obs, classes=np.arange(n) % 2)
    assert check["overall"] == pytest.approx(2 / 3, abs=0.01)
    assert check["passed"]
    assert not coverage_check(preds, obs + 10)["passed"]


def _heldout(shift):
    def predict(params):
        return _rollout(params, jnp.array([1.0, -0.5, 0.3]), 20)

    return predict, np.asarray(predict(_true_params())) + shift


def test_M10_calibration_labels_probe_and_triggers_laplace_fallback():
    members = train_ensemble(_init, _data(), k=3, last_stage=1, steps=2)
    predict, observed = _heldout(100.0)  # far outside the range: coverage 0

    def residual_fn(params):
        return predict(params)[:, :3] - observed[:, :3]

    report = calibrate_ensemble(
        members, predict, observed, classes=np.arange(N), residual_fn=residual_fn
    )
    assert report["label"] == "sensitivity probe"
    assert report["check"]["overall"] == 0.0 and not report["check"]["passed"]
    assert report["fallback"]["components"] == 3
    assert report["fallback"]["check"]["k"] == 3
    assert report["calibrated"] is False and "credible" in report["interval"]
    assert calibrate_ensemble(members, predict, observed)["fallback"] is None


def test_M10_no_fallback_when_check_passes():
    members = train_ensemble(_init, _data(), k=3, last_stage=0, steps=1)
    shift = {id(m["result"].params): s for m, s in zip(members, [-1.0, 1.0, 2.0])}

    def predict(params):
        return jnp.zeros((4, 2)) + shift[id(params)]

    observed = np.zeros((4, 2))
    observed[2:] = 5.0  # half inside [-1, 2]; nominal for k=3 is 0.5
    report = calibrate_ensemble(members, predict, observed, residual_fn=lambda p: 0)
    assert report["check"]["overall"] == 0.5 and report["check"]["passed"]
    assert report["fallback"] is None and report["calibrated"]
    assert report["label"].startswith("calibrated")


def test_M10_unusable_members_excluded():
    members = train_ensemble(
        _init, CurriculumData(kinetics=_data().kinetics), k=2, steps=1
    )
    assert not any(m["usable"] for m in members)
    with pytest.raises(ValueError):
        calibrate_ensemble(members, lambda p: None, np.zeros((1, 1)))


def _nuisance_problem():
    rng = np.random.default_rng(0)
    gain_types = np.array([0, 0, 0, 1, 1, 1])
    drive_types = np.array([0, 0, 1, 1])
    log_gain = np.array([0.4, -0.1, -0.3, 0.2, -0.5, 0.3])
    log_drive = np.array([0.3, -0.3, 0.5, -0.5])
    r = jnp.asarray(rng.uniform(0.5, 1.5, size=(6, 4)))
    t = np.linspace(0, 6, 30)
    u = jnp.asarray([np.sin((j + 1) * t) + 1.5 for j in range(4)])  # [4,L]

    def forward(ld, lg):
        return jnp.exp(lg)[:, None] * ((r * jnp.exp(ld)[None]) @ u)

    y = forward(jnp.asarray(log_drive), jnp.asarray(log_gain))
    recording = Recording(
        y, jnp.arange(30) * 0.1, np.arange(6), jnp.ones(6), "GCaMP6s", "immobilized", {}
    )
    return recording, forward, gain_types, drive_types, log_gain, log_drive


def test_M9_fit_recovers_planted_drives_and_gains_up_to_gauge():
    recording, forward, gain_types, drive_types, log_gain, log_drive = (
        _nuisance_problem()
    )
    fit = fit_animal_nuisance(
        recording, forward, gain_types, drive_types, steps=1500, noise_sd=0.1
    )
    assert fit["final_loss"] < fit["initial_loss"]
    np.testing.assert_allclose(fit["log_drive"], log_drive, atol=0.05)
    np.testing.assert_allclose(fit["log_gain"], log_gain, atol=0.05)
    assert fit["gauge_residual"]["drive"] < 1e-6
    assert fit["gauge_residual"]["gain"] < 1e-6
    for ids, values in ((gain_types, fit["log_gain"]), (drive_types, fit["log_drive"])):
        for t in np.unique(ids):
            assert abs(float(values[ids == t].mean())) < 1e-6


def test_M9_measured_entries_held_exactly_and_gauge_applies_to_rest():
    recording, forward, gain_types, drive_types, *_ = _nuisance_problem()
    measured = np.full(4, np.nan)
    measured[0] = 0.3
    fit = fit_animal_nuisance(
        recording,
        forward,
        gain_types,
        drive_types,
        measured_log_drive=measured,
        steps=5,
    )
    assert float(fit["log_drive"][0]) == 0.3
    assert abs(float(fit["log_drive"][1])) < 1e-6  # lone free entry of type 0
    assert fit["gauge_residual"]["drive"] < 1e-6


def test_M9_fit_without_gain_leaves_gains_at_zero():
    recording, forward, gain_types, drive_types, *_ = _nuisance_problem()
    fit = fit_animal_nuisance(
        recording, forward, gain_types, drive_types, fit_gain=False, steps=20
    )
    np.testing.assert_array_equal(fit["log_gain"], np.zeros(6))
    with pytest.raises(ValueError):
        fit_animal_nuisance(recording, forward, gain_types[:3], drive_types)
