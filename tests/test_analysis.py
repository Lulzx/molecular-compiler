import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler import (
    Candidate,
    CandidateCatalog,
    RuleEnsemble,
    RuleNetwork,
    propose_experiments,
)
from molecular_compiler.experiments import calibration_check
from molecular_compiler.kinetics import (
    KineticRecord,
    KineticsLibrary,
    binding_step,
    markov_step,
    nernst,
    rush_larsen,
)
from molecular_compiler.observation import Recording, fix_gauge
from molecular_compiler.phase0 import (
    Phase0Thresholds,
    feasibility_report,
    freeze_registration,
    gauge_audit,
    load_registration,
    molecular_design,
    rank_analysis,
    split_diagnostics,
)
from molecular_compiler.surrogates import fit_surrogate, validate_rollout
from molecular_compiler.training import (
    energy_distance,
    laplace_samples,
    make_windows,
    mmd,
    train,
)


def test_M4_R1_budget_enforced_after_gauge_audit():
    with pytest.raises(ValueError, match="gauge"):
        RuleNetwork.initialize(rank=100)
    with pytest.raises(ValueError, match="budget"):
        RuleNetwork.initialize(rank=100, gauge_audited=True)
    small = RuleNetwork.initialize(d_z=1, plm_dim=1, rank=100, gauge_audited=True)
    assert small.parameter_count <= small.budget


def test_C_R2_parameter_count_independent_of_N():
    a, b = RuleNetwork.initialize(seed=0), RuleNetwork.initialize(seed=99)
    assert a.parameter_count == b.parameter_count


def test_M5_R1_family_fallback_inflates_covariance():
    source = KineticRecord(
        "known",
        "HH",
        (1.0,),
        ((1.0,),),
        {},
        "fixture",
        family="channel",
        embedding=(0.0, 1.0),
        reversal_mV=0.0,
    )
    fallback = KineticsLibrary({"known": source}).ensure(
        "new", np.array([0.1, 1.0]), "channel"
    )
    assert fallback.fallback_from == "known"
    np.testing.assert_allclose(fallback.params_prior_cov, [[4.0]])
    with pytest.raises(ValueError, match="prior"):
        KineticsLibrary({}).ensure("new", np.array([0.1, 1.0]))


def test_M5_R2_prior_deviations_and_q10():
    r = KineticRecord(
        "known", "HH", (1.0,), ((1.0,),), {}, "fixture", params=(5.0,), reversal_mV=0.0
    )
    report = KineticsLibrary({"known": r}).deviation_report()[0]
    assert report["flagged"] and report["mahalanobis_squared"] == 16.0
    assert r.time_constant(30) == r.tau_s / 2


def test_kinetic_integrators_and_chloride_reversal():
    assert nernst(120.0, 5.0, charge=-1) < nernst(120.0, 40.0, charge=-1)
    assert 0 < binding_step(0.0, 10.0, 1.0, 0.1, 0.01) < 1
    generator = jnp.array([[-1.0, 1.0], [2.0, -2.0]])
    probabilities = markov_step(jnp.array([1.0, 0.0]), generator, 1.0)
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(), 1)
    np.testing.assert_allclose(rush_larsen(0.0, 1.0, 1.0, 1.0), 1 - np.exp(-1))


def test_M9_R2_gauge_fixes_each_type():
    centered = fix_gauge(jnp.array([1.0, 3.0, 2.0, 4.0]), np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(np.exp(centered[:2]).prod(), 1)
    np.testing.assert_allclose(np.exp(centered[2:]).prod(), 1)
    measured = fix_gauge(
        jnp.array([1.0, 3.0]), np.array([0, 0]), measured=[True, False]
    )
    np.testing.assert_allclose(measured, [1.0, 0.0])


def test_M1_R3_low_confidence_retained_but_not_trained():
    recording = Recording(
        jnp.ones((2, 4)),
        jnp.arange(4.0),
        np.arange(2),
        np.array([1.0, 0.5]),
        "GCaMP6s",
        "immobilized",
        {},
    )
    recording.validate()
    assert list(recording.training_mask()) == [True, False]
    assert recording.y.shape[0] == 2


def test_M7_cascade_fits_fixed_ode_on_heldout():
    rng = np.random.default_rng(0)
    states, inputs = rng.normal(size=(40, 3)), rng.normal(size=(40, 6))
    coefficient = rng.normal(size=(10, 3))
    derivative = np.column_stack([np.ones(40), states, inputs]) @ coefficient
    candidate = fit_surrogate(
        states[:30],
        inputs[:30],
        derivative[:30],
        (states[30:], inputs[30:], derivative[30:]),
        tolerance=1e-4,
        conditioned=True,
    )
    assert candidate.family == "fixed_ode" and candidate.conditioned
    assert not candidate.inside(jnp.full((1, 6), 100))[0]
    invalid = validate_rollout(candidate, np.ones((3, 3)), np.zeros((3, 3)), 0.1)
    assert invalid.family == "full"


def test_distribution_losses_and_gradients():
    a, b = jnp.arange(12.0).reshape(4, 3), jnp.arange(12.0).reshape(4, 3) + 1
    np.testing.assert_allclose(energy_distance(a, a), 0, atol=1e-10)
    np.testing.assert_allclose(mmd(a, a), 0, atol=1e-10)
    assert energy_distance(a, b) > 0
    assert np.all(np.isfinite(jax.grad(lambda x: energy_distance(x, b))(a)))


def test_AdamW_training_decreases_loss():
    final, history = train(
        {"x": jnp.array([3.0])},
        lambda p, _: (p["x"] ** 2).sum(),
        steps=20,
        learning_rate=0.1,
    )
    assert abs(final["x"][0]) < 3
    assert history[-1]["loss"] < history[0]["loss"]


def test_multiple_shooting_rejects_long_windows():
    recording = Recording(
        jnp.ones((2, 20)),
        jnp.arange(20.0) * 0.01,
        np.arange(2),
        np.ones(2),
        "GCaMP6s",
        "immobilized",
        {},
    )
    with pytest.raises(ValueError, match="Lyapunov"):
        make_windows(recording, jnp.zeros((20, 2)), 10, 2, 0.05)
    assert len(make_windows(recording, jnp.zeros((20, 2)), 3, 2, 0.05)) == 6


def test_Laplace_head_weight_fallback():
    draws = laplace_samples({"x": jnp.array([1.0, 2.0])}, lambda p: p["x"] - 1, count=3)
    assert len(draws) == 3
    assert all(draw["x"].shape == (2,) for draw in draws)


def test_M11_priority_budget_and_sensitivity_label():
    ensemble = RuleEnsemble((RuleNetwork.initialize(), RuleNetwork.initialize(seed=1)))
    catalog = CandidateCatalog(
        (
            Candidate(
                "sign",
                "mutant",
                1.0,
                "K3",
                ("chloride",),
                [[0], [100]],
                availability_source="public strain catalog",
            ),
            Candidate(
                "peptide",
                "mutant",
                1.0,
                "K2",
                ("release",),
                [[0], [1]],
                availability_source="public strain catalog",
            ),
            Candidate(
                "unavailable",
                "mutant",
                1.0,
                "K2",
                ("release",),
                [[0], [1]],
                available=False,
            ),
        )
    )
    proposals = propose_experiments(ensemble, catalog, 1)
    assert [p.candidate_id for p in proposals] == ["peptide"]
    assert proposals[0].score_label == "sensitivity_proxy"
    assert proposals[0].rule_directions


def test_calibration_requires_noise_and_checks_each_class():
    predictions = np.array([np.full((2, 3), i) for i in range(5)])
    observations = np.array([[2.0, 2.0, 8.0], [2.0, 2.0, 8.0]])
    report = calibration_check(
        predictions, observations, ["A", "B"], includes_observation_noise=True
    )
    assert report["passed"] and report["target"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="noise"):
        calibration_check(predictions, observations, ["A", "B"])


def test_K1_rank_and_factorized_design(system):
    result = rank_analysis(np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert result["rank_epsilon"] == 1
    assert result["participation_ratio"] == pytest.approx(1)
    molecular = molecular_design(system[0])
    assert molecular["implicit_peptide_rows"] == 4
    assert molecular["gram"].shape == (17, 17)


def test_K4_novel_gene_and_hull_diagnostics():
    result = split_diagnostics(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], [[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]]
    )
    assert result["novel_gene_fraction"] == pytest.approx(1 / 3)
    assert result["heldout_profiles"][0]["interpretation"] == "interpolation"
    assert result["heldout_profiles"][1]["interpretation"] == "extrapolation"


def test_phase0_registration_is_immutable(tmp_path):
    path = tmp_path / "split.json"
    digest = freeze_registration(path, {"heldout": ["A"]})
    assert load_registration(path)["processing_hash"] == digest
    with pytest.raises(FileExistsError):
        freeze_registration(path, {"heldout": ["B"]})
    path.write_text(path.read_text().replace('"A"', '"B"'))
    with pytest.raises(ValueError, match="hash"):
        load_registration(path)


def test_missing_gauge_or_synthetic_data_cannot_clear_phase0():
    gauge = gauge_audit([[1, 2], [2, 3]])
    assert not gauge["complete"]
    thresholds = Phase0Thresholds(
        1, 1, 1, 0.1, 12000, 0.0, {"perturbation_state": -1.0}
    )
    diagnostics = split_diagnostics([[0.0, 0.0], [1.0, 1.0]], [[0.5, 0.5]])
    report = feasibility_report(
        rank_analysis(np.eye(2)), diagnostics, gauge, 0.0, (1.0, 2.0), thresholds
    )
    assert report["status"] != "passed"


def test_low_confidence_neurons_do_not_affect_any_nonperturbation_loss():
    from molecular_compiler.training import objective

    prediction = jnp.ones((2, 5))
    observed = jnp.ones((2, 5)).at[1].set(100.0)
    a, _ = objective(prediction, observed, mask=jnp.array([1.0, 0.0]))
    b, _ = objective(prediction, prediction, mask=jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(a, b)


def test_resolution_freezes_only_after_both_checks():
    from molecular_compiler import ResolutionPolicy
    from molecular_compiler.benchmark import resolution_convergence

    policy, report = resolution_convergence(
        ResolutionPolicy(), lambda p: {"metric": 1 + p.dt_s}, {"metric": 1.0}
    )
    assert report["passed"] and report["frozen"]
    assert policy.n_comp == 3
    with pytest.raises(RuntimeError):
        resolution_convergence(
            ResolutionPolicy(),
            lambda p: {"metric": 1.0 / p.dt_s + p.n_comp},
            {"metric": 1.0},
            max_attempts=1,
        )


def test_absolute_capacity_budget_does_not_treat_input_rank_as_a_bound():
    dense = RuleNetwork.initialize(max_parameters=12000, gauge_audited=True)
    assert dense.parameter_count <= dense.budget == 12000
    compact = RuleNetwork.initialize_compact(max_parameters=2, gauge_audited=True)
    assert compact.parameter_count == compact.budget == 2
    with pytest.raises(ValueError, match="gauge"):
        RuleNetwork.initialize(max_parameters=12000)
    with pytest.raises(ValueError, match="budget"):
        RuleNetwork.initialize(max_parameters=2, gauge_audited=True)
