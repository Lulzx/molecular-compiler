from dataclasses import replace

import numpy as np
import pytest

from molecular_compiler import (
    EvaluationCase,
    ObservationModel,
    ResolutionPolicy,
    RuleEnsemble,
    Split,
    Stimulus,
    compile,
    evaluate,
    observe,
    simulate,
)
from molecular_compiler.baselines import BlackBoxKernel, connectome_only, fit_linear
from molecular_compiler.evaluation import (
    auroc,
    paired_bootstrap,
    score_recordings,
    split_half_ceiling,
)


def test_paired_bootstrap_and_AUROC():
    result = paired_bootstrap([1.0, 2.0, 3.0])
    assert result["interval"][0] > 0
    assert auroc([False, True, False, True], [0, 1, 0, 1]) == 1
    assert auroc([True, True], [0, 1]) is None
    assert auroc([False, True], [1, 1]) == 0.5


def test_noise_ceiling_requires_repeat_evidence():
    with pytest.raises(ValueError, match="four"):
        split_half_ceiling(np.ones((2, 3, 4)))
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(3, 20))
    assert split_half_ceiling(truth[None] + rng.normal(0, 0.01, (6, 3, 20))) > 0.99


def test_unknown_noise_ceiling_metrics_unavailable():
    values = np.arange(12.0).reshape(3, 4)
    scored = score_recordings(values, values, {}, 0.1)
    assert all(value is None for value in scored["metrics"].values())
    assert scored["raw_metrics"]["perturbation_latency"] == 0
    assert scored["metric_units"]["perturbation_latency"] == "s"


def test_B2_B5_linear_fits_and_B1_training(system):
    graph, _, _ = system
    y = np.array([np.exp(-0.2 * np.arange(20)) * i for i in range(1, 5)])
    b2, b5 = fit_linear(y, 0.1), fit_linear(y, 0.1, graph)
    assert b2.baseline_id == "B2" and b5.baseline_id == "B5"
    assert b5.training_hash
    assert b5.weights[3, 0] == 0
    predicted = b2.predict(y[:, 0], np.zeros((5, 4)), 0.1)
    assert predicted.shape == (4, 5)
    assert connectome_only(graph).baseline_id == "B0"
    kernel = BlackBoxKernel.initialize(2)
    _, history = kernel.fit(np.ones((5, 2)), np.ones(5), steps=5)
    assert history[-1]["loss"] < history[0]["loss"]


def test_class_holdout_requires_bilateral_class_members(system):
    graph, _, _ = system
    case = EvaluationCase(
        "case", graph, Stimulus(), 0.002, np.zeros((4, 4)), {}, heldout_neuron_ids=(1,)
    )
    split = Split(
        "leave_class_out",
        (case,),
        ("A",),
        ("B",),
        {"0": "A", "1": "B", "2": "A", "3": "B"},
        "frozen",
        {"audit": True},
        {},
    )
    with pytest.raises(ValueError, match="entire classes"):
        split.validate()


def test_headline_evaluation_reports_missing_empirical_gates(system):
    graph, rules, kinetics = system
    policy = replace(ResolutionPolicy.default(), n_comp=1)
    stimulus = Stimulus()
    observation = ObservationModel(half_saturation=1e-6)
    values = np.asarray(
        observe(
            simulate(
                compile(graph, rules, kinetics, resolution=policy), stimulus, 0.002
            ),
            observation,
        ).y
    )
    case = EvaluationCase(
        "synthetic",
        graph,
        stimulus,
        0.002,
        values,
        {"perturbation_state": 1.0, "perturbation_amplitude": 1.0},
        observation,
        resolution=policy,
        baselines={"B0": values},
        heldout_neuron_ids=(1, 3),
    )
    split = Split(
        "leave_class_out",
        (case,),
        ("A",),
        ("B",),
        {"0": "A", "1": "B", "2": "A", "3": "B"},
        "frozen",
        {"audit": True},
        {"perturbation_state": -1.0},
    )
    report = evaluate(
        RuleEnsemble((rules, rules)),
        kinetics,
        split,
        ["perturbation_state", "perturbation_amplitude"],
    )
    assert not report.accepted
    assert report.metadata["residuals_enabled"] is False
    assert report.metadata["identity_samples"] == 8
    assert report.metadata["uncertainty_label"] == "sensitivity_probe"
    assert any("synthetic data" in reason for reason in report.missing_gates)
    assert any("B2" in reason for reason in report.missing_gates)
    assert report.cases[0]["scored_neuron_ids"] == [1, 3]
    result = report.comparisons["B0:perturbation_state"]
    assert result["n_clusters"] == 1 and result["interval"] is None
    assert report.comparisons["B0:perturbation_amplitude"]["interval"] is None


def test_global_perturbation_metrics_have_paired_intervals():
    from molecular_compiler.evaluation import paired_statistic_bootstrap

    observed = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    reference = observed
    baseline = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    result = paired_statistic_bootstrap(
        reference, baseline, observed, "perturbation_amplitude", 1.0, samples=100
    )
    assert result["interval"][0] > 0


def test_real_split_requires_its_frozen_definition(system, tmp_path):
    graph, _, _ = system
    case = EvaluationCase(
        "case",
        graph,
        Stimulus(),
        0.002,
        np.zeros((4, 4)),
        {},
        heldout_neuron_ids=(1, 3),
    )
    split = Split(
        "leave_class_out",
        (case,),
        ("A",),
        ("B",),
        {"0": "A", "1": "B", "2": "A", "3": "B"},
        "unregistered",
        {"audit": True},
        {},
        real_data=True,
    )
    with pytest.raises(ValueError, match="pre-registration"):
        split.validate()
    split.freeze(tmp_path / "split.json")
    split.validate()
    split.metric_targets["perturbation_state"] = 1
    with pytest.raises(ValueError, match="pre-registration"):
        split.validate()


def test_statistic_bootstrap_respects_class_clusters():
    from molecular_compiler.evaluation import paired_statistic_bootstrap

    observed = np.array([0.0, 1.0, 2.0, 3.0])
    result = paired_statistic_bootstrap(
        observed, -observed, observed, "perturbation_amplitude", 1.0, clusters=["A"] * 4
    )
    assert result["n_clusters"] == 1 and result["interval"] is None
