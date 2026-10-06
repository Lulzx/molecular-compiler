from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from molecular_compiler.compiler import ResolutionPolicy
from molecular_compiler.kinetics import KineticRecord
from molecular_compiler.randi_traces import flag_events, match_label
from molecular_compiler.rules import RuleNetwork
from molecular_compiler.worm_kinetics import (
    UNMEASURED_COVARIANCE_SCALE,
    build_worm_library,
    kinetic_family,
)

NEURONS = ["AVAL", "AVAR", "AVDL", "AVDR", "AWCL", "AWCR", "DB1", "DB2", "IL1VL"]


def test_M1_randi_labels_match_with_declared_confidences():
    assert match_label("AVAL", NEURONS) == ("AVAL", 1.0)
    assert match_label("AVAL?", NEURONS) == ("AVAL", 0.5)
    assert match_label("AVD", NEURONS) == ("AVDL", 0.5)
    assert match_label("DB", NEURONS) == ("DB1", 0.5)
    assert match_label("AWCON", NEURONS) == ("AWCL", 0.5)
    assert match_label("glia", NEURONS) == (None, 0.0)
    assert match_label("", NEURONS) == (None, 0.0)


def test_M1_randi_event_flags_reproduce_atlas_inclusion_and_outliers():
    responses = {
        "stimulated": np.array([0, 0, 1, 1]),
        "responder": np.array([0, 1, 1, 0]),
        "event": np.array([0, 0, 1, 1]),
        "mean_dff": np.array([0.8, 6.0, 0.3, 0.1]),
    }
    atlas = {"trials": [[np.array([0.8]), np.array([])], [np.array([]), np.array([])]]}
    flags = flag_events(responses, atlas)
    np.testing.assert_array_equal(flags["atlas_included"], [True, True, False, False])
    np.testing.assert_array_equal(flags["outlier"], [False, True, False, False])


def test_M5_curated_families_route_by_function_not_class():
    assert kinetic_family("glc-3", "receptor") == "GluCl_anion"
    assert kinetic_family("acr-16", "receptor") == "nAChR_cation"
    assert kinetic_family("exp-1", "receptor") == "GABA_cation"
    assert kinetic_family("unc-49", "receptor") == "GABA_anion"
    assert kinetic_family("egl-19", "channel") == "CaV1"
    assert kinetic_family("npr-1", "gpcr") == "gpcr"


def _graph(names, classes):
    genes = [
        {"gene_id": f"WB{i}", "molecule_class": c, "plm_embedding": [float(i), 1.0]}
        for i, c in enumerate(classes)
    ]
    table = SimpleNamespace(to_pylist=lambda: genes)
    return (
        SimpleNamespace(molecular=SimpleNamespace(genes=table)),
        {f"WB{i}": n for i, n in enumerate(names)},
    )


def test_M5_curated_library_flags_unmeasured_and_inflates_covariance():
    graph, names = _graph(
        ["egl-19", "twk-18", "glc-3", "unc-49", "kcc-2", "inx-1", "flp-1", "gpa-1"],
        [
            "channel",
            "channel",
            "receptor",
            "receptor",
            "transporter",
            "innexin",
            "peptide",
            "effector",
        ],
    )
    library = build_worm_library(graph, names, metadata={})
    assert set(library.records) == {f"WB{i}" for i in range(6)}
    cav, k2p = library.records["WB0"], library.records["WB1"]
    assert cav.source.startswith("measured") and cav.h_v_half_mV is not None
    assert k2p.source.startswith("UNMEASURED")
    assert library.records["WB2"].ion_selectivity == {"Cl": 1.0}
    assert library.records["WB2"].reversal_mV is None
    assert library.records["WB4"].transport_equilibrium_mM == {"Cl": 5.0}
    measured_tau = np.asarray(library.records["WB3"].params_prior_cov)[0, 0]
    unmeasured_tau = np.asarray(library.records["WB2"].params_prior_cov)[0, 0]
    tau3, tau2 = library.records["WB3"].tau_s, library.records["WB2"].tau_s
    assert unmeasured_tau / (0.5 * tau2) ** 2 == pytest.approx(
        UNMEASURED_COVARIANCE_SCALE * measured_tau / (0.5 * tau3) ** 2
    )
    assert library.family_for("WB0", "channel") == "CaV1"


def _inactivating(**changes):
    values = {
        "molecule_id": "cav",
        "model_form": "HH",
        "params_prior_mean": (1.0,),
        "params_prior_cov": ((1.0,),),
        "ion_selectivity": {"Ca": 1.0},
        "source": "test",
        "v_half_mV": -40.0,
        "slope_mV": 5.0,
        "tau_s": 0.002,
        "h_v_half_mV": -60.0,
        "h_slope_mV": 6.0,
        "h_tau_s": 0.05,
        "m_power": 2,
    }
    return KineticRecord(**{**values, **changes})


def test_M5_inactivation_gate_steady_state_and_relaxation():
    record = _inactivating()
    assert record.state_size == 2
    voltage = jnp.array([-80.0, -50.0, -20.0])
    gates = record.initial_gates(voltage, 2)
    m = 1 / (1 + np.exp(-(np.asarray(voltage) + 40) / 5))
    h = 1 / (1 + np.exp((np.asarray(voltage) + 60) / 6))
    np.testing.assert_allclose(record.open_probability(gates), m**2 * h, rtol=1e-5)
    stepped = gates
    for _ in range(400):
        stepped = record.step_gates(jnp.full(3, -20.0), stepped, 0.001, 20)
    expected = record.initial_gates(jnp.full(3, -20.0), 2)
    np.testing.assert_allclose(stepped, expected, atol=1e-3)
    with pytest.raises(ValueError):
        _inactivating(model_form="ligand_gated")
    with pytest.raises(ValueError):
        _inactivating(h_slope_mV=0.0)


def test_M4_R1_partial_rules_respect_absolute_budget():
    base = RuleNetwork.initialize(seed=0, d_z=2)
    trainable = {k: base.params[k] for k in ("density", "context", "gap", "bias")}
    rules = RuleNetwork.initialize_partial(base.params, trainable, 13)
    assert rules.parameter_count == 9
    changed = rules.with_params({**rules.params, "bias": jnp.array([2.0])})
    assert float(changed.weights["bias"][0]) == 2.0
    np.testing.assert_array_equal(
        changed.weights["projection"], base.params["projection"]
    )
    with pytest.raises(ValueError):
        RuleNetwork.initialize_partial(base.params, trainable, 8)
    with pytest.raises(ValueError):
        RuleNetwork.initialize_partial(base.params, {"unknown": jnp.zeros(1)}, 13)


def test_M7_connectome_unit_conversion_is_validated():
    policy = ResolutionPolicy.default()
    assert policy.synapse_unit_nS == policy.gap_unit_nS == 1.0
    with pytest.raises(ValueError):
        replace(policy, synapse_unit_nS=0.0)
    with pytest.raises(ValueError):
        replace(policy, gap_unit_nS=-1.0)


def test_M10_phase1_fold_training_resumes_identically(tmp_path, monkeypatch):
    import molecular_compiler.phase1_worm as p1

    n = 6
    rng = np.random.default_rng(0)
    target = rng.normal(size=(n, n))
    weights = (rng.random((n, n)) < 0.7).astype(float)

    def predict(params, nuisance, columns, auto):
        base = jnp.outer(params["a"], params["b"])[:, columns]
        return jnp.exp(nuisance["log_gain"]) * base * auto[columns][None, :]

    params = {"a": jnp.ones(n), "b": jnp.linspace(0.5, 1.5, n)}
    auto = np.ones(n)
    args = (predict, params, target, weights, auto, 4, 2)
    full, full_history = p1.train_fold(*args)

    class Stop(Exception):
        pass

    original = p1._save_checkpoint

    def save_then_stop(path, data):
        original(path, data)
        if len(data["history"]) == 2:
            raise Stop

    path = tmp_path / "fold.ckpt"
    monkeypatch.setattr(p1, "_save_checkpoint", save_then_stop)
    with pytest.raises(Stop):
        p1.train_fold(*args, checkpoint=path)
    monkeypatch.setattr(p1, "_save_checkpoint", original)
    resumed, history = p1.train_fold(*args, checkpoint=path)
    assert [h["loss"] for h in history] == pytest.approx(
        [h["loss"] for h in full_history]
    )
    np.testing.assert_allclose(resumed["rules"]["a"], full["rules"]["a"])
    with pytest.raises(ValueError):
        p1.train_fold(predict, params, target, weights, auto, 5, 2, checkpoint=path)


def test_S12_1_family_recovery_on_curated_library():
    from molecular_compiler.worm_kinetics import family_recovery

    graph, names = _graph(
        ["glc-1", "glc-2", "acr-2", "acr-3", "egl-19"],
        ["receptor"] * 4 + ["channel"],
    )
    genes = graph.molecular.genes.to_pylist()
    for gene, vector in zip(genes, ([0, 0], [0, 1], [10, 0], [10, 1], [5, 5])):
        gene["plm_embedding"] = vector
    result = family_recovery(build_worm_library(graph, names, metadata={}))
    assert result["rate"] == 1.0 and result["passed"]
    assert result["singleton_families"] == ["CaV1"]
    assert result["per_form"] == {"ligand_gated": 1.0}
    genes[1]["plm_embedding"] = [10, 0.5]
    result = family_recovery(build_worm_library(graph, names, metadata={}))
    assert result["per_family"]["GluCl_anion"]["recovered"] == 0
    assert not result["passed"]


def test_M5_neutral_transporters_follow_the_basal_chloride():
    graph, names = _graph(["eat-4", "kcc-2"], ["transporter", "transporter"])
    for basal in (5.0, 12.0):
        library = build_worm_library(graph, names, {}, basal_chloride_mM=basal)
        assert library.records["WB0"].transport_equilibrium_mM == {"Cl": basal}
        assert library.records["WB1"].transport_equilibrium_mM == {"Cl": 5.0}


def test_M10_phase1_residual_report_flags_unused_structure():
    from molecular_compiler import linear_response as lr
    from molecular_compiler.phase1_worm import residual_report

    n = 30
    rng = np.random.default_rng(0)
    chemical = (rng.random((n, n)) < 0.3).astype(float)
    inputs = lr.AtlasInputs(
        chemical=chemical,
        gap=np.zeros((n, n)),
        expression=rng.random((n, 4)),
        tokens=np.zeros((4, 2)),
        releases=np.zeros((n, 3)),
        receptor_ligand=np.full(4, -1),
        innexins=np.array([], dtype=int),
        pair_peptide=np.array([0]),
        pair_receptor=np.array([1]),
        pair_potency=np.array([1.0]),
        transmitter_sign=np.zeros(n),
        identity=np.zeros((n, 2)),
    )
    pairs = np.array([(i, j) for i in range(n) for j in range(n) if i != j])
    unwired = chemical[pairs[:, 0], pairs[:, 1]] == 0
    observed = np.where(unwired, 1.0, 0.0) + rng.normal(0, 0.1, len(pairs))
    rows = residual_report(inputs, np.ones(n), pairs, observed, np.zeros(len(pairs)))
    by_name = {row["feature"]: row for row in rows}
    assert by_name["no_direct_connection"]["candidate_addition"] is True
    assert by_name["peptide_coupling"]["candidate_addition"] is False
    assert "log_autoresponse" not in by_name  # constant feature is skipped


def test_M12_pair_matrix_aggregates_trials_by_responder_and_stimulated():
    from molecular_compiler.phase1_worm import pair_matrix

    responses = {
        "animal": np.array([1, 1, 2, 2]),
        "responder": np.array([0, 0, 1, 0]),
        "stimulated": np.array([1, 1, 0, 1]),
        "mean_dff": np.array([1.0, 3.0, 5.0, 2.0]),
    }
    mean, count, trials = pair_matrix(responses, 2)
    assert mean[0, 1] == 2.0 and count[0, 1] == 3 and np.isnan(mean[0, 0])
    np.testing.assert_array_equal(np.sort(trials[0][1]), [1.0, 2.0, 3.0])
    mean, count, _ = pair_matrix(responses, 2, rows=np.array([2, 3]))
    assert mean[1, 0] == 5.0 and mean[0, 1] == 2.0 and count[0, 1] == 1
