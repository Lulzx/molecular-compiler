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
